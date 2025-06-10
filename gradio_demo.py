"""
Gradio Demo for Self-Forcing with real-time video preview.
"""

import os
import time
import argparse
import urllib.request
import tempfile
from PIL import Image
import numpy as np
import torch
from omegaconf import OmegaConf
import gradio as gr
import imageio

from pipeline import CausalInferencePipeline
from demo_utils.constant import ZERO_VAE_CACHE
from demo_utils.vae_block3 import VAEDecoderWrapper
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder
from demo_utils.utils import generate_timestamp
from demo_utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller, move_model_to_device_with_memory_preservation

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_path", type=str, default='./checkpoints/self_forcing_dmd.pt')
parser.add_argument("--config_path", type=str, default='./configs/self_forcing_dmd.yaml')
parser.add_argument('--trt', action='store_true')
args = parser.parse_args()

print(f'Free VRAM {get_cuda_free_memory_gb(gpu)} GB')
low_memory = get_cuda_free_memory_gb(gpu) < 40

# Load models
config = OmegaConf.load(args.config_path)
default_config = OmegaConf.load("configs/default_config.yaml")
config = OmegaConf.merge(default_config, config)

text_encoder = WanTextEncoder()

# Global variables for dynamic model switching
current_vae_decoder = None
current_use_taehv = False
fp8_applied = False
torch_compile_applied = False
generation_active = False

def initialize_vae_decoder(use_taehv=False, use_trt=False):
    """Initialize VAE decoder based on the selected option"""
    global current_vae_decoder, current_use_taehv

    if use_trt:
        from demo_utils.vae import VAETRTWrapper
        current_vae_decoder = VAETRTWrapper()
        return current_vae_decoder

    if use_taehv:
        from demo_utils.taehv import TAEHV
        taehv_checkpoint_path = "checkpoints/taew2_1.pth"
        if not os.path.exists(taehv_checkpoint_path):
            print(f"taew2_1.pth not found in checkpoints folder {taehv_checkpoint_path}. Downloading...")
            os.makedirs("checkpoints", exist_ok=True)
            download_url = "https://github.com/madebyollin/taehv/raw/main/taew2_1.pth"
            try:
                urllib.request.urlretrieve(download_url, taehv_checkpoint_path)
                print(f"Successfully downloaded taew2_1.pth to {taehv_checkpoint_path}")
            except Exception as e:
                print(f"Failed to download taew2_1.pth: {e}")
                raise

        class DotDict(dict):
            __getattr__ = dict.__getitem__
            __setattr__ = dict.__setitem__

        class TAEHVDiffusersWrapper(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dtype = torch.float16
                self.taehv = TAEHV(checkpoint_path=taehv_checkpoint_path).to(self.dtype)
                self.config = DotDict(scaling_factor=1.0)

            def decode(self, latents, return_dict=None):
                return self.taehv.decode_video(latents, parallel=False).mul_(2).sub_(1)

        current_vae_decoder = TAEHVDiffusersWrapper()
    else:
        current_vae_decoder = VAEDecoderWrapper()
        vae_state_dict = torch.load('wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth', map_location="cpu")
        decoder_state_dict = {}
        for key, value in vae_state_dict.items():
            if 'decoder.' in key or 'conv2' in key:
                decoder_state_dict[key] = value
        current_vae_decoder.load_state_dict(decoder_state_dict)

    current_vae_decoder.eval()
    current_vae_decoder.to(dtype=torch.float16)
    current_vae_decoder.requires_grad_(False)
    current_vae_decoder.to(gpu)
    current_use_taehv = use_taehv

    print(f"✅ VAE decoder initialized with {'TAEHV' if use_taehv else 'default VAE'}")
    return current_vae_decoder

def tensor_to_pil(frame_tensor):
    """Convert a single frame tensor to PIL Image."""
    frame = torch.clamp(frame_tensor.float(), -1., 1.) * 127.5 + 127.5
    frame = frame.to(torch.uint8).cpu().numpy()
    if len(frame.shape) == 3:
        frame = np.transpose(frame, (1, 2, 0))
    if frame.shape[2] == 3:
        image = Image.fromarray(frame, 'RGB')
    else:
        image = Image.fromarray(frame)
    return image

def create_video_from_frames(frames, fps=24):
    """Create a temporary video file from PIL Images."""
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
        temp_path = temp_file.name
    
    # Convert PIL Images to numpy arrays
    np_frames = [np.array(frame) for frame in frames]
    
    # Write video using imageio
    writer = imageio.get_writer(temp_path, fps=fps)
    for frame in np_frames:
        writer.append_data(frame)
    writer.close()
    
    return temp_path

def generate_video(prompt, seed, enable_torch_compile, enable_fp8, use_taehv, progress=gr.Progress()):
    """Generate video and yield frames and video updates."""
    global generation_active, current_vae_decoder, current_use_taehv, fp8_applied, torch_compile_applied
    
    generation_active = True
    all_frames = []
    video_update_interval = 5  # Update video every N frames
    temp_video_path = None
    
    try:
        # Handle VAE decoder switching
        if use_taehv != current_use_taehv:
            progress(0.02, desc="Switching VAE decoder...")
            current_vae_decoder = initialize_vae_decoder(use_taehv=use_taehv)
            pipeline.vae = current_vae_decoder

        # Handle FP8 quantization
        if enable_fp8 and not fp8_applied:
            progress(0.03, desc="Applying FP8 quantization...")
            from torchao.quantization.quant_api import quantize_, Float8DynamicActivationFloat8WeightConfig, PerTensor
            quantize_(transformer, Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor()))
            fp8_applied = True

        # Text encoding
        progress(0.08, desc="Encoding text prompt...")
        conditional_dict = text_encoder(text_prompts=[prompt])
        for key, value in conditional_dict.items():
            conditional_dict[key] = value.to(dtype=torch.float16)
        if low_memory:
            gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
            move_model_to_device_with_memory_preservation(
                text_encoder, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

        # Handle torch.compile if enabled
        torch_compile_applied = enable_torch_compile
        if enable_torch_compile and not models_compiled:
            transformer.compile(mode="max-autotune-no-cudagraphs")
            if not current_use_taehv and not low_memory and not args.trt:
                current_vae_decoder.compile(mode="max-autotune-no-cudagraphs")

        # Initialize generation
        progress(0.12, desc="Initializing generation...")
        rnd = torch.Generator(gpu).manual_seed(seed)
        
        pipeline._initialize_kv_cache(batch_size=1, dtype=torch.float16, device=gpu)
        pipeline._initialize_crossattn_cache(batch_size=1, dtype=torch.float16, device=gpu)

        noise = torch.randn([1, 21, 16, 60, 104], device=gpu, dtype=torch.float16, generator=rnd)

        # Generation parameters
        num_blocks = 7
        current_start_frame = 0
        num_input_frames = 0
        all_num_frames = [pipeline.num_frame_per_block] * num_blocks
        if current_use_taehv:
            vae_cache = None
        else:
            vae_cache = ZERO_VAE_CACHE
            for i in range(len(vae_cache)):
                vae_cache[i] = vae_cache[i].to(device=gpu, dtype=torch.float16)

        generation_start_time = time.time()

        progress(0.15, desc="Generating frames...")

        for idx, current_num_frames in enumerate(all_num_frames):
            if not generation_active:
                break

            block_progress = ((idx + 1) / len(all_num_frames)) * 0.8 + 0.15
            progress(block_progress, desc=f"Processing block {idx+1}/{len(all_num_frames)}...")

            noisy_input = noise[:, current_start_frame -
                                num_input_frames:current_start_frame + current_num_frames - num_input_frames]

            # Denoising loop
            for index, current_timestep in enumerate(pipeline.denoising_step_list):
                if not generation_active:
                    break

                timestep = torch.ones([1, current_num_frames], device=noise.device,
                                      dtype=torch.int64) * current_timestep

                if index < len(pipeline.denoising_step_list) - 1:
                    _, denoised_pred = transformer(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=pipeline.kv_cache1,
                        crossattn_cache=pipeline.crossattn_cache,
                        current_start=current_start_frame * pipeline.frame_seq_length
                    )
                    next_timestep = pipeline.denoising_step_list[index + 1]
                    noisy_input = pipeline.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep * torch.ones([1 * current_num_frames], device=noise.device, dtype=torch.long)
                    ).unflatten(0, denoised_pred.shape[:2])
                else:
                    _, denoised_pred = transformer(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=pipeline.kv_cache1,
                        crossattn_cache=pipeline.crossattn_cache,
                        current_start=current_start_frame * pipeline.frame_seq_length
                    )

            if not generation_active:
                break

            # Update KV cache for next block
            if idx != len(all_num_frames) - 1:
                transformer(
                    noisy_image_or_video=denoised_pred,
                    conditional_dict=conditional_dict,
                    timestep=torch.zeros_like(timestep),
                    kv_cache=pipeline.kv_cache1,
                    crossattn_cache=pipeline.crossattn_cache,
                    current_start=current_start_frame * pipeline.frame_seq_length,
                )

            # Decode to pixels
            if args.trt:
                all_current_pixels = []
                for i in range(denoised_pred.shape[1]):
                    is_first_frame = torch.tensor(1.0).cuda().half() if idx == 0 and i == 0 else \
                        torch.tensor(0.0).cuda().half()
                    outputs = vae_decoder.forward(denoised_pred[:, i:i + 1, :, :, :].half(), is_first_frame, *vae_cache)
                    current_pixels, vae_cache = outputs[0], outputs[1:]
                    all_current_pixels.append(current_pixels.clone())
                pixels = torch.cat(all_current_pixels, dim=1)
                if idx == 0:
                    pixels = pixels[:, 3:, :, :, :]
            else:
                if current_use_taehv:
                    if vae_cache is None:
                        vae_cache = denoised_pred
                    else:
                        denoised_pred = torch.cat([vae_cache, denoised_pred], dim=1)
                        vae_cache = denoised_pred[:, -3:, :, :, :]
                    pixels = current_vae_decoder.decode(denoised_pred)
                    if idx == 0:
                        pixels = pixels[:, 3:, :, :, :]
                    else:
                        pixels = pixels[:, 12:, :, :, :]
                else:
                    pixels, vae_cache = current_vae_decoder(denoised_pred.half(), *vae_cache)
                    if idx == 0:
                        pixels = pixels[:, 3:, :, :, :]

            # Convert frames to PIL Images and yield them
            block_frames = pixels.shape[1]
            for frame_idx in range(block_frames):
                if not generation_active:
                    break

                frame_tensor = pixels[0, frame_idx].cpu()
                pil_image = tensor_to_pil(frame_tensor)
                all_frames.append(pil_image)
                
                # Update video preview periodically
                if len(all_frames) % video_update_interval == 0 or frame_idx == block_frames - 1:
                    if temp_video_path and os.path.exists(temp_video_path):
                        os.unlink(temp_video_path)
                    temp_video_path = create_video_from_frames(all_frames)
                    yield {
                        output_gallery: all_frames,
                        output_video: temp_video_path,
                        status: f"Generated {len(all_frames)} frames so far..."
                    }

            current_start_frame += current_num_frames

        # Final update with all frames
        if temp_video_path and os.path.exists(temp_video_path):
            os.unlink(temp_video_path)
        temp_video_path = create_video_from_frames(all_frames)
        yield {
            output_gallery: all_frames,
            output_video: temp_video_path,
            status: f"Generation complete! {len(all_frames)} frames generated"
        }

    except Exception as e:
        raise gr.Error(f"Generation failed: {str(e)}")
    finally:
        generation_active = False
        if temp_video_path and os.path.exists(temp_video_path):
            os.unlink(temp_video_path)

def stop_generation():
    """Stop the current generation process."""
    global generation_active
    generation_active = False
    return "Generation stopped"

# Initialize with default VAE
vae_decoder = initialize_vae_decoder(use_taehv=False, use_trt=args.trt)

transformer = WanDiffusionWrapper(is_causal=True)
state_dict = torch.load(args.checkpoint_path, map_location="cpu")
transformer.load_state_dict(state_dict['generator_ema'])

text_encoder.eval()
transformer.eval()

transformer.to(dtype=torch.float16)
text_encoder.to(dtype=torch.bfloat16)

text_encoder.requires_grad_(False)
transformer.requires_grad_(False)

pipeline = CausalInferencePipeline(
    config,
    device=gpu,
    generator=transformer,
    text_encoder=text_encoder,
    vae=vae_decoder
)

if low_memory:
    DynamicSwapInstaller.install_model(text_encoder, device=gpu)
else:
    text_encoder.to(gpu)
transformer.to(gpu)

# Create Gradio interface
with gr.Blocks(title="Self-Forcing Demo") as demo:
    gr.Markdown("# Self-Forcing Demo")
    gr.Markdown("Generate videos from text prompts using the Self-Forcing model")
    
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt", placeholder="Enter your text prompt here...")
            seed = gr.Number(label="Seed", value=31337, precision=0)
            
            with gr.Accordion("Advanced Options", open=False):
                enable_torch_compile = gr.Checkbox(label="Enable Torch Compile (faster but slower first run)", value=False)
                enable_fp8 = gr.Checkbox(label="Enable FP8 Quantization (reduced memory)", value=False)
                use_taehv = gr.Checkbox(label="Use TAEHV Decoder (alternative VAE)", value=False)
            
            with gr.Row():
                generate_btn = gr.Button("Generate Video", variant="primary")
                stop_btn = gr.Button("Stop Generation")
            
            status = gr.Textbox(label="Status", interactive=False)
            
        with gr.Column():
            output_gallery = gr.Gallery(label="Generated Frames", columns=4, height="auto")
            output_video = gr.Video(label="Video Preview", format="mp4", autoplay=True)
    
    # Event handlers
    generate_event = generate_btn.click(
        fn=generate_video,
        inputs=[prompt, seed, enable_torch_compile, enable_fp8, use_taehv],
        outputs=[output_gallery, output_video, status]
    )
    
    stop_btn.click(
        fn=stop_generation,
        outputs=[status],
        cancels=[generate_event]
    )
    
    # System info
    gr.Markdown(f"### System Info")
    gr.Markdown(f"Free VRAM: {get_cuda_free_memory_gb(gpu):.2f} GB")
    gr.Markdown(f"Low memory mode: {'Yes' if low_memory else 'No'}")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share = True)
