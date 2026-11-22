"""Lab 4 - Multimodal Prompt-to-Image Generation (Stable Diffusion v1.5).

ETMMGA304 Laboratory Manual, pages 13-14.

Generates 512x512 images for two prompt variants that differ only in the
requested style, at matched seeds (42, 43) and settings, and saves:

    lab_outputs/library_prompt_1_seed_42.png
    lab_outputs/library_prompt_1_seed_43.png
    lab_outputs/library_prompt_2_seed_42.png
    lab_outputs/library_prompt_2_seed_43.png

Requires a CUDA GPU (~4 GB VRAM in float16). Run in a fresh session:
    python lab4_stable_diffusion.py
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline

SEED = 42
OUT_DIR = Path("lab_outputs")
DATA_DIR = Path("data")
REPO = "stable-diffusion-v1-5/stable-diffusion-v1-5"

PROMPTS: list[str] = [
    "A small library with a reading table, wide view, watercolor illustration",
    "A small library with a reading table, wide view, pencil sketch",
]
SEEDS: tuple[int, ...] = (42, 43)
GEN_SETTINGS: dict[str, float | int] = {
    "width": 512,
    "height": 512,
    "num_inference_steps": 25,
    "guidance_scale": 7.5,
}


def setup(seed: int = SEED) -> torch.device:
    """Seed RNGs, create folders and insist on a CUDA device."""
    random.seed(seed)
    torch.manual_seed(seed)
    OUT_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)
    if not torch.cuda.is_available():
        raise SystemExit("Lab 4 needs a CUDA GPU. Select a GPU runtime "
                         "(Colab: Runtime > Change runtime type) and rerun.")
    device = torch.device("cuda")
    print(f"Device: {device} ({torch.cuda.get_device_name(0)}) | Seed: {seed}")
    return device


def load_pipeline(device: torch.device) -> StableDiffusionPipeline:
    pipe = StableDiffusionPipeline.from_pretrained(REPO,
                                                   torch_dtype=torch.float16)
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    return pipe


def generate_variants(pipe: StableDiffusionPipeline,
                      device: torch.device) -> list[Path]:
    """Generate every prompt x seed combination; skip NSFW-flagged images."""
    saved: list[Path] = []
    for i, prompt in enumerate(PROMPTS, 1):
        for seed in SEEDS:
            rng = torch.Generator(device=device).manual_seed(seed)
            result = pipe(prompt, generator=rng, **GEN_SETTINGS)
            flags = result.nsfw_content_detected
            if flags and flags[0]:
                print(f"Prompt {i}, seed {seed}: image flagged. "
                      "Review the classroom prompt.")
                continue
            path = OUT_DIR / f"library_prompt_{i}_seed_{seed}.png"
            result.images[0].save(path)
            saved.append(path)
            print(f"Saved {path}")
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args()
    device = setup()
    pipe = load_pipeline(device)
    saved = generate_variants(pipe, device)
    print(f"\n{len(saved)}/{len(PROMPTS) * len(SEEDS)} images saved. "
          f"Settings: {GEN_SETTINGS}, model: {REPO}")


if __name__ == "__main__":
    main()
