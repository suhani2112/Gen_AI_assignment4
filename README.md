# Assignment 4: Multimodal Prompt-to-Image Generation

**ETMMGA304 – Generative AI and Its Applications** · MCA, K.R. Mangalam University
· Lab 4 (lab manual pages 13–14)

Generates images with Stable Diffusion v1.5 for two prompts that differ only in style (watercolor vs. pencil sketch), at matched seeds and settings, and checks the safety-checker flags.

**Settings:** stable-diffusion-v1-5/stable-diffusion-v1-5, float16, attention slicing; 512×512, 25 steps, guidance 7.5, seeds 42 and 43.

## Files

```
GenAI-Assignment-4/
├── README.md
├── LAB_RECORD.md        # observation tables, results, viva answers
├── requirements.txt
├── lab4_stable_diffusion.py
├── lab4_stable_diffusion.ipynb
├── data/                # created automatically (datasets)
└── lab_outputs/         # created automatically (results)
```

## Setup

Python 3.10 or newer is required.

```bash
git clone https://github.com/Aman13082001/GenAI-Assignment-4.git
cd GenAI-Assignment-4
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

**Hardware:** **Needs a CUDA GPU** (about 4 GB VRAM in float16). Use a Colab T4 runtime if you don't have one. The first run downloads about 4 GB of model weights.

### CPU or CUDA GPU

On Windows, `pip install -r requirements.txt` installs the CPU build of PyTorch.
For an NVIDIA GPU, install the CUDA build first (choose the CUDA version for
your driver at [pytorch.org](https://pytorch.org/get-started/locally/)):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Check with `python -c "import torch; print(torch.cuda.is_available())"`.

## Run

```bash
python lab4_stable_diffusion.py
```

Outputs:

- `lab_outputs/library_prompt_1_seed_42.png`
- `lab_outputs/library_prompt_1_seed_43.png`
- `lab_outputs/library_prompt_2_seed_42.png`
- `lab_outputs/library_prompt_2_seed_43.png`

## Google Colab

Open [lab4_stable_diffusion.ipynb](lab4_stable_diffusion.ipynb) in Colab (File → Open notebook → GitHub, then paste this repo's URL). Choose **Runtime → Change runtime type → T4 GPU** if you need a GPU, then run the cells in order. Download `lab_outputs/` before the session ends.

## Lab record

The observation tables, expected ranges and viva answers are in
[LAB_RECORD.md](LAB_RECORD.md). Fill the tables in from your own run.
