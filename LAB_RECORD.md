# Lab Record: Assignment 4, ETMMGA304 Generative AI and Its Applications

**Student:** ____________________  **Roll no.:** ____________  **Programme:** MCA

Fill in each blank table from **your own run**. The *expected ranges* below
are approximate values for the default settings, to help you spot a broken
run. They are not results to copy. Values change with hardware, library
versions and random seeds.

Write down the following for every experiment:

| Field | Value |
|-------|-------|
| Date | |
| Device (CPU / GPU model) | |
| Python / PyTorch / Transformers versions | |
| Seed | 42 |

---

## Lab 4: Multimodal Prompt-to-Image Generation

**Aim.** Generate images from text with Stable Diffusion, changing one prompt
element (the style) while keeping the seed and settings the same.

**Software / model.** Diffusers, `stable-diffusion-v1-5/stable-diffusion-v1-5`,
float16 on a CUDA GPU with attention slicing. Script:
`lab4_stable_diffusion.py`.

**Parameters.** 512×512, 25 inference steps, guidance scale 7.5, seeds 42
and 43, built-in safety checker enabled.

### Observation table (0 = missing, 1 = partial, 2 = clear)

| Style requested | Seed | Subject score 0–2 | Style score 0–2 | Composition or errors |
|-----------------|------|-------------------|-----------------|-----------------------|
| Watercolor | 42 | | | |
| Watercolor | 43 | | | |
| Pencil sketch | 42 | | | |
| Pencil sketch | 43 | | | |

**Typical pattern.** At the same seed, the two styles often share a similar
layout (camera angle, position of the table and shelves), because the
starting noise sets the coarse structure. Changing the style mainly changes
colour, texture and line quality (soft colour washes compared with
monochrome hatching). Changing the seed changes layout, viewpoint and
object placement. Common errors include warped shelves and book spines,
gibberish text on books, and chairs with impossible geometry.

If you use Replicate or the DALL·E API instead, record the model version,
prompts, seed and settings, and only compare images made with the same
backend.

### Result, application and risk

**Application:** quick concept art and illustrations for teaching material
or storyboards. **Risk:** models trained on web-scraped images can copy
artists' styles without permission and repeat demographic bias (for
example, who appears in a "librarian" image). Photorealistic outputs can
also be passed off as real photographs.

### Viva answers

1. **Why should prompt variants be compared at the same seed?**
   The seed fixes the initial latent noise, which largely sets the
   composition. With the seed and all other settings held constant, the
   prompt is the only thing that changes, so any difference can be credited
   to the wording. Changing both at once would mix the two effects.

2. **What is the purpose of denoising steps?**
   Diffusion models learn to reverse a gradual noising process. At each
   step the U-Net predicts the noise in the current latent, guided by the
   text embedding, and the scheduler removes part of it. Over 25 steps, pure
   noise is refined into a clean latent, which the VAE decoder turns into
   pixels. More steps give finer detail, up to a point, at the cost of speed.

3. **How can you rate subject and style adherence independently?**
   Use separate rubric questions. *Subject:* is there a small library with
   a reading table in a wide view, whatever the medium? *Style:* does the
   rendering look like watercolour or pencil, whatever the content? An image
   can score 2 on subject and 0 on style (a photo-real library), or the
   reverse. Scoring each one blind to the other, or by two raters, reduces
   halo bias.

4. **When should a generated image carry a disclosure?**
   Whenever viewers could mistake it for a real photograph or human-made
   work, or when it is used in news, education, advertising, academic
   submissions, or anything showing real people or places. Disclosure is
   also required wherever the platform, institution or law demands it. A
   visible label, plus provenance metadata such as C2PA, supports
   transparency and helps prevent misinformation.
