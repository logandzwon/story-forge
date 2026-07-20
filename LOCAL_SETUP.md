# Local Apple Silicon setup

This checkout has a portable, project-local environment for the lean Story
Forge path, with optional speech, score, and talking-head support:

```text
.sf -> FLUX.1-schnell still -> LTX-Video or Wan motion -> ffmpeg -> MP4
                         +-> Piper narration
                         +-> ACE-Step music
                         +-> Wav2Lip talking head
```

## Install

Requirements: Apple Silicon, `uv`, `git`, `ffmpeg`, a Hugging Face account,
and at least 95 GB of free disk space. Wan accounts for about 35 GB.

```bash
./bin/setup-local
.venv/bin/hf auth login
```

Accept the FLUX model terms before rendering:
https://huggingface.co/black-forest-labs/FLUX.1-schnell

The first render downloads FLUX into `models/huggingface`. LTX model files
live in `models/ltx`; generated assets stay under `work` and `outputs`.

## Verify

```bash
.venv/bin/python bin/sf parse story_forge/examples/test_tiny.sf
.venv/bin/python bin/sf render story_forge/examples/test_tiny.sf
```

The finished test film is `outputs/test_tiny.mp4`.

## Make a narrated movie

The complete example is `story_forge/examples/integrated_demo.sf`:

```bash
.venv/bin/python bin/sf render \
  story_forge/examples/integrated_demo.sf \
  --out outputs/integrated_demo.mp4
```

Its DSL demonstrates all audio and lip-sync declarations:

```text
voice warm: piper/en_US-libritts_r-medium speaker=0 length=1.08
music gentle: ace/soft-cinematic-piano vol=0.22

narrate warm with lipsync=wav2lip:
    line: "Welcome. Your stories can now come alive."
music gentle vol=0.22
```

Piper synthesizes the narration locally. ACE-Step creates an instrumental
score from the music preset and mixes it beneath the voice. Wav2Lip animates
a talking-head inset. Set `STORY_FORGE_LIPSYNC_DRIVER` to a front-facing
portrait, or omit it to use the current scene's character still.

The public Wav2Lip checkpoint is licensed only for personal, research, and
non-commercial use. Obtain commercial rights or substitute a suitably
licensed lip-sync backend before commercial distribution.

## Use Wan motion

Replace a scene's motion declaration with:

```text
motion wan:
    prompt: "gentle cinematic motion, subtle expression"
    duration: 3
```

Wan2.2 I2V A14B uses Q4_K_M HighNoise and LowNoise experts through ComfyUI,
with the matching LightX2V four-step LoRAs. It accepts the generated FLUX
still as its first frame and is the high-quality hero-motion path. For a
quick direct check:

```bash
./bin/make-wan-a14b --i2v work/mayce_and_the_little_star/still_02.png \
  --duration 1 --res 256x256 --label wan-check --seed 123 \
  "subtle natural movement"
```

## Model selection

The default is the official 2B distilled LTX checkpoint, which is suitable
for a 64 GB Apple Silicon machine. To reproduce the author's 13B path, first
download the larger checkpoint:

```bash
.venv/bin/hf download Lightricks/LTX-Video \
  ltxv-13b-0.9.8-distilled.safetensors \
  --local-dir models/ltx
```

Then render with:

```bash
STORY_FORGE_LTX_MODEL=13b \
  .venv/bin/python bin/sf render story_forge/examples/test_tiny.sf
```

The 13B model needs substantially more unified memory. The repository author
validated it on a 128 GB Mac; use 2B on this 64 GB M2 Max unless you explicitly
want to test memory pressure.

## Overrides

- `STORY_FORGE_PYTHON`: Python executable containing inference dependencies.
- `STORY_FORGE_FLUX_BIN`: alternative FLUX still generator.
- `STORY_FORGE_FLUX_MODEL`: alternative Diffusers FLUX model identifier.
- `STORY_FORGE_HF_CACHE`: Hugging Face cache directory.
- `STORY_FORGE_LTX_REPO`: LTX-Video source checkout.
- `STORY_FORGE_LTX_MODEL_DIR`: directory containing LTX checkpoints.
- `STORY_FORGE_LTX_MODEL`: `2b` (default) or `13b`.
- `STORY_FORGE_OUTPUT_DIR`: rendered motion and final output directory.
- `STORY_FORGE_PIPER_BIN`, `STORY_FORGE_PIPER_MODEL`: narration overrides.
- `STORY_FORGE_ACE_BIN`: ACE-Step music generator override.
- `STORY_FORGE_WAN_MODEL_DIR`: Wan2.2 Diffusers model directory.
- `STORY_FORGE_LIPSYNC_DRIVER`: portrait used for the talking-head overlay.
