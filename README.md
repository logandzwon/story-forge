# Story Forge

> A local-only generative cinema pipeline. Animated films on one laptop. No cloud.

```
   ╔══════════════════════════════════════════════════╗
   ║                                                  ║
   ║    flux  →  wan/ltx  →  piper  →  ace-step  →  mux   ║
   ║                                                  ║
   ║          a script.  a laptop.  a film.           ║
   ║                                                  ║
   ╚══════════════════════════════════════════════════╝
```

Story Forge is a local generative-cinema pipeline that turns a structured
story into animated scenes with optional narration, instrumental music, and
lip-sync, then assembles them with `ffmpeg`. Model inference runs on local
hardware after the required model files have been downloaded.

This repository is a fork of
[`nicedreamzapp/story-forge`](https://github.com/nicedreamzapp/story-forge),
created by Matt Macosko / Nice Dreamz LLC. The upstream project supplies the
Story Forge DSL, compiler, original rendering pipeline, UI, films, performance
experiments, and manifesto. This fork is by Logan Dzwonkowski / Cloud Coast Labs, LLC and
retains the upstream MIT license and adds a reproducible environment for a
64 GB Apple Silicon Mac, together with the integrations listed below.

---




## Upstream project and manifesto

The following principles and public-first claims come from the upstream
project. They are preserved here as part of its history; they are not claims
that this fork originated that work.

We're not bound by what was taught. We don't accept upstream library defaults as the speed ceiling. We write our own software when the open-source one's wrong, we write our own DSL when JSON's too clumsy, we write our own Metal kernels when the vendor's path is slow.

Cloud companies will tell you AI cinema needs a server farm. It doesn't. It needs a laptop, a script, and somebody willing to read the source.

What the cloud charges $300-$1000 per film for, this pipeline does for the price of electricity. What people paid big data centers to run, the upstream project proved runs on a MacBook Pro on a kitchen table. Its reported public firsts include:

1. LTX 13B distilled 0.9.8 working on Apple Silicon MPS
2. LPIPS-gated speedup harness for Mac video diffusion (CI-style regression gates on render quality)
3. 1-step Wan 2.2 i2v distillation on Apple Silicon — a rank-32 LoRA that collapses 4 denoising steps into 1 (see [`distill/`](distill/))

We make our own rules. We build new things constantly. We make possible what people said wasn't possible. That's the whole point.


---

## Changes from upstream in this fork

This fork adds a portable, project-local implementation of the complete lean
movie path. It removes the original author's machine-specific absolute paths
and keeps source checkouts, Python environments, models, work files, and final
movies underneath the Story Forge checkout.

The currently supported local path is:

```text
.sf -> FLUX.1-schnell still -> Wan2.2 A14B or LTX motion -> ffmpeg -> MP4
                              +-> Piper narration
                              +-> ACE-Step music
                              +-> optional Wav2Lip inset
```

### Added or changed

- Portable repository-relative paths in `bin/sf`, `story_forge/run.py`, the
  motion router, and the LTX wrapper.
- A local FLUX.1-schnell still generator for Apple Silicon MPS.
- A ComfyUI API workflow for Wan2.2 I2V A14B using Q4_K_M HighNoise and
  LowNoise experts plus their matching LightX2V four-step LoRAs.
- Correct Wan temporal sizing: a native five-second render at 16 fps is 81
  frames (`4n+1`), rather than being rounded down to 77 frames.
- An optional Diffusers Wan2.2 TI2V 5B implementation retained for lower-cost
  experiments; `motion wan` now routes to the higher-quality A14B workflow.
- Selectable LTX 2B or 13B distilled 0.9.8 checkpoints. The portable setup
  defaults to 2B; 13B remains opt-in for machines with more unified memory.
- Local Piper narration, ACE-Step 1.5 instrumental score generation, and
  narration-plus-music mixing.
- Local Wav2Lip support with a compatibility patch for current Python,
  NumPy, librosa, OpenCV, and PyTorch.
- Scene-aware lip-sync portraits. When no explicit driver portrait is set,
  Story Forge uses the current scene still instead of a generic adult face.
- A repeatable setup script, integrated narration/music/lip-sync example,
  additional tests, and Git ignores for all large/generated local assets.

### Retained from upstream

- The indentation-aware `.sf` DSL, parser, resolver, and story-plan emitter.
- Automatic Wan-versus-LTX scene-routing logic and the legacy CLI/UI
  scaffolding. The referenced `story_pipeline.py` full-render entry point is
  not present in this checkout; `--engine lean` is the supported local path.
- The UI, build dashboard, LPIPS measurement harness, Metal experiments, and
  Wan distillation research.
- The original films and supporting documentation.

### Parsed but not rendered by the lean path

The DSL accepts SFX declarations, global transition directives, and mix-duck
directives. The portable lean renderer currently preserves those fields in
the story plan but does not generate per-scene SFX, apply configurable
sidechain ducking, or create title and credit cards. Those features should not
be assumed when using the default `--engine lean` command.

### Requirements

- Apple Silicon Mac
- `git`, `uv`, `ffmpeg`, and `curl`
- Python 3.12 (created and managed by `uv`)
- A Hugging Face account with the FLUX.1-schnell terms accepted
- At least 95 GB free disk space for the default complete setup

### Install

```bash
git clone https://github.com/logandzwon/story-forge.git
cd story-forge
./bin/setup-local
.venv/bin/hf auth login
```

Accept the FLUX license before the first still render:
[black-forest-labs/FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell).
The Hugging Face token is stored in the user's Hugging Face cache, outside
this repository, and is not committed by Git.

The setup script installs isolated environments and downloads:

- FLUX.1-schnell on first use.
- Wan2.2 I2V A14B HighNoise and LowNoise Q4_K_M experts.
- Both matching LightX2V four-step LoRAs, UMT5 XXL FP8, and the Wan VAE.
- LTX-Video 2B distilled 0.9.8 and its spatial upscaler.
- Piper LibriTTS_R medium narration voice.
- ACE-Step 1.5 Turbo, its language/embedding models, and VAE.
- Wav2Lip GAN and S3FD face-detection checkpoints.
- ComfyUI and the ComfyUI-GGUF custom node.

Downloaded models, cloned inference repositories, virtual environments,
renders, and work files are deliberately ignored by Git.

### Quickstart

```bash
.venv/bin/python bin/sf parse story_forge/examples/test_tiny.sf
.venv/bin/python bin/sf render story_forge/examples/test_tiny.sf
# output: outputs/test_tiny.mp4
```

`test_tiny.sf` is the smallest end-to-end parser and renderer check. The
complete feature example is
[`story_forge/examples/integrated_demo.sf`](story_forge/examples/integrated_demo.sf).
It demonstrates Piper narration, ACE-Step music, and optional Wav2Lip.

[`mayce_and_the_little_star.sf`](mayce_and_the_little_star.sf) records the
four-scene narrator-only Mayce project used to verify Wan A14B on this Mac. In
spite of its legacy `.sf` filename, this particular file is compiled story-plan
JSON, not DSL source, and cannot be passed to `bin/sf parse` or `bin/sf render`.
It was rendered by loading the plan directly through `story_forge.run`. Its
approved movie used Wan A14B for every scene, narration plus instrumental
music, and no talking-head overlay or character voice-over.

### Model selection

Wan A14B is now the default `motion wan` path. For a direct A14B check:

```bash
./bin/make-wan-a14b \
  --i2v work/example/still_01.png \
  --duration 5 --res 832x480 --label wan-check --seed 123 \
  "subtle natural character movement, cinematic camera motion"
```

LTX defaults to the 2B checkpoint. To use the original author's 13B path,
download the larger checkpoint and opt in explicitly:

```bash
.venv/bin/hf download Lightricks/LTX-Video \
  ltxv-13b-0.9.8-distilled.safetensors \
  --local-dir models/ltx

STORY_FORGE_LTX_MODEL=13b \
  .venv/bin/python bin/sf render story_forge/examples/test_tiny.sf
```

The 13B LTX checkpoint was validated by the upstream author on a 128 GB Mac.
The 2B checkpoint is the safer LTX choice on a 64 GB machine. Wan A14B's two
quantized experts are loaded sequentially by ComfyUI and have been verified
on the 64 GB M2 Max used for this fork, although full clips render slowly.

### Narration, music, and lip-sync

```text
voice warm: piper/en_US-libritts_r-medium speaker=0 length=1.08
music gentle: ace/soft-cinematic-piano vol=0.22

narrate warm:
    line: "A little star fell softly into the garden."
music gentle vol=0.22
```

Add `with lipsync=wav2lip` to a narration declaration to request a Wav2Lip
inset. By default, its face source is that scene's generated still. Set
`STORY_FORGE_LIPSYNC_DRIVER=/absolute/path/to/portrait.png` only when a
specific front-facing portrait is desired.

Wav2Lip's public checkpoint is licensed for personal, research, and
non-commercial use. Obtain appropriate rights or replace the backend before
commercial distribution.

### Current lean-renderer limitations

- Narration is placed at each scene's start and mixed without overlap
  normalization or automatic music ducking.
- The final audio is limited to the stitched visual duration. A narration line
  that runs past its scene, especially the last scene, can be truncated. The
  released Mayce test movie was mastered separately by extending its final
  frame; that extension is not yet generalized in `story_forge.run`.
- SFX declarations, configurable `@mix` behavior, and title/credit generation
  are not implemented in the lean renderer.

### Environment overrides

| Variable | Purpose |
|---|---|
| `STORY_FORGE_PYTHON` | Python executable used by generators |
| `STORY_FORGE_FLUX_BIN` | Alternative FLUX still generator |
| `STORY_FORGE_FLUX_MODEL` | Alternative Diffusers FLUX model ID |
| `STORY_FORGE_HF_CACHE` | Hugging Face model-cache directory |
| `STORY_FORGE_LTX_REPO` | LTX-Video source checkout |
| `STORY_FORGE_LTX_MODEL_DIR` | Directory containing LTX checkpoints |
| `STORY_FORGE_LTX_MODEL` | Select `2b` (default) or `13b` |
| `STORY_FORGE_OUTPUT_DIR` | Motion and final movie output directory |
| `STORY_FORGE_PIPER_BIN` | Piper executable override |
| `STORY_FORGE_PIPER_MODEL` | Piper ONNX voice model override |
| `STORY_FORGE_ACE_PYTHON` | ACE-Step environment Python |
| `STORY_FORGE_ACE_BIN` | Alternative ACE-Step music wrapper |
| `STORY_FORGE_COMFY_PORT` | Local ComfyUI API port (default `8190`) |
| `STORY_FORGE_LIPSYNC_DRIVER` | Explicit lip-sync driver portrait |

For additional setup notes, see [`LOCAL_SETUP.md`](LOCAL_SETUP.md).

---

## Architecture

```
.sf script ──► parser ──► resolver ──► emitter ──► .storyplan.json IR
                                                          │
                                                          ▼
                                                    run.py bridge
                                                          │
                                                          ▼
                                          render-route (per-scene engine pick)
                                              │                       │
                                              ▼                       ▼
                                  make-ltx-lightricks        make-wan-a14b
                                  (LTX 2B/13B distilled)     (Wan2.2 A14B GGUF + ComfyUI)
                                              │                       │
                                              └───────────┬───────────┘
                                                          ▼
                                            Piper narration + ACE-Step music
                                                          │
                                                          ▼
                                                 ffmpeg stitch + mix
                                                          │
                                                          ▼
                                                      finished.mp4
```

- **`render-route`** auto-selects Wan (hero shots with character action / faces / dialogue) or LTX (B-roll / atmosphere / wide shots) per scene based on the motion prompt, or honors an explicit `motion wan:` / `motion ltx:` block in the DSL.
- **Wan A14B** uses dual quantized experts through the local ComfyUI API. The
  checked-in Metal experiment remains documented, but is not part of this
  fork's default Wan route.
- **Piper + ACE-Step** create narration and an instrumental bed, then `ffmpeg`
  places narration at scene boundaries, mixes the score, and stitches scenes.

---

## The DSL grammar

Story Forge films are written as `.sf` scripts — indentation-aware, comment-friendly, stdlib-only parser. The full grammar as of 2026-05-24:

The grammar includes constructs used by the upstream/full pipeline. In the
portable lean renderer, music and lip-sync are implemented, while SFX and
configurable `@mix`/`@transition` behavior are currently parse-only.

```
# Comments start with '#' and go to end of line.

# --- Variables (substituted in any "{$name}" inside a string) ----
$style = "Studio Ghibli watercolor, soft snowfall, golden hour, painterly, 4k"
$child = "a small child in a red hooded cloak, mittens"
$cabin = "a hand-built wooden cabin with warm yellow window light"

# --- Film header (one per file) ----------------------------------
film "Cabin Open" slug=cabin_open target=m5+mini scene_dur=8.5

# --- Voice presets -----------------------------------------------
# voice <name>: <engine>/<model> <kv attrs>
voice warm:   piper/en_US-libritts_r-medium speaker=0 length=1.18
voice gravel: piper/en_US-libritts_r-medium speaker=14 length=1.05
voice child:  piper/en_US-amy-medium length=1.30

# --- Music presets -----------------------------------------------
# music <name>: <engine>/<style-slug> <kv attrs>
music wintry: ace/wintry-soft-piano vol=0.35

# --- SFX presets -------------------------------------------------
# sfx <name>: <engine>/sfx prompt="..." duration=N vol=0.NN
sfx fire_crackle: ace/sfx prompt="fire crackling, warm hearth" duration=8 vol=0.25
sfx wind_low:     ace/sfx prompt="low wind through pines" duration=10 vol=0.20

# --- Global directives -------------------------------------------
@transition xfade dur=0.5
@mix duck voice -> music threshold=-22 ratio=4

# --- Scenes ------------------------------------------------------
scene snow_walk:
    still flux:
        prompt: "{$style}, wide shot of {$child} crossing a snowfield toward {$cabin}"
        seed: auto                    # or an explicit int e.g. seed: 42
    motion wan:                       # or "motion ltx:" for B-roll
        prompt: "gentle handheld push-in, soft falling snow, child takes slow steps"
        duration: 5.0
    narrate warm:                     # full block form
        line: "The snow came down like a hush."
    sfx wind_low at=0.0               # per-scene SFX ref with offset
    music wintry vol=0.30             # per-scene music ref (overrides preset vol)

scene fireside:
    still flux:
        prompt: "{$style}, interior, {$child} unwrapping by a stone fireplace"
        seed: auto
    motion wan:
        prompt: "intimate close shot, firelight flickers, slow zoom to flames"
        duration: 5.0
    narrate warm with lipsync:        # 'with lipsync' flag → drives Wav2Lip
        line: "And the cold outside became a story she would only tell on warm nights."
    sfx fire_crackle at=2.0
    music wintry vol=0.40
```

Constructs at a glance:

| Form | Purpose |
|---|---|
| `# comment` | Line comment, stripped before parse |
| `$name = value` | Variable, interpolated via `{$name}` in any string |
| `film "Title" slug=... target=... scene_dur=...` | Film header (one per file) |
| `voice NAME: piper/model speaker=N length=F` | Define a reusable voice preset |
| `music NAME: ace/style-slug vol=F` | Define a reusable music preset |
| `sfx NAME: ace/sfx prompt="..." duration=N vol=F` | Define a reusable SFX preset |
| `@transition xfade dur=0.5` | Global film-level directive |
| `@mix duck voice -> music threshold=-22 ratio=4` | Global mix directive |
| `scene NAME:` | Scene block (one per cut) |
| `still flux:` + `prompt:` / `seed:` | Per-scene Flux still spec |
| `motion wan:` or `motion ltx:` + `prompt:` / `duration:` | Per-scene i2v motion spec |
| `narrate VOICE:` + `line:` | Narration in this scene |
| `narrate VOICE with lipsync:` + `line:` | Same, but flag for Wav2Lip pass |
| `sfx NAME at=N.N` | Per-scene SFX ref, `at=` is start offset in seconds |
| `music NAME vol=F` | Per-scene music ref, vol overrides preset |

The parser, resolver, and emitter live in [`story_forge/parser.py`](story_forge/parser.py), [`story_forge/resolver.py`](story_forge/resolver.py), and [`story_forge/emitter.py`](story_forge/emitter.py). The AST shape is documented in the parser docstring. Full reference example: [`story_forge/examples/cabin_open.sf`](story_forge/examples/cabin_open.sf).

---

## What's inside the repo

```
story-forge/
├── bin/
│   ├── sf                    # DSL CLI: sf parse / sf render
│   ├── setup-local           # Reproducible local environments + model setup
│   ├── make-flux-still       # FLUX.1-schnell stills on Apple Silicon MPS
│   ├── make-ltx-lightricks   # Selectable LTX 2B/13B distilled wrapper
│   ├── make-wan-a14b         # Wan2.2 A14B dual-expert ComfyUI workflow
│   ├── make-wan-motion       # Optional Wan2.2 TI2V 5B Diffusers path
│   ├── make-music            # ACE-Step 1.5 instrumental generator
│   ├── render-route          # Per-scene engine picker (Wan vs LTX)
│   └── measure-render        # LPIPS-gated speedup harness
│
├── story_forge/
│   ├── parser.py             # Indentation-aware .sf → AST
│   ├── resolver.py           # Variable interpolation + preset resolution
│   ├── emitter.py            # AST → .storyplan.json IR
│   ├── run.py                # IR → render-route + ffmpeg bridge
│   ├── examples/             # test_tiny, cabin, and integrated A/V examples
│   └── tests/                # Parser, resolver, runner, and lip-sync tests
│
├── patches/
│   └── wav2lip-modern.patch  # Modern dependency compatibility patch
├── LOCAL_SETUP.md            # Concise local installation and usage guide
├── mayce_and_the_little_star.sf # Compiled JSON plan (legacy suffix), not DSL
├── metal/
│   ├── flash_attn_mps.py     # 144-line MSL tiled flash-attn kernel
│   ├── verify_flash_attn.py  # PSNR + speedup validator
│   ├── metal_rmsnorm_linear.py / verify_rmsnorm_linear.py
│   └── hello_metal.py        # Minimal compile_shader example
│
├── build_status/             # Live build dashboard (localhost:17602)
├── ui/                       # Story Forge UI v2 — DSL editor + engine toggles (localhost:17600/story)
├── server.py                 # Flask server that hosts the UI + DSL endpoints
├── saga.mp4                  # The first film — The Bear Sister, 4:08
├── STORYBOOK.md              # Full prose transcript of saga.mp4
└── YOUTUBE_METADATA.md       # Tags / description for the YT upload
```

---

## What the portable lean renderer does

You write a `.sf` script and run it through `bin/sf`. The default lean path:

1. Generates a Flux still per scene
2. Animates each still with Wan (hero) or LTX (B-roll), routed automatically per scene
3. Renders each narration line with Piper TTS using the selected speaker and
   length controls
4. Optionally creates one instrumental score with ACE-Step and mixes it under
   the narration at the configured fixed volume
5. Optionally creates Wav2Lip character insets for marked narration lines
6. Composes the film with `ffmpeg` using scene-synced narration,
   `adelay+amix`, fixed half-second scene crossfades, fades, and H.264/AAC muxing

Every step runs locally on Apple Silicon. The output is a regular `.mp4`.

---

## Component stack

| Stage | Tool | Model | Purpose |
|---|---|---|---|
| Still image per scene | [FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell) | ~31 GB cache | Four-step MPS still generation |
| Hero motion (faces / action) | [Wan2.2 I2V A14B GGUF](https://huggingface.co/QuantStack/Wan2.2-I2V-A14B-GGUF) | ~27 GB complete | Dual-expert Q4_K_M ComfyUI workflow with LightX2V LoRAs |
| Optional lower-cost motion | Wan2.2 TI2V 5B | ~32 GB | Diffusers I2V experiment retained outside the default route |
| B-roll motion (atmosphere) | [LTX-Video distilled 0.9.8](https://huggingface.co/Lightricks/LTX-Video) | ~6.4 GB for 2B | 2B default; optional 13B on high-memory Macs |
| Voice narration | [Piper TTS](https://github.com/rhasspy/piper) | LibriTTS_R medium + others | Per-voice presets in DSL |
| Music | [ACE-Step 1.5](https://github.com/ACE-Step/ACE-Step-1.5) | ~9.5 GB checkpoints | Original instrumental score beneath narration |
| Optional lip-sync | [Wav2Lip](https://github.com/Rudrabha/Wav2Lip) | ~225 MB checkpoints | Audio-driven character inset using the scene still by default |
| Compose | [ffmpeg](https://ffmpeg.org/) | — | Scene crossfades, fixed-volume audio mixing, fades, and muxing |

The upstream/full pipeline contains additional concepts and experiments. The
table above describes what is wired into this fork's default lean renderer.

---

## Verification

The focused Story Forge suite currently contains 29 passing parser, resolver,
runner, and lip-sync tests:

```bash
.venv/bin/python -m pytest -q story_forge/tests
```

These tests validate dispatch and composition logic. They do not rerun the
large diffusion, music, or lip-sync models.

---

## License and model terms

The pipeline code retains the upstream MIT license and copyright notice in
[`LICENSE`](LICENSE). The upstream `saga.mp4` has separate CC BY-NC-SA 4.0
terms recorded in that file. Downloaded models and checkpoints retain their
own licenses and acceptable-use terms; review them before redistribution or
commercial use. In particular, the public Wav2Lip checkpoint is restricted to
personal, research, and non-commercial use.
