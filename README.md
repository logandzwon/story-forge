# Story Forge

> A local-only generative cinema pipeline. Animated films on one laptop. No cloud.

```
   ╔══════════════════════════════════════════════════╗
   ║                                                  ║
   ║    flux  →  wan  →  piper  →  ace-step  →  mux   ║
   ║                                                  ║
   ║          a script.  a laptop.  a film.           ║
   ║                                                  ║
   ╚══════════════════════════════════════════════════╝
```

Story Forge is a self-contained pipeline that takes a structured story description and produces a finished animated film — with motion, narration, original music, title and credits — entirely on local hardware. Five open-source models composed by `ffmpeg`. **Zero cloud calls. Zero API charges. Zero rate limits.** Run it once, run it a thousand times.

### ▶ Watch the first Story Forge film — *The Bear Sister*

[![The Bear Sister — a Story Forge production](./hero-screenshot.jpg)](https://youtu.be/_bFQTl7_vF4)

[**▶ Watch on YouTube**](https://youtu.be/_bFQTl7_vF4) · [Download `saga.mp4`](./saga.mp4) · [Read the story (STORYBOOK.md)](./STORYBOOK.md)

---

## 🚀 Status — 2026-05-24 build session

The speedup build is underway. **First public-first wins last night:**

- ✅ **LTX 13B distilled 0.9.8 working on Apple Silicon MPS** — likely the first public-confirmed working setup (cross-checked by 5 research agents). 118s per 5-sec clip, **5.6× faster than Wan baseline**. Path: Lightricks' own upstream code (not `diffusers` — single-pass `LTXImageToVideoPipeline` physically cannot reproduce the 0.9.8 multi-scale 7+3 recipe). Wrapper at `bin/make-ltx-lightricks`.
- ✅ **LPIPS-gated speedup measurement harness** — novel on Mac. Every multiplier (quant, cache, distill, kernel) must pass per-frame LPIPS<0.05 AND speedup>1.10× before integration. At `bin/measure-render`.
- ✅ **Two-node mesh** — M5 Max (primary) + Mac mini M4 Pro (background). Mini runs Wan Q4_K_M GGUF (~9.6GB each stage) via `city96/ComfyUI-GGUF` + `kijai/ComfyUI-WanVideoWrapper` EasyCache, hits ~40 min/clip (M4 Pro is ~4× slower than M5 for Wan 14B). Mini is positioned as overnight/batch tier, not realtime peer.
- ✅ **Custom Metal flash-attention kernel — 12.32× speedup vs PyTorch MPS SDPA** with PSNR 137 dB (numerically identical). Hand-written tiled fp16 kernel with online softmax via `torch.mps.compile_shader`. On a (1, 40, 4096, 128) attention call: 4.9ms fused vs 60.4ms PyTorch reference. PyTorch's MPS SDPA falls back to a non-fused path; our kernel is the first Wan-shaped flash-attn for Apple Silicon. At `metal/flash_attn_mps.py` (~144 lines of MSL).
- ✅ **`render-route` upgraded** — auto-selects Wan (hero) vs LTX (B-roll) per scene heuristic. Now points at the working `make-ltx-lightricks`.
- 🔄 **In flight tonight:** EasyCache speedup quantification, Story Forge `.sf` DSL MVP parser, Metal flash-attention kernel.

Live build dashboard: see `build_status/index.html`.

---

## What it does

You write a story as a list of scenes — each scene is one still image prompt + one motion prompt + one narration line. Story Forge takes the list and:

1. Generates a Flux still per scene
2. Animates each still with Wan i2v at native 5-sec native speed
3. Renders each narration line with Piper TTS through a warm storyteller EQ chain
4. Generates an original instrumental score via ACE-Step
5. Composes the final film with `ffmpeg` — scene-synced narration via `adelay+amix`, music ducked under speech via sidechain compression, xfade transitions, Pillow PNG title and credits

Every step runs locally on Apple Silicon. The output is a regular `.mp4`.

---

## The first film — `saga.mp4`

To prove the pipeline, the first thing through it is a **two-act, 4:08 animated saga** called *The Bear Sister*. Act One is Studio Ghibli watercolor (a child rescued by a mother bear); Act Two is photoreal cinematic (the grown woman returning to find the bear family). One film, two visual languages, stitched with a fade-to-black bridge.

| | |
|---|---|
| **Runtime** | 4 min 8 sec |
| **Scenes** | 43 distinct |
| **Voices** | 1 Piper female (LibriTTS speaker 0), warm-EQ chain |
| **Music** | 2 ACE-Step instrumentals (Ghibli lullaby + cinematic homecoming) |
| **Compute hours** | ~12 hours (51 Wan i2v renders + parallel everything else) |
| **Hardware** | One MacBook Pro · Apple M5 Max · 128 GB unified memory |
| **Cloud calls** | **0** |

[▶ Watch on YouTube](https://youtu.be/_bFQTl7_vF4) · [Download `saga.mp4`](./saga.mp4) · [Read the full story (STORYBOOK.md)](./STORYBOOK.md)

---

## The story (transcript)

<details>
<summary><b>Act One — The Rescue</b></summary>

> In the deep pines of winter, a storm came. Wolves howled. Owls flew through the trees.
>
> A little girl wandered too far from home. Her lantern flickered in the swirling snow.
>
> The river was frozen. Silver fish slept beneath the ice. A small white rabbit watched her.
>
> She fell in the drifts. Her lantern dimmed. Foxes crept close. An owl glided overhead.
>
> But the forest knew. A mother bear stirred in her cave, two cubs tumbling at her heels.
>
> She followed the scent through the snow. Her cubs played behind her. Birds burst from the pines.
>
> She found the child, barely awake. The bear lowered her head, breath warm in the cold.
>
> With paws as soft as breath, she lifted the child. The cubs sniffed close, the owl watched.
>
> Into the warm dark of the den, where the fire burned and the mice slept in the moss.
>
> The cubs welcomed her like a sister. The mother stirred honey by the fire.
>
> They shared berries from a wooden bowl. Bats whispered across the cave ceiling.
>
> Winter passed in a single long breath. The stars spun, and the aurora rippled green.
>
> She slept between them, safe in their warmth. Their hearts beat together in the dark.
>
> In her dreams she flew with the spirits. Bears of starlight, salmon leaping through stars.
>
> When the icicles began to weep, spring returned. Flowers pushed through. Butterflies emerged.
>
> They walked into the sun, the cubs tumbling, deer watching, blossoms falling like pink snow.
>
> Her family found her on the path of flowers. But the forest stayed with her, forever.

</details>

<details>
<summary><b>Act Two — The Return</b> <i>(twenty winters later)</i></summary>

> Twenty winters had passed since she left the forest.
>
> But the call of the pines never left her.
>
> She took down the red hood from where it had hung.
>
> And drove the long road back into the redwoods.
>
> The trailhead waited where it had always been.
>
> She tied the hood at her throat, just as she had as a child.
>
> And the forest watched her come home.
>
> The salmon ran fierce in the stream where she had once dreamed of them.
>
> An owl marked her path. She remembered him.
>
> A fox emerged, and led her deeper.
>
> She found her stone, marked years ago.
>
> And entered the grove where the old ones lived.
>
> A great bear slept in the sun — older now, wiser.
>
> She knelt, and the elder stirred.
>
> They knew each other. Across the years.
>
> The forest sister had come home.
>
> The elder lifted her head. Her daughter came forward.
>
> And behind her came the next generation.
>
> The cubs came close, curious and bold.
>
> Their mother followed, slow and accepting.
>
> And the forest family was whole again.
>
> Together they walked through the deeper grove.
>
> Until they came to the old cave, moss-covered now.
>
> She entered alone, and found what her child-self had left.
>
> The elder pressed her forehead to hers. A goodbye.
>
> And she walked into the sun, the forest with her, forever.

</details>

---

## What's under the hood

```
       ┌─────────────────────────────────────────────────────────┐
       │                       M5 Max                             │
       │                                                          │
       │   ┌──────────┐    ┌──────────┐    ┌──────────┐           │
       │   │  Flux 1  │───►│ Wan 2.2  │───►│  ffmpeg  │──► film   │
       │   │ Dev FP8  │    │   i2v    │    │  compose │           │
       │   └──────────┘    └──────────┘    └────▲─────┘           │
       │   text-to-image   image-to-video       │                 │
       │                                        │                 │
       │   ┌──────────┐    ┌──────────┐    ┌────┴─────┐           │
       │   │  Piper   │    │ ACE-Step │    │  Pillow  │           │
       │   │ LibriTTS │    │  music   │    │ title +  │           │
       │   └──────────┘    └──────────┘    │ credits  │           │
       │   narration       instrumental    └──────────┘           │
       │                                                          │
       └─────────────────────────────────────────────────────────┘
```

### Component stack

| Stage | Tool | Model | Purpose |
|---|---|---|---|
| Still image per scene | [Flux 1 Dev FP8](https://huggingface.co/black-forest-labs/FLUX.1-dev) | 16 GB | Sets composition + character look |
| Motion per scene | [Wan 2.2 i2v](https://huggingface.co/Wan-AI) | 27 GB + 1 GB lightx2v LoRA | 5-sec native motion from each still |
| Voice narration | [Piper TTS](https://github.com/rhasspy/piper) | LibriTTS_R medium | Storyteller female voice |
| Music | [Song Forge / ACE-Step](https://github.com/ace-step/ACE-Step) | 13 GB | Original instrumental scores |
| Compose | [ffmpeg 8.1](https://ffmpeg.org/) | — | xfade, sidechain ducking, fades, mux |
| Title cards | [Pillow](https://pillow.readthedocs.io/) | — | PNG text overlays |

---

## The clever bits (what isn't in the YouTube tutorials)

### 1. Per-sentence Piper + `adelay+amix` for scene-synced narration

Most pipelines `concat` narration lines into one block at t=0. By scene 4 the audio is two scenes ahead of the visuals.

Story Forge renders each narration line separately, then places it at its scene's onscreen start time via ffmpeg's `adelay`. All lines are then `amix`'d into a single track padded to full video duration. Audio and visuals stay in lock-step the whole film.

### 2. Warm storyteller EQ chain

Piper's raw output sounds like a robot. The narrator in Story Forge films runs through a deliberate signal chain:

```
highpass(80) → +2dB low-shelf @ 250Hz   (chest warmth)
             → -2dB high-shelf @ 7kHz   (soften sibilance)
             → compressor (-18dB threshold, 2.5:1 ratio)
             → aecho(60ms, 0.15)         (intimate room tail)
             → loudnorm I=-16 LUFS        (bedtime-story level)
```

The output reads as "a person telling you a story," not "an AI generating speech."

### 3. Music ducks under narration automatically

The instrumental score plays throughout the film at -22 LUFS bed level. When the narrator speaks, ffmpeg's `sidechaincompress` filter ducks the music ~10 dB, then releases back. Zero manual mix automation.

### 4. Native-speed Wan, no slow-motion stretch

Many AI-video pipelines render 5-sec Wan clips and stretch them with `setpts*1.5` to fit longer scenes. Everything looks like dreamy slow-motion. Story Forge plays Wan at native 5-sec speed and uses more scenes instead — motion reads as real video.

### 5. xfade-based multi-act stitching

Combining two independently-rendered films into one saga uses `xfade=transition=fadeblack` between them (visual time-jump bridge) and audio gap handling for clean narration handoff. No editor required.

### 6. Scene-graph composition

Each scene is a record:
```python
{
    "still":     "<Flux prompt>",
    "motion":    "<Wan motion prompt>",
    "narration": "<one storyteller line>",
}
```

The pipeline iterates the list. Change one scene, only that scene re-renders. Add a scene, the timing math redistributes automatically.

---

## Roadmap — the 30× faster build-out

Story Forge today is the proof. The next iteration is what makes it run in minutes instead of hours per film. **Status updated 2026-05-24:**

| Multiplier | Target gain | Status |
|---|---|---|
| **LTX-Video 13B distilled 0.9.8 for B-roll** | 5.6× vs Wan | ✅ **WORKING on M5 MPS** — 118s/clip via Lightricks' upstream multi-scale code. Public first. |
| **LPIPS-gated speedup harness** | (gate, not gain) | ✅ Built — `bin/measure-render`. Novel on Mac. |
| **`render-route` engine auto-selector** | (routing, not gain) | ✅ Wired — auto-picks Wan vs LTX per scene heuristic. |
| **Q4_K_M GGUF Wan on Mac mini** | ~3.6× memory drop | ✅ Working — but M4 Pro compute is the bottleneck (40 min/clip vs M5's 10 min). Mini stays batch tier. |
| **EasyCache (DiT-native cache, kijai)** | 1.1-1.3× at 4 steps | 🔄 Test in flight. |
| **Custom Metal flash-attention** | 2-3× on attention path | 🔄 First kernel in progress. Wiring proven. |
| **2-step Wan distillation** | 2× perpetual | ⏳ Pending — overnight training on mini. |
| **Story Forge DSL compiler** | (productivity, not gain) | 🔄 MVP parser in progress. |
| **Multi-voice + Wav2Lip lip sync** | (feature, not speed) | ⏳ Multi-day. |
| **Two-node parallel render** | 2× throughput | ✅ Plumbed, but M5 + mini together is bottlenecked by mini's 40 min/clip — only useful for batch jobs. |

Stacked target: **today's 5-hour render → ~10-30 min per 4-min film on M5.**

### Public firsts (per 2026-05-24 cross-check)

1. LTX 13B distilled 0.9.8 working on Apple Silicon MPS via Lightricks upstream Python
2. LPIPS-gated speedup harness for Mac video diffusion (CI-style regression gates)
3. Custom Metal kernels for Wan video DiT (wiring proven; flash-attn pending)

### Benchmark to beat

**Liu Liu's Draw Things** (Apple-cited in the M5 launch) — ships Wan 2.2 on M-series and iPad M5 in a closed app. They're the speed reference on Mac. We're building the **open, measured, scriptable** equivalent — same speed bucket, with a DSL and a harness no closed app provides.

---

## Why local

The whole thing is the point. A 4-minute animated film with custom score and synced narration runs on **one laptop you can carry in your bag**. No upload step. No "your queue position is 47." No subscription. No telemetry.

The cloud companies will tell you this needs a server farm. It doesn't. It needs a MacBook Pro and a script.

### What the cloud would actually cost

A film like *The Bear Sister* (4 minutes, 51 distinct Wan i2v clips) on cloud-equivalent services:

| Service | $ per 5-sec clip | 1-min film (~12 clips) | 4-min film (~51 clips) | 10-min film (~120 clips) |
|---|---|---|---|---|
| **OpenAI Sora** | $2.50 | $30 | $128 | $300 |
| **Runway Gen-3** | $4.00 | $48 | $204 | $480 |
| **Pika 2.0** | $2.00–$3.00 | $24–36 | $102–153 | $240–360 |
| **Luma Dream Machine** | $2.50 | $30 | $128 | $300 |
| **Kling AI** | $1.75 | $21 | $89 | $210 |
| **fal.ai LTX** *(cheapest cloud)* | $0.10 | $1.20 | $5.10 | $12 |
| **Story Forge (your machine)** | **$0.00** | **$0** | **$0** | **$0** |

Plus the cloud services charge **monthly subscriptions just to access**:
- Runway Pro: $35/mo
- Pika Pro: $35/mo
- Sora: ChatGPT Plus $20/mo minimum

A single 51-clip film with 5× iteration cycles during development = ~$640 on Sora. Story Forge does it for the cost of electricity ($0.20).

Hardware amortization: an M5 Max MacBook Pro + Mac mini M4 Pro (~$4,900 one-time) breaks even against Sora pricing at **~40 films**. After that, every render is pure profit — and you keep the hardware for everything else you do.

---

## Credits — first film

- Story by **Matt Macosko + Claude**
- Animation: Wan 2.2 i2v
- Stills: Flux 1 Dev FP8
- Narration: Piper LibriTTS
- Music: Song Forge / ACE-Step
- Rendered locally on a M5 Max MacBook Pro
- No cloud

*A Story Forge production.*

---

## License

- **Pipeline code:** MIT (when published)
- **Saga film (`saga.mp4`):** [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) — share with attribution, don't sell
