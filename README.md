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

Story Forge today is the proof. The next iteration is what makes it run in minutes instead of hours per film:

- **1-step Wan distillation** — Train a LoRA that collapses Wan's 4-step inference into 1. *(4×)*
- **Metal kernels for attention** — Apple Metal Shading Language for Wan's hot path. *(2.5×)*
- **LTX-Video drop-in for B-roll** — Lightricks' 2 B-param LTX for ambient scenes. *(scenes drop from 11 min to ~30 sec)*
- **Two-node parallel render** — M5 + Mac mini Wan workers split the queue. *(2× throughput)*
- **Optical-flow warp** — Wan renders keyframes only, flow net interpolates. *(5-10×)*
- **Multi-voice + lip-sync** — Multiple Piper speakers + Wav2Lip for dialogue.
- **Web UI extensions** — fast/hero per-scene toggle, multi-voice routing, SFX library.

Stacked: today's 5-hour render becomes ~3 minutes. Every component is published, off-the-shelf, just needs wiring.

---

## Why local

The whole thing is the point. A 4-minute animated film with custom score and synced narration runs on **one laptop you can carry in your bag**. No upload step. No "your queue position is 47." No subscription. No telemetry.

The cloud companies will tell you this needs a server farm. It doesn't. It needs a MacBook Pro and a script.

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
