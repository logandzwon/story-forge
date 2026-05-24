# Story Forge — Session Handoff

> Read this first when resuming the build. Captures exact state as of the last commit.

---

## What Story Forge is

A local-only generative cinema pipeline. Five open-source models composed by ffmpeg:
- **Flux 1 Dev FP8** — still per scene
- **Wan 2.2 i2v + lightx2v 4-step LoRA** — motion per scene (5-sec native)
- **Piper LibriTTS speaker 0** — warm storyteller narration
- **ACE-Step (Song Forge)** — original instrumental music
- **ffmpeg** — xfade stitch, sidechain ducking, fade in/out, mux, Pillow PNG title/credits

GitHub: https://github.com/nicedreamzapp/story-forge
First film live on YouTube: https://youtu.be/_bFQTl7_vF4
Live website: https://nicedreamzwholesale.com/software/story-forge/
First Facebook Reel: posted (in processing as of last session)

---

## What's working RIGHT NOW (verified end-to-end)

**Pipeline (M5 Max):**
- Story Forge UI at `http://127.0.0.1:17600/story` — form-driven scene authoring
- `story_pipeline.py` at `~/Desktop/PROJECTS/AI/videopipe/story_pipeline.py` — runs the full Flux → Wan → Piper → ffmpeg pipeline given a JSON config
- Scene-synced narration via `adelay+amix` (each Piper line at its scene's onscreen start)
- Warm EQ chain for narration: highpass(80) → +2dB low-shelf @ 250Hz → -2dB high-shelf @ 7kHz → compressor → aecho 60ms → loudnorm I=-16 LUFS
- Sidechain music ducking under narration (automatic)
- Saga stitch recipe (multi-film): strip credits from each act, `xfade=fadeblack` 1.8s between, new combined credit roll. Validated on bear-sister + bear-return saga (4:08 total).

**M5 Forge running services:**
- Flask/Story Forge server: `127.0.0.1:17600`
- ComfyUI: `127.0.0.1:8188`
- Song Forge / ACE-Step server: `localhost:8767`

**Apps / launchers:**
- `~/Desktop/Story Forge.app` (renamed from MakeVideo.app) — clicking opens Brave at `localhost:17600/story`

---

## Critical workflow rules (lessons learned the hard way)

These are FROZEN — do not re-derive, just apply:

1. **Piper flag is `--noise-w-scale`, NOT `--noise-scale-w`** (word order matters; the wrong order silently gets read aloud as part of the speech).
2. **Pipe each Piper sentence to its own file** — `-f` only saves the LAST stdin line, so multi-sentence Piper requires per-sentence rendering + concat with silence.
3. **Scene-synced narration via `adelay+amix`, never naive concat at t=0** — concat puts all narration in the first scene's audio band.
4. **Native 5-sec Wan, not stretched** — never `setpts=PTS*1.5`. Looks like dreamy slow-motion. Use more scenes if you need a longer movie.
5. **QC every audio output before claiming it works** — use silencedetect + spectrogram, not duration alone. The Piper flag bug only got caught after Matt heard "noise-scale-w 0.7" said aloud.
6. **QC every visual output before showing Matt** — extract 3-4 frames with `ffmpeg -ss`, Read them, judge yourself. The baseball commercial fiasco came from shipping without checking.
7. **Wan doesn't understand physics** — never ask for object collisions (bat-meets-ball), trajectories (curveball arcs), or coordinated multi-character action. Wan handles: ambient motion, camera moves, walking, breathing, atmosphere, particles. Stay inside that lane.
8. **First-person POV doesn't work** — Wan pulls back to third-person. If you need POV, use a reference photo with the right composition baked in (we used a real MLB home-plate photo + Wan animated it correctly).

---

## The two-week speedup build — status

Stack goal: drop a 4-min film render from 5 hours to ~10-30 min. Stacked savings target: 30× baseline.

### ✅ Plumbed (model files, configs, scripts written)

| Item | Where | Status |
|---|---|---|
| LTX-Video 13B distilled model | `~/AI/ComfyUI/models/diffusion_models/ltx/ltxv-13b-0.9.8-distilled.safetensors` (27 GB) | On disk |
| LTX-Video ComfyUI custom node | `~/AI/ComfyUI/custom_nodes/ComfyUI-LTXVideo` | Installed |
| `make-ltx-video` CLI | `~/Desktop/PROJECTS/AI/videopipe/bin/make-ltx-video` | Written, NOT WORKING — see "Open problems" |
| `render-route` CLI (Wan/LTX auto-selector) | `~/Desktop/PROJECTS/AI/videopipe/bin/render-route` | Written, depends on make-ltx-video |
| Mini ComfyUI install | `~/AI/ComfyUI/` on mini | Installed, listening on `:8188` |
| Mini Wan i2v models | `~/AI/ComfyUI/models/diffusion_models/wan2.2_i2v_*.safetensors` on mini (54 GB) | Downloaded |
| Mini Wan i2v LoRAs | `~/AI/ComfyUI/models/loras/wan2.2_i2v_lightx2v_*.safetensors` on mini (2.2 GB) | Downloaded |
| Mini Wan VAE | `~/AI/ComfyUI/models/vae/wan_2.1_vae.safetensors` on mini | Downloaded (verify with `mini "ls -lh ~/AI/ComfyUI/models/vae/"`) |
| Mini text encoder | `~/AI/ComfyUI/models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors` on mini (6.3 GB) | Downloaded |
| Mini Piper + LibriTTS voice | `~/Library/Python/3.9/bin/piper` + `~/piper_voices/en_US-libritts_r-medium.onnx` on mini | Installed |
| Mini Real-ESRGAN weights | `~/AI/upscale/RealESRGAN_x4plus.pth` on mini | Downloaded |
| Mini training stack | `peft 0.19.1`, `accelerate 1.13.0` in mini's ComfyUI venv | Installed |
| Mini-side Wan render CLI | `~/mini_wan_render.py` on mini | Pushed |
| Wan distill v1 scaffold | `~/AI/distill/wan_distill_v1.py` on mini | Scaffolded (stubs for ComfyUI WanVideoWrapper integration) |

### ❌ NOT working yet — open problems

1. **LTX-Video standalone render fails:** `make-ltx-video` errors with `SingleFileComponentError: Failed to load T5EncoderModel`. The distilled single-file checkpoint doesn't include the T5 text encoder weights. Fix: load text encoder separately via `T5EncoderModel.from_pretrained("Lightricks/LTX-Video")` and pass to the pipeline, OR use `LTXImageToVideoPipeline.from_pretrained()` with the full HF repo (will redownload).

2. **Mini Wan inference — workflow JSON rejected (HTTP 400):** Test job `e94c64efc006` failed with `urllib.error.HTTPError: HTTP Error 400: Bad Request` when posting workflow to mini's ComfyUI `/prompt` endpoint. Mini's ComfyUI is up + accepting requests, just rejecting our specific workflow JSON. Debug path: (a) `mini "~/AI/ComfyUI/venv/bin/python -c 'import json,urllib.request; print(urllib.request.urlopen(\"http://localhost:8188/object_info\").read()[:2000])'"` to see what node classes mini actually has, then (b) compare against the workflow in `mini_wan_render.py` lines ~55-130. Likely culprits: node class name mismatches (`WanImageToVideo` may be `WanVideo_I2V` or similar), or missing custom node for the GGUF unet loader (we used FP16 path on mini, not GGUF — workflow may need updating to `UNETLoader` with `weight_dtype="fp16"`).

3. **Wan distill training scaffold has TODO stubs:** `wan_distill_v1.py` model loader and teacher/student inference are explicit TODOs. The genuinely hard part is importing ComfyUI's `WanVideoWrapper` code as a Python module (not just via ComfyUI's web queue) to expose Wan's UNet for PEFT LoRA training. Estimated: 1-2 days of focused dev.

4. **Metal kernels blocked on Xcode app install:** CLI tools alone don't include the `metal` compiler. Matt needs to manually install full Xcode from the App Store (~7 GB). Until then, attention-block Metal kernels can't be compiled.

5. **Story Forge UI extensions never built:** multi-voice routing, SFX library, fast-mode toggle per scene, lip-sync (Wav2Lip) integration. All planned, none implemented. UI is still single-voice + procedural overlays only.

### Roadmap (in priority order)

1. **Validate mini Wan inference** (highest impact / lowest risk — if it works, parallel rendering halves all future renders with zero code change)
2. **Fix LTX standalone CLI** (T5 encoder loading) — once fixed, B-roll renders drop from 11 min to ~30 sec
3. **Build mini-aware story_pipeline.py routing** — alternating scenes across M5 + mini
4. **Wan distill training** — multi-day project; the genuine 4× perpetual speedup
5. **Story Forge UI extensions** (multi-voice + SFX + fast-mode toggle)
6. **Wav2Lip integration** for dialogue scenes (requires character voice clones + face still)
7. **Metal kernels** (when Xcode app installed)

---

## File locations cheatsheet

| Thing | Path |
|---|---|
| Pipeline core (Python) | `~/Desktop/PROJECTS/AI/videopipe/story_pipeline.py` |
| Story Forge server | `~/Desktop/PROJECTS/AI/videopipe/server.py` (port 17600) |
| Story Forge UI HTML | `~/Desktop/PROJECTS/AI/videopipe/ui/story.html` |
| Story Forge app launcher | `~/Desktop/Story Forge.app` |
| Wan inference CLI (M5) | `~/Desktop/PROJECTS/AI/videopipe/bin/make-video` |
| LTX inference CLI (M5) | `~/Desktop/PROJECTS/AI/videopipe/bin/make-ltx-video` (BROKEN — see open problems) |
| Render router (M5) | `~/Desktop/PROJECTS/AI/videopipe/bin/render-route` |
| Mini Wan render CLI | `~/mini_wan_render.py` on mini |
| Mini distill scaffold | `~/AI/distill/wan_distill_v1.py` on mini |
| Flux T2I script | `~/Scripts/flux_t2i.py` |
| Piper binary | `~/Library/Python/3.9/bin/piper` |
| Piper voice model | `~/Desktop/PROJECTS/Song Forge/piper_voices/en_US-libritts_r-medium.onnx` |
| Song Forge server | `localhost:8767` |
| ComfyUI (M5) | `localhost:8188`, dir `~/AI/ComfyUI/` |
| Mini access CLI | `~/.local/bin/mini "<cmd>"` |
| Saga output | `~/Desktop/AI Videos/bear-sister-saga.mp4` (161 MB) |
| Saga compressed for repo | `~/Desktop/PROJECTS/story-forge/saga.mp4` (92 MB) |
| Saga 9:16 vertical (Reels) | `~/Desktop/AI Videos/bear-sister-saga-vertical.mp4` |

---

## How to drive Matt's real Brave for web tasks

ComfyUI restart pattern needed: `--remote-debugging-port=9222`. Memory rule: never use Playwright/Chromium — always Matt's authenticated Brave. To enable CDP control:

```bash
osascript -e 'tell application "Brave Browser" to quit'
sleep 3
open -na "Brave Browser" --args --remote-debugging-port=9222 --restore-last-session
sleep 4
curl -s http://127.0.0.1:9222/json/version  # verify
```

Then `mcp__chrome-devtools__*` tools work. YouTube + Facebook Reels upload was validated this way.

---

## Memory entries that matter (in `~/.claude/projects/-Users-dtribe/memory/`)

- `reference_story_forge_pipeline.md` — full pipeline architecture
- `reference_warm_narration_recipe.md` — the exact Piper + EQ chain for warm storyteller voice
- `reference_saga_stitch_recipe.md` — how to combine films into a saga
- `feedback_narration_must_sync_scenes.md` — adelay+amix rule
- `feedback_qc_ai_videos_before_sending.md` — always frame-check before showing Matt

---

## Last actions in the session

1. Saga published to YouTube: https://youtu.be/_bFQTl7_vF4 (public, made for kids, AI + sensory-friendly tag stack)
2. Saga posted to Facebook Reels as 9:16 vertical with blurred-background extension (full original preserved + tasteful padding)
3. Story Forge GitHub repo updated with YouTube embed thumbnail
4. nicedreamzwholesale.com/software/story-forge/ page deployed
5. /software/ index card added under "🧠 The local-AI stack"
6. Started mini Wan validation (job in flight at session end — check status with mini --done command)
7. Tried LTX-Video standalone test → failed on T5 encoder loading (documented above)

---

## Next session — drop right in with

```bash
# 1. Check if the mini Wan validation job finished
~/.local/bin/mini --done $(cat /tmp/mini_wan_job.txt 2>/dev/null)

# 2. Or restart validation cleanly
~/.local/bin/mini --queue "~/AI/ComfyUI/venv/bin/python ~/mini_wan_render.py --still /tmp/mini_test_still.png --prompt 'gentle wind moves' --label mini_validate --duration 5 --width 832 --height 480"

# 3. Fix LTX (open problem #1) — needs T5 encoder loader fix in make-ltx-video
```

Resume from "Validate mini Wan inference" in the roadmap above.
