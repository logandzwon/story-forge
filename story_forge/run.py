#!/usr/bin/env python3
"""Story Forge runner — translates a .storyplan.json into a real mp4.

Two render paths:
  - "lean"  (default): per-scene Flux still -> render-route (Wan/LTX) i2v ->
            optional Piper narration -> ffmpeg xfade stitch. Tight, fast,
            no overlays. Honors per-scene `motion.engine` from the DSL.
  - "full"  (--engine full): delegate to story_pipeline.py for the legacy
            Bear-Sister pipeline (overlays, chapter grades, particle layers).

The IR -> config translation for the full path lives in
`storyplan_to_pipeline_config()` so existing callers keep working.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path("/Users/dtribe/Desktop/PROJECTS/AI/videopipe")
PIPELINE = REPO / "story_pipeline.py"
RENDER_ROUTE = REPO / "bin" / "render-route"
HOME = Path.home()
FLUX = HOME / "Scripts" / "flux_t2i.py"
WAN_OUT = HOME / "AI" / "videopipe" / "outputs"
DEFAULT_OUT_DIR = HOME / "AI" / "videopipe" / "outputs"
PIPER = HOME / "Library" / "Python" / "3.9" / "bin" / "piper"
PIPER_MODEL = (HOME / "Desktop" / "PROJECTS" / "Song Forge"
               / "piper_voices" / "en_US-libritts_r-medium.onnx")

# ---------------------------------------------------------------------------
# Full-pipeline config translation (kept for --engine full / legacy use)
# ---------------------------------------------------------------------------

def storyplan_to_pipeline_config(plan: dict[str, Any]) -> dict[str, Any]:
    """Convert a storyplan dict into the JSON shape story_pipeline.py expects."""
    meta = plan["film_meta"]
    scenes_dict: dict[str, dict[str, Any]] = plan["scenes"]

    pipeline_scenes = []
    for name, sc in scenes_dict.items():
        still = sc.get("still_spec") or {}
        motion = sc.get("motion_spec") or {}
        narrate = sc.get("narration_spec") or {}
        pipeline_scenes.append({
            "name": name,
            "still": still.get("prompt", ""),
            "motion": motion.get("prompt", ""),
            "narration": narrate.get("line", ""),
            "still_seed": still.get("seed"),
            "motion_seed": motion.get("seed"),
            "duration": float(motion.get("duration",
                                         meta.get("scene_duration", 8.5))),
        })

    voice_presets = plan.get("voice_presets", {})
    voice = "warm_female_storyteller" if voice_presets else "none"

    return {
        "title": meta.get("title", "Untitled"),
        "slug": meta.get("slug", "untitled"),
        "style": "",
        "character": "",
        "scenes": pipeline_scenes,
        "voice": voice,
        "scene_duration": meta.get("scene_duration", 8.5),
        "overlays": {
            "dust_all": True, "film_grain": True, "vignette": True,
            "chapter_grades": {},
        },
        "_storyplan": plan,
    }


# ---------------------------------------------------------------------------
# Lean per-scene renderer
# ---------------------------------------------------------------------------

def _sh(cmd: list[str], **kw) -> None:
    print(f"$ {' '.join(str(c) for c in cmd)}", flush=True)
    subprocess.run(cmd, check=True, **kw)


def _ffprobe_duration(path: Path) -> float:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        text=True,
    ).strip()
    return float(out)


def _render_still(prompt: str, out_png: Path, seed: int,
                  width: int = 768, height: int = 512) -> None:
    """Flux T2I to out_png. Idempotent: skip if already there."""
    if out_png.exists():
        print(f"[still] cached: {out_png}")
        return
    out_png.parent.mkdir(parents=True, exist_ok=True)
    _sh([str(FLUX), prompt, "--out", str(out_png),
         "--w", str(width), "--h", str(height), "--seed", str(int(seed))])


def _render_motion(prompt: str, still_png: Path, out_mp4: Path,
                   engine: str, duration: float, label: str) -> None:
    """render-route i2v -> moves result into out_mp4. Idempotent."""
    if out_mp4.exists():
        print(f"[motion] cached: {out_mp4}")
        return
    WAN_OUT.mkdir(parents=True, exist_ok=True)

    # render-route writes into WAN_OUT/<label>_*<ts>.mp4; we glob for it after.
    before = set(WAN_OUT.glob(f"{label}_*.mp4"))
    eng_arg = engine if engine in ("wan", "ltx") else "auto"
    _sh(["python3", str(RENDER_ROUTE),
         "--still", str(still_png),
         "--duration", str(duration),
         "--label", label,
         "--engine", eng_arg,
         prompt])
    after = sorted(set(WAN_OUT.glob(f"{label}_*.mp4")) - before,
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if not after:
        # Fallback: most-recent file with that label prefix
        all_match = sorted(WAN_OUT.glob(f"{label}_*.mp4"),
                           key=lambda p: p.stat().st_mtime, reverse=True)
        if not all_match:
            raise RuntimeError(f"render-route produced no output for {label}")
        produced = all_match[0]
    else:
        produced = after[0]
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(produced, out_mp4)


def _conform_clip(src: Path, dst: Path, scene_dur: float,
                  width: int = 1280, height: int = 720, fps: int = 30) -> None:
    """Resample to target res/fps/duration with letterbox-crop."""
    if dst.exists():
        return
    src_dur = _ffprobe_duration(src)
    pts_mult = scene_dur / max(0.1, src_dur)
    vf = (f"setpts=PTS*{pts_mult:.4f},"
          f"scale={width}:{height}:force_original_aspect_ratio=increase,"
          f"crop={width}:{height},fps={fps}")
    _sh(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-i", str(src), "-vf", vf,
         "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(fps),
         "-t", str(scene_dur), str(dst)])


def _render_narration(line: str, voice_spec: dict[str, Any],
                      out_wav: Path) -> bool:
    """Piper one line -> wav. Returns True if file produced."""
    if out_wav.exists():
        return True
    if not line or not line.strip():
        return False
    if not PIPER.exists() or not PIPER_MODEL.exists():
        print(f"[narrate] piper or model missing; skipping line", flush=True)
        return False
    attrs = (voice_spec or {}).get("attrs", {}) if voice_spec else {}
    length_scale = str(attrs.get("length", 1.10))
    speaker = str(attrs.get("speaker", 0))
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [str(PIPER), "-m", str(PIPER_MODEL), "-f", str(out_wav),
         "--speaker", speaker,
         "--length-scale", length_scale,
         "--noise-scale", "0.5",
         "--noise-w-scale", "0.7"],
        input=line, text=True, check=True,
    )
    return out_wav.exists()


def _stitch(clips: list[Path], out_mp4: Path,
            xfade: float = 0.5, scene_dur: float = 5.0) -> None:
    """xfade-stitch the visual track (no audio) -> out_mp4."""
    if len(clips) == 1:
        shutil.copy2(clips[0], out_mp4)
        return
    inputs: list[str] = []
    for c in clips:
        inputs += ["-i", str(c)]
    fc = []
    last = "[0:v]"
    off = scene_dur - xfade
    for i in range(1, len(clips)):
        label = f"v{i}"
        fc.append(f"{last}[{i}:v]xfade=transition=fade:"
                  f"duration={xfade}:offset={off:.3f}[{label}]")
        last = f"[{label}]"
        off += scene_dur - xfade
    _sh(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         *inputs, "-filter_complex", ";".join(fc),
         "-map", last,
         "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-preset", "medium", "-crf", "18",
         "-r", "30", "-an", str(out_mp4)])


def _mux_narration(visuals: Path, vo_wav: Path | None, out: Path) -> None:
    """Mux visuals + (optional) narration to final mp4 with fade in/out."""
    visuals_dur = _ffprobe_duration(visuals)
    fade_out = max(0.1, visuals_dur - 1.0)

    if vo_wav and vo_wav.exists():
        _sh(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-i", str(visuals), "-i", str(vo_wav),
             "-filter_complex",
             f"[0:v]fade=in:st=0:d=0.5,fade=out:st={fade_out}:d=1.0[v];"
             f"[1:a]volume=1.0,afade=in:st=0:d=0.3,"
             f"afade=out:st={fade_out}:d=1.0,apad[a]",
             "-map", "[v]", "-map", "[a]",
             "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-preset", "medium", "-crf", "18",
             "-c:a", "aac", "-b:a", "192k", "-shortest",
             "-movflags", "+faststart", str(out)])
    else:
        _sh(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-i", str(visuals),
             "-vf", f"fade=in:st=0:d=0.5,fade=out:st={fade_out}:d=1.0",
             "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-preset", "medium", "-crf", "18",
             "-movflags", "+faststart", str(out)])


def render_lean(plan: dict[str, Any],
                out_path: Path | None = None,
                scene_filter: list[str] | None = None,
                work_dir: Path | None = None) -> dict[str, Any]:
    """End-to-end render via per-scene render-route + ffmpeg stitch."""
    meta = plan["film_meta"]
    slug = meta.get("slug", "untitled")
    scene_dur = float(meta.get("scene_duration", 5.0))
    scenes_all = plan["scenes"]  # dict preserves insertion order

    # Filter scenes if requested
    if scene_filter:
        wanted = set(scene_filter)
        scenes = {k: v for k, v in scenes_all.items() if k in wanted}
        missing = wanted - set(scenes.keys())
        if missing:
            raise RuntimeError(f"scenes not in plan: {sorted(missing)}")
        if not scenes:
            raise RuntimeError("no scenes left after --scenes filter")
    else:
        scenes = scenes_all

    work_dir = work_dir or (HOME / "Desktop" / "AI Videos" / slug)
    work_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_path or (DEFAULT_OUT_DIR / f"{slug}.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    voice_presets = plan.get("voice_presets", {})
    # First voice preset = default narrator
    default_voice = (next(iter(voice_presets.values()))
                     if voice_presets else None)

    t0 = time.time()
    enhanced_clips: list[Path] = []
    narration_pieces: list[tuple[int, Path]] = []  # (scene_index_1based, wav)

    for idx, (name, sc) in enumerate(scenes.items(), start=1):
        print(f"\n=== scene {idx}/{len(scenes)}: {name} ===", flush=True)
        still_spec = sc.get("still_spec") or {}
        motion_spec = sc.get("motion_spec") or {}
        narrate_spec = sc.get("narration_spec") or {}

        still_prompt = still_spec.get("prompt", "")
        motion_prompt = motion_spec.get("prompt", "")
        engine = (motion_spec.get("engine") or "auto").lower()
        duration = float(motion_spec.get("duration", scene_dur))
        still_seed = int(still_spec.get("seed") or (1000 + idx * 17))

        label = f"{slug}_{idx:02d}_{name}"
        still_png = work_dir / f"still_{idx:02d}.png"
        raw_mp4 = work_dir / f"raw_{idx:02d}.mp4"
        conformed = work_dir / f"clip_{idx:02d}.mp4"

        # 1. Still (skip if motion engine doesn't need one — but both Wan-i2v
        #    and LTX-i2v do, so always render).
        if still_prompt:
            _render_still(still_prompt, still_png, seed=still_seed)
        else:
            raise RuntimeError(f"scene {name}: still.prompt is required")

        # 2. Motion
        _render_motion(motion_prompt or still_prompt, still_png, raw_mp4,
                       engine=engine, duration=duration, label=label)

        # 3. Conform
        _conform_clip(raw_mp4, conformed, scene_dur=duration)
        enhanced_clips.append(conformed)

        # 4. Narration piece (per scene)
        line = (narrate_spec or {}).get("line", "").strip()
        if line:
            voice_name = narrate_spec.get("engine")  # 'narrate warm:' -> 'warm'
            voice_spec = voice_presets.get(voice_name) or default_voice
            piece = work_dir / "vo_pieces" / f"p_{idx:02d}.wav"
            if _render_narration(line, voice_spec, piece):
                narration_pieces.append((idx, piece))

    # 5. Stitch visuals (use the first scene's duration as scene_dur for offsets;
    #    when scenes differ in length this is approximate but acceptable for MVP)
    visuals = work_dir / "visuals.mp4"
    if visuals.exists():
        visuals.unlink()
    _stitch(enhanced_clips, visuals,
            xfade=0.5,
            scene_dur=float(next(iter(scenes.values()))
                            .get("motion_spec", {}).get("duration", scene_dur)))

    # 6. Build narration track (adelay+amix, scene-synced) if we have any
    vo_wav: Path | None = None
    if narration_pieces:
        xfade = 0.5
        per_scene = float(next(iter(scenes.values()))
                          .get("motion_spec", {}).get("duration", scene_dur))
        scene_advance = per_scene - xfade
        total_audio_dur = (len(scenes) - 1) * scene_advance + per_scene
        args = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "warning"]
        fc_parts = []
        mix_labels = []
        for inp_i, (scene_i, piece) in enumerate(narration_pieces):
            args += ["-i", str(piece)]
            delay_ms = int((scene_i - 1) * scene_advance * 1000)
            fc_parts.append(f"[{inp_i}:a]adelay={delay_ms}|{delay_ms}"
                            f"[a{inp_i}]")
            mix_labels.append(f"[a{inp_i}]")
        fc = (";".join(fc_parts) + ";" + "".join(mix_labels)
              + f"amix=inputs={len(narration_pieces)}:"
                f"duration=longest:normalize=0,"
              + f"apad=whole_dur={total_audio_dur}[amixed]")
        vo_wav = work_dir / "vo.wav"
        args += ["-filter_complex", fc, "-map", "[amixed]",
                 "-t", str(total_audio_dur),
                 "-c:a", "pcm_s16le", str(vo_wav)]
        _sh(args)

    # 7. Final mux
    _mux_narration(visuals, vo_wav, out_path)

    elapsed = time.time() - t0
    return {
        "output": str(out_path),
        "scenes": len(scenes),
        "seconds": round(elapsed, 1),
        "narration_lines": len(narration_pieces),
        "work_dir": str(work_dir),
    }


# ---------------------------------------------------------------------------
# Top-level dispatch
# ---------------------------------------------------------------------------

def render(plan_path: Path, dry_run: bool = False,
           engine: str = "lean",
           out: Path | None = None,
           scenes: list[str] | None = None) -> dict[str, Any]:
    plan = json.loads(Path(plan_path).read_text())
    if dry_run:
        return storyplan_to_pipeline_config(plan)

    if engine == "full":
        cfg = storyplan_to_pipeline_config(plan)
        proc = subprocess.run(
            [sys.executable, str(PIPELINE)],
            input=json.dumps(cfg), text=True, check=True,
        )
        return {"returncode": proc.returncode, "engine": "full",
                "config": cfg}

    # default lean path
    result = render_lean(plan, out_path=out, scene_filter=scenes)
    result["engine"] = "lean"
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("plan", help="path to .storyplan.json")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--engine", choices=["lean", "full"], default="lean")
    ap.add_argument("--out", default=None,
                    help="output mp4 path (default: ~/AI/videopipe/outputs/<slug>.mp4)")
    ap.add_argument("--scenes", default=None,
                    help="comma-separated scene names to render (default: all)")
    args = ap.parse_args()
    scene_filter = ([s.strip() for s in args.scenes.split(",") if s.strip()]
                    if args.scenes else None)
    out = Path(args.out) if args.out else None
    res = render(Path(args.plan), dry_run=args.dry, engine=args.engine,
                 out=out, scenes=scene_filter)
    print(json.dumps(res, indent=2, default=str))
