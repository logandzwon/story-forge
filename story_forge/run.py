#!/usr/bin/env python3
"""Story Forge runner — translates a .storyplan.json into the legacy
story_pipeline.py config shape and dispatches the render.

We deliberately keep the render logic in story_pipeline.py — this file is the
thin bridge between the new DSL world and the existing Flux+Wan+Piper pipeline.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

PIPELINE = Path("/Users/dtribe/Desktop/PROJECTS/AI/videopipe/story_pipeline.py")


def storyplan_to_pipeline_config(plan: dict[str, Any]) -> dict[str, Any]:
    """Convert a storyplan dict into the JSON shape story_pipeline.py expects."""
    meta = plan["film_meta"]
    scenes_dict: dict[str, dict[str, Any]] = plan["scenes"]

    # Preserve insertion order — Python 3.7+ dict ordering matches scene definition order.
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

    # Pick a voice — first defined preset wins for the MVP.
    voice_presets = plan.get("voice_presets", {})
    voice = "warm_female_storyteller" if voice_presets else "none"

    return {
        "title": meta.get("title", "Untitled"),
        "slug": meta.get("slug", "untitled"),
        "style": "",   # DSL handles styling per-scene via prompts
        "character": "",
        "scenes": pipeline_scenes,
        "voice": voice,
        "scene_duration": meta.get("scene_duration", 8.5),
        "overlays": {
            "dust_all": True,
            "film_grain": True,
            "vignette": True,
            "chapter_grades": {},
        },
        # Carry the storyplan along so downstream tools can inspect it.
        "_storyplan": plan,
    }


def render(plan_path: Path, dry_run: bool = False) -> dict[str, Any]:
    plan = json.loads(Path(plan_path).read_text())
    cfg = storyplan_to_pipeline_config(plan)
    if dry_run:
        return cfg

    proc = subprocess.run(
        [sys.executable, str(PIPELINE)],
        input=json.dumps(cfg), text=True, check=True,
    )
    return {"returncode": proc.returncode, "config": cfg}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: run.py PLAN.storyplan.json [--dry]", file=sys.stderr)
        raise SystemExit(2)
    dry = "--dry" in sys.argv
    out = render(Path(sys.argv[1]), dry_run=dry)
    print(json.dumps(out, indent=2, default=str))
