#!/usr/bin/env python3
"""Smoke test: cabin_open.sf parses + resolves + emits to a valid storyplan dict."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from story_forge.parser import parse                  # noqa: E402
from story_forge.resolver import resolve              # noqa: E402
from story_forge.emitter import emit_storyplan        # noqa: E402

EXAMPLE = (Path(__file__).resolve().parents[1]
           / "examples" / "cabin_open.sf")


class TestStoryForgeMVP(unittest.TestCase):

    def setUp(self) -> None:
        self.ast = parse(EXAMPLE)
        self.resolved = resolve(self.ast)
        self.plan = emit_storyplan(self.resolved)

    def test_ast_has_film_node(self):
        film_nodes = [n for n in self.ast if n["type"] == "film"]
        self.assertEqual(len(film_nodes), 1)
        self.assertEqual(film_nodes[0]["title"], "Cabin Open")

    def test_vars_captured(self):
        self.assertIn("style", self.resolved["vars"])
        self.assertIn("child", self.resolved["vars"])
        self.assertIn("cabin", self.resolved["vars"])

    def test_three_scenes(self):
        scenes = self.plan["scenes"]
        self.assertEqual(len(scenes), 3)
        self.assertIn("snow_walk", scenes)
        self.assertIn("cabin_glow", scenes)
        self.assertIn("fireside", scenes)

    def test_each_scene_has_still_motion_narration(self):
        for name, sc in self.plan["scenes"].items():
            self.assertIsNotNone(sc["still_spec"], f"{name} missing still")
            self.assertIsNotNone(sc["motion_spec"], f"{name} missing motion")
            self.assertIsNotNone(sc["narration_spec"],
                                 f"{name} missing narration")

    def test_variable_interpolation(self):
        prompt = self.plan["scenes"]["snow_walk"]["still_spec"]["prompt"]
        self.assertIn("Studio Ghibli watercolor", prompt)
        self.assertIn("a small child in a red hooded cloak", prompt)
        self.assertNotIn("{$", prompt)

    def test_deterministic_seeds(self):
        s1 = self.plan["scenes"]["snow_walk"]["still_spec"]["seed"]
        s2 = self.plan["scenes"]["cabin_glow"]["still_spec"]["seed"]
        # Both must be ints in the 32-bit unsigned range and distinct per scene.
        self.assertIsInstance(s1, int)
        self.assertIsInstance(s2, int)
        self.assertNotEqual(s1, s2)
        # Re-emit and confirm reproducibility.
        again = emit_storyplan(resolve(parse(EXAMPLE)))
        self.assertEqual(
            again["scenes"]["snow_walk"]["still_spec"]["seed"], s1)

    def test_voice_preset_defined(self):
        vp = self.plan["voice_presets"]
        self.assertIn("warm", vp)
        self.assertTrue(vp["warm"]["value"].startswith("piper/"))

    def test_music_preset_and_per_scene_ref(self):
        self.assertIn("wintry", self.plan["music_presets"])
        music = self.plan["scenes"]["snow_walk"]["music_spec"]
        self.assertEqual(music["preset"], "wintry")
        self.assertAlmostEqual(music["attrs"]["vol"], 0.30, places=3)

    def test_directives_captured(self):
        t_names = [t["name"] for t in self.plan["transitions"]]
        self.assertIn("xfade", t_names)
        self.assertEqual(len(self.plan["mixes"]), 1)
        self.assertEqual(self.plan["mixes"][0]["args"][0], "duck")

    def test_film_meta(self):
        meta = self.plan["film_meta"]
        self.assertEqual(meta["slug"], "cabin_open")
        self.assertEqual(meta["target"], "m5+mini")
        self.assertAlmostEqual(meta["scene_duration"], 8.5, places=3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
