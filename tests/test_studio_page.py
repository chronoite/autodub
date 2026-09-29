"""Studio v2 page contracts: the studio_progress PATCH field and the page's
source-level wiring (assets exist, endpoints referenced are real)."""
from __future__ import annotations

import unittest

from autodub.casting import sanitize_studio_progress

from tests import ROOT


class StudioProgressTests(unittest.TestCase):
    def test_valid_progress_is_clamped_and_sorted(self) -> None:
        result = sanitize_studio_progress(
            {"step": 3, "clean_labels": ["speaker-02", "speaker-01", "speaker-02", "  "]})
        self.assertEqual(result["step"], 3)
        self.assertEqual(result["clean_labels"], ["speaker-01", "speaker-02"])
        self.assertEqual(result["draft_cast"], {})

    def test_bad_step_rejected_with_400_class_error(self) -> None:
        # None used to raise TypeError -> a 500 at the PATCH boundary
        for bad in (0, 5, "x", None, [], {}):
            with self.assertRaises(ValueError):
                sanitize_studio_progress({"step": bad, "clean_labels": []})

    def test_oversized_lists_truncate_instead_of_rejecting(self) -> None:
        # convenience checklist: silently clamp, never brick persistence (panel)
        labels = [f"speaker-{i:03d}" for i in range(80)]
        result = sanitize_studio_progress({"step": 1, "clean_labels": labels})
        self.assertEqual(len(result["clean_labels"]), 64)
        with self.assertRaises(ValueError):
            sanitize_studio_progress(["not", "a", "dict"])

    def test_labels_truncated_to_80_chars(self) -> None:
        result = sanitize_studio_progress({"step": 1, "clean_labels": ["a" * 200]})
        self.assertEqual(len(result["clean_labels"][0]), 80)

    def test_draft_cast_round_trip_and_clamps(self) -> None:
        result = sanitize_studio_progress({"step": 2, "clean_labels": [], "draft_cast": {
            "speaker-01": {"character": "Marco", "voice": "qwen-auto:speaker-01"},
            "speaker-02": "not-a-dict",
            "speaker-03": {"character": "x" * 500, "voice": None},
        }})
        self.assertEqual(result["draft_cast"]["speaker-01"],
                         {"character": "Marco", "voice": "qwen-auto:speaker-01"})
        self.assertNotIn("speaker-02", result["draft_cast"])
        self.assertEqual(len(result["draft_cast"]["speaker-03"]["character"]), 120)
        self.assertEqual(result["draft_cast"]["speaker-03"]["voice"], "")
        with self.assertRaises(ValueError):
            sanitize_studio_progress({"step": 1, "draft_cast": ["list"]})


class StudioPageSourceContract(unittest.TestCase):
    """The page is static — pin that its assets exist and that every API path the JS
    calls is a route the server actually has (catches endpoint drift)."""

    def test_assets_exist(self) -> None:
        for name in ("studio.html", "studio-page.js", "studio-page.css"):
            self.assertTrue((ROOT / "src" / "autodub" / "web" / "static" / name).is_file(), name)

    def test_js_endpoints_are_real_routes(self) -> None:
        js = (ROOT / "src" / "autodub" / "web" / "static" / "studio-page.js").read_text(encoding="utf-8")
        server = (ROOT / "src" / "autodub" / "server.py").read_text(encoding="utf-8")
        for marker in ("apply-cast", "speakers/merge", "speakers/reject-merge",
                       "demos/prerender", "arm-gpu", "episode-video", "evidence-video",
                       "evidence-frame", "characters"):
            self.assertIn(marker, js, f"page no longer calls {marker}?")
            self.assertIn(marker.split("/")[-1], server, f"server lost {marker}")
        self.assertIn("studio_progress", server)
        self.assertIn("sanitize_studio_progress", server)

    def test_review_panel_fixes_pinned(self) -> None:
        server = (ROOT / "src" / "autodub" / "server.py").read_text(encoding="utf-8")
        html = (ROOT / "src" / "autodub" / "web" / "static" / "studio.html").read_text(encoding="utf-8")
        js = (ROOT / "src" / "autodub" / "web" / "static" / "studio-page.js").read_text(encoding="utf-8")
        # the render-demos entry point must exist (the panel's top finding)
        self.assertIn('id="render-demos"', html)
        # force must be threaded to prerender_demos (stale-demo purge)
        self.assertIn("force=force", server)
        # PATCH mutations run under the per-job lock (lost-update fix)
        self.assertIn("with job_lock(parts[2]):", server)
        # speaker reassign purges the stale cached wavs
        self.assertIn("reassigned", server)
        # drafts persist: names/voices ride studio_progress
        self.assertIn("draft_cast", js)
        # the vacuous exclusive-timeline verdict is gone; raw evidence drives it
        self.assertIn("simultaneous_speech_longest_run_ms", js)

    def test_job_page_button_navigates_to_the_page(self) -> None:
        entry = (ROOT / "src" / "autodub" / "web" / "static" / "studio.js").read_text(encoding="utf-8")
        self.assertIn("studio.html?job=", entry)

    def test_no_webfonts(self) -> None:
        # the app works fully offline: pages must never fetch web fonts
        for name in ("studio.html", "studio-page.css"):
            text = (ROOT / "src" / "autodub" / "web" / "static" / name).read_text(encoding="utf-8")
            self.assertNotIn("fonts.googleapis", text)
            self.assertNotIn("http://", text.replace("http://127.0.0.1", ""))
            self.assertNotIn("https://", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
