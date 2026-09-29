"""Readable export names.

Every export gets a human-readable hardlink named from the source file's own name;
the opaque job-ID file stays canonical so app links and existing jobs never break.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from autodub import pipeline, state
from autodub.config import WORK_ROOT
from autodub.state import friendly_output_path, sanitize_export_stem

from tests import ROOT


class ExportNamingTests(unittest.TestCase):
    def test_sanitize_strips_illegal_characters_and_caps_length(self) -> None:
        self.assertEqual(sanitize_export_stem('A<b>:c"d|e?f*g.mkv'), "Abcdefg")
        # path-y names reduce to their basename (uploads may carry directories)
        self.assertEqual(sanitize_export_stem("season1/ep e01.mkv"), "ep e01")
        self.assertEqual(sanitize_export_stem("  spaced.  .mkv  "), "spaced")
        self.assertEqual(len(sanitize_export_stem("x" * 400 + ".mkv")), 150)
        self.assertEqual(sanitize_export_stem(""), "")

    def test_friendly_path_from_source_name(self) -> None:
        job = {"id": "dub-20260822-172012-4a77",
               "source": {"original_name": "Show S01E01 [1080p].mkv"}}
        final = state.output_path(job["id"])
        friendly = friendly_output_path(job, final)
        self.assertEqual(friendly.name, "Show S01E01 [1080p] [ENGLISH DUB].mp4")
        variant = state.output_variant_path(job["id"], "directors")
        self.assertEqual(friendly_output_path(job, variant).name,
                         "Show S01E01 [1080p] [ENGLISH DUB directors].mp4")
        self.assertIsNone(friendly_output_path({"id": "x", "source": {}}, final))

    def test_link_creates_hardlink_and_records_artifact(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORK_ROOT) as temporary:
            out = Path(temporary)
            with patch.object(state, "OUTPUT_ROOT", out):
                final = out / "dub-test-english.mp4"
                final.write_bytes(b"fake-mp4")
                job = {"id": "dub-test", "artifacts": {},
                       "source": {"original_name": "Show S01E05.mkv"}}
                pipeline._link_friendly_output(job, final)
                friendly = out / "Show S01E05 [ENGLISH DUB].mp4"
                self.assertTrue(friendly.is_file())
                self.assertEqual(friendly.read_bytes(), b"fake-mp4")
                self.assertEqual(job["artifacts"]["friendly_output"], friendly.name)
                # re-render refreshes the link instead of failing on the existing name
                pipeline._link_friendly_output(job, final)
                self.assertTrue(friendly.is_file())

    def test_every_export_site_links_and_delete_cleans_up(self) -> None:
        source = (ROOT / "src" / "autodub" / "pipeline.py").read_text(encoding="utf-8")
        # one def + three call sites (CPU render, quality render, remix variant)
        self.assertEqual(source.count("_link_friendly_output("), 4)
        state_src = (ROOT / "src" / "autodub" / "state.py").read_text(encoding="utf-8")
        delete_body = state_src.split("def delete_job", 1)[1].split("\ndef ", 1)[0]
        self.assertIn("friendly_output_path", delete_body)


if __name__ == "__main__":
    unittest.main(verbosity=2)
