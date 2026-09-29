# -*- coding: utf-8 -*-
"""Contracts for state.replace_retry — the Windows save-collision fix.

A plain open() on Windows blocks os.replace with PermissionError, which is exactly how
the UI's 8-second poll broke two season-batch saves and made the end-to-end test flake.
"""
import sys
import tempfile
import threading
import unittest
from pathlib import Path

from autodub.state import replace_retry


class ReplaceRetryContracts(unittest.TestCase):
    def test_survives_a_transient_reader(self):
        with tempfile.TemporaryDirectory() as td:
            dst = Path(td) / "x.json"; dst.write_text("old", encoding="utf-8")
            src = Path(td) / "x.tmp";  src.write_text("new", encoding="utf-8")
            handle = open(dst, "r", encoding="utf-8")
            timer = threading.Timer(0.25, handle.close)
            timer.start()
            try:
                replace_retry(str(src), dst)      # must outlast the 0.25s reader
            finally:
                timer.cancel(); handle.close()
            self.assertEqual(dst.read_text(encoding="utf-8"), "new")

    @unittest.skipUnless(sys.platform == "win32", "only Windows refuses a rename while a reader holds the file")
    def test_raises_when_the_reader_never_leaves(self):
        with tempfile.TemporaryDirectory() as td:
            dst = Path(td) / "x.json"; dst.write_text("old", encoding="utf-8")
            src = Path(td) / "x.tmp";  src.write_text("new", encoding="utf-8")
            handle = open(dst, "r", encoding="utf-8")
            try:
                with self.assertRaises(PermissionError):
                    replace_retry(str(src), dst, attempts=2)
            finally:
                handle.close()

    def test_plain_replace_still_works(self):
        with tempfile.TemporaryDirectory() as td:
            dst = Path(td) / "x.json"; dst.write_text("old", encoding="utf-8")
            src = Path(td) / "x.tmp";  src.write_text("new", encoding="utf-8")
            replace_retry(str(src), dst)
            self.assertEqual(dst.read_text(encoding="utf-8"), "new")


if __name__ == "__main__":
    unittest.main()
