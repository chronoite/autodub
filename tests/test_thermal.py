"""Thermal reading contracts: parsing, fail-closed behaviour and retries."""
from __future__ import annotations

import subprocess
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autodub import thermal  # noqa: E402

# Human-readable format accepted from AUTODUB_GPU_TEMP_COMMAND.
SENSOR_OUTPUT = """GPU thermals
  core                 46 C
  hot spot           58.8 C
  memory junction      58 C
  power             90.53 W  (limit 390.00 W)
"""


class ThermalParseTests(unittest.TestCase):
    def test_parses_sensor_command_output(self) -> None:
        reading = thermal.parse(SENSOR_OUTPUT)
        self.assertTrue(reading["ok"])
        self.assertEqual(reading["core"], 46.0)
        self.assertEqual(reading["hot_spot"], 58.8)
        self.assertEqual(reading["junction"], 58.0)
        self.assertEqual(reading["peak"], 58.8)
        self.assertIn("junction 58C", thermal.summary(reading))

    def test_parses_nvidia_smi_and_takes_the_hottest_card(self) -> None:
        reading = thermal.parse_nvidia_smi("61\n74\n")
        self.assertTrue(reading["ok"])
        self.assertEqual(reading["core"], 74.0)
        self.assertEqual(reading["peak"], 74.0)
        self.assertIsNone(reading["junction"])

    def test_unreadable_outputs_fail_closed(self) -> None:
        for output in ("", "ERROR: sensor library could not be loaded", "power 90 W"):
            reading = thermal.parse(output)
            self.assertFalse(reading["ok"], repr(output))
            self.assertIsNone(reading["peak"])
        for output in ("", "[N/A]", "NVIDIA-SMI has failed"):
            self.assertFalse(thermal.parse_nvidia_smi(output)["ok"], repr(output))
        self.assertEqual(thermal.summary(thermal.parse("")), "gpu temps unreadable")

    def test_read_temps_survives_subprocess_failure_without_raising(self) -> None:
        with (
            patch.object(thermal.subprocess, "run",
                         side_effect=subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=60)),
            patch.object(thermal.time, "sleep"),
        ):
            reading = thermal.read_temps()
        self.assertFalse(reading["ok"])
        self.assertIn("TimeoutExpired", reading["raw"])

    def test_read_temps_retries_then_succeeds(self) -> None:
        results = [
            types.SimpleNamespace(stdout="", stderr=""),
            types.SimpleNamespace(stdout=SENSOR_OUTPUT, stderr=""),
        ]
        with (
            patch.object(thermal.config, "GPU_TEMP_COMMAND", "sensors-tool"),
            patch.object(thermal.subprocess, "run", side_effect=results),
            patch.object(thermal.time, "sleep"),
        ):
            reading = thermal.read_temps()
        self.assertTrue(reading["ok"])
        self.assertEqual(reading["junction"], 58.0)

    def test_default_source_is_nvidia_smi(self) -> None:
        with patch.object(thermal.config, "GPU_TEMP_COMMAND", ""):
            command, parser = thermal._command()
        self.assertIn("--query-gpu=temperature.gpu", command)
        self.assertIs(parser, thermal.parse_nvidia_smi)


if __name__ == "__main__":
    unittest.main(verbosity=2)
