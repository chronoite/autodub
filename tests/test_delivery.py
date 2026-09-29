from __future__ import annotations

import math
import struct
import tempfile
import unittest
import wave
from pathlib import Path

from autodub.config import WORK_ROOT
from autodub.delivery import analyze_delivery, chatterbox_controls


class DeliveryContractTests(unittest.TestCase):
    def test_relative_energy_hints_are_conservative_and_modular(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORK_ROOT) as temporary:
            audio = Path(temporary) / "delivery.wav"
            rate = 8000
            amplitudes = (300, 3000, 15000)
            with wave.open(str(audio), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(rate)
                for amplitude in amplitudes:
                    frames = b"".join(
                        struct.pack("<h", round(amplitude * math.sin(2 * math.pi * 220 * i / rate)))
                        for i in range(rate)
                    )
                    handle.writeframes(frames)
            segments = [
                {"i": index, "start": float(index), "end": float(index + 1)}
                for index in range(3)
            ]
            summary = analyze_delivery(audio, segments)
            self.assertEqual(summary["status"], "ready")
            self.assertEqual(
                [item["delivery"]["label"] for item in segments],
                ["calm", "neutral", "intense"],
            )
            self.assertLess(
                chatterbox_controls(segments[0])["exaggeration"],
                chatterbox_controls(segments[2])["exaggeration"],
            )
            uncertain = {"delivery": {"label": "intense", "confidence": 0.1}}
            self.assertEqual(chatterbox_controls(uncertain)["exaggeration"], 0.5)


if __name__ == "__main__":
    unittest.main()
