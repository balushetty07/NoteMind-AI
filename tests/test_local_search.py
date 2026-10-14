"""Unit tests for NoteMind AI's local keyword-search feature."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import app


class LocalSearchTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.docs_dir = Path(self.temp_dir.name)
        (self.docs_dir / "dsp_notes.txt").write_text(
            "Fourier transform maps a signal to its frequency components.\n"
            "The discrete Fourier transform is used in DSP.\n",
            encoding="utf-8",
        )
        (self.docs_dir / "control.md").write_text(
            "# Control Systems\nA transfer function describes a system.\n",
            encoding="utf-8",
        )
        (self.docs_dir / "ignore.pdf").write_bytes(b"not a searchable text note")
        self.docs_patch = patch.object(app, "DOCS_DIR", self.docs_dir)
        self.docs_patch.start()

    def tearDown(self):
        self.docs_patch.stop()
        self.temp_dir.cleanup()

    def test_lists_only_supported_note_types(self):
        names = [path.name for path in app.get_note_files()]
        self.assertEqual(names, ["control.md", "dsp_notes.txt"])

    def test_search_is_case_insensitive(self):
        results = app.find_local_matches("TRANSFER FUNCTION")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["source"], "control.md")
        self.assertIn("transfer function", results[0]["text"].casefold())

    def test_phrase_can_match_multiple_lines(self):
        results = app.find_local_matches("Fourier transform")
        self.assertEqual(len(results), 2)
        self.assertTrue(all(result["source"] == "dsp_notes.txt" for result in results))

    def test_blank_query_returns_no_results(self):
        self.assertEqual(app.find_local_matches("   "), [])


if __name__ == "__main__":
    unittest.main()
