import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import needs_reply_label as annotation


class AnnotationTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(hasattr(annotation, "AnnotationSession"), "AnnotationSession must support canonical JSON labels")
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.items = self.root / "items.json"
        self.labels = self.root / "labels.json"
        self.items.write_text(json.dumps({"items": [
            {"id": 12, "text": "<script>alert(1)</script> **literal**", "stars": 1,
             "draft": {"needs_reply": True, "reasons": ["defect"]}, "answer": "secret"},
            {"id": 13, "text": "second"}]}), encoding="utf-8")
        self.original = {"note": "keep metadata", "labels": [
            {"id": "other-batch", "need": True, "reasons": [], "unsure": False},
            {"id": 13, "need": False, "reasons": ["wb_logistics"], "unsure": True}]}
        self.labels.write_text(json.dumps(self.original), encoding="utf-8")
        self.session = annotation.AnnotationSession(self.items, self.labels)

    def test_blind_resume_and_existing_label_navigation(self):
        self.assertEqual(self.session.first_open(), 0)
        self.assertEqual(self.session.show(0), {
            "id": "12", "text": "<script>alert(1)</script> **literal**", "need": None,
            "reasons": [], "unsure": False, "completed": 1, "total": 2})
        saved = self.session.show(1)
        self.assertIs(saved["need"], False)
        self.assertEqual(saved["reasons"], ["wb_logistics"])
        self.assertTrue(saved["unsure"])

    def test_need_independent_from_reasons_and_preserves_other_batches(self):
        self.session.save(0, False, ["defect", "wb_logistics"], True)
        payload = json.loads(self.labels.read_text(encoding="utf-8"))
        self.assertEqual(payload["note"], "keep metadata")
        self.assertEqual(payload["labels"][0], self.original["labels"][0])
        record = payload["labels"][-1]
        self.assertEqual(record["id"], "12")
        self.assertIs(record["need"], False)
        self.assertEqual(record["reasons"], ["defect", "wb_logistics"])
        self.assertTrue(record["unsure"])
        self.assertIn("at", record)
        self.session.save(0, True, [], False)
        self.assertEqual(self.session.show(0)["reasons"], [])
        self.assertIs(self.session.show(0)["need"], True)
        self.assertEqual(self.session.first_open(), 0)

    def test_backups_retain_previous_bytes_and_failed_replace_preserves_labels(self):
        before = self.labels.read_bytes()
        self.session.save(0, False, [], False)
        backups = list(self.root.glob(".annotation-backups/labels.json.*.bak"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_bytes(), before)
        saved = self.labels.read_bytes()
        with patch("annotation_store.os.replace", side_effect=OSError("disk error")):
            with self.assertRaises(OSError):
                self.session.save(0, True, [], False)
        self.assertEqual(self.labels.read_bytes(), saved)
        self.assertEqual(list(self.root.glob("*.tmp")), [])

    def test_invalid_input_never_changes_file(self):
        before = self.labels.read_bytes()
        for need, reasons, unsure in [(None, [], False), (1, [], False),
                                       (False, ["unknown"], False), (False, [], "yes")]:
            with self.subTest(need=need, reasons=reasons, unsure=unsure):
                with self.assertRaises(ValueError):
                    self.session.save(0, need, reasons, unsure)
                self.assertEqual(self.labels.read_bytes(), before)

    def test_reloads_disk_before_saving_and_rejects_corrupt_labels(self):
        external = self.original.copy()
        external["labels"] = self.original["labels"] + [
            {"id": "external", "need": False, "reasons": [], "unsure": False}]
        self.labels.write_text(json.dumps(external), encoding="utf-8")
        self.session.save(0, False, [], False)
        self.assertIn("external", [r["id"] for r in json.loads(self.labels.read_text())["labels"]])
        self.labels.write_text('{"labels": [{"id": "x", "need": "false", "reasons": []}]}')
        before = self.labels.read_bytes()
        with self.assertRaises(ValueError):
            self.session.save(0, False, [], False)
        self.assertEqual(self.labels.read_bytes(), before)

    def test_duplicate_string_ids_are_rejected(self):
        self.items.write_text(json.dumps({"items": [{"id": 1, "text": "a"}, {"id": "1", "text": "b"}]}))
        with self.assertRaises(ValueError):
            annotation.AnnotationSession(self.items, self.labels)

    def test_interprocess_saves_preserve_both_new_labels(self):
        import subprocess
        import sys
        self.labels.write_text(json.dumps({"labels": self.original["labels"][:1]}), encoding="utf-8")
        # Both processes start from the same canonical file and save different ids.
        script = ("from annotation_store import AnnotationSession; import sys; "
                  "s=AnnotationSession(sys.argv[1],sys.argv[2]); "
                  "s.save(int(sys.argv[3]),False,[],False)")
        jobs = [subprocess.Popen([sys.executable, "-c", script, str(self.items), str(self.labels), str(i)])
                for i in (0, 1)]
        for job in jobs:
            self.assertEqual(job.wait(timeout=20), 0)
        records = json.loads(self.labels.read_text(encoding="utf-8"))["labels"]
        self.assertEqual({r["id"] for r in records}, {"12", "13", "other-batch"})


if __name__ == "__main__":
    unittest.main()
