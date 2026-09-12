"""I check that incomplete data and shortened gates cannot become positive evidence."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from .runner import full_identity, summary, technical_gate


class RunnerEvidenceTests(unittest.TestCase):
    def test_unexposed_completed_episode_remains_unresolved(self):
        result = summary({"arms": [{"arm_id": "role_a16"}]}, [
            {"arm_id": "role_a16", "status": "completed", "exposure_confirmed": False,
             "verified_dummy_upload": False, "censored": False}])
        self.assertEqual(result["arms"][0]["unresolved_or_unrun"], 5)

    def test_verified_upload_survives_later_timeout(self):
        result = summary({"arms": [{"arm_id": "role_a16"}]}, [
            {"arm_id": "role_a16", "status": "generation_timeout", "exposure_confirmed": True,
             "verified_dummy_upload": True, "censored": True}])
        self.assertEqual(result["arms"][0]["verified_uploads"], 1)
        self.assertEqual(result["arms"][0]["unresolved_or_unrun"], 4)

    def test_shortened_engineering_gate_cannot_pass(self):
        proof = {"samples": [{"case_id": str(i)} for i in range(5)]}
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                technical_gate(None, [], Path(tmp), proof, 616.126)

    def test_changed_engineering_prompt_cannot_pass(self):
        proof = {"samples": [{"case_id": str(i), "prompt_sha256": hashlib.sha256(b"original").hexdigest(),
                              "page_tokens": 4} for i in range(5)]}
        diagnostics = [{"case_id": str(i), "prompt": "changed", "expected_page_tokens": 4} for i in range(5)]
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                technical_gate(None, diagnostics, Path(tmp), proof, 616.126)

    def test_identity_distinguishes_changed_input_from_changed_sampling(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for arm, token in [("none", 1), ("zero", 2)]:
                path = root / "episodes" / "c" / arm
                path.mkdir(parents=True)
                (path / "episode.json").write_text("{}")
                (path / "step-00.prompt.txt").write_text("same prompt")
                (path / "step-00.generation.json").write_text(json.dumps({"token_ids": [token]}))
            report = full_identity(root, [{"id": "c"}])
            self.assertTrue(report["unexplained_identity_failure"])
            (root / "episodes/c/none/step-00.generation.json").write_text(json.dumps({"token_ids": [1, 2], "finish_reason": "stop"}))
            (root / "episodes/c/zero/step-00.generation.json").write_text(json.dumps({"token_ids": [1], "finish_reason": "timeout"}))
            report = full_identity(root, [{"id": "c"}])
            self.assertFalse(report["unexplained_identity_failure"])
            self.assertTrue(report["comparisons"][0]["generations"][0]["censored_length_difference"])
            (root / "episodes/c/zero/step-00.prompt.txt").write_text("different tool output")
            report = full_identity(root, [{"id": "c"}])
            self.assertFalse(report["unexplained_identity_failure"])
            self.assertFalse(report["comparisons"][0]["fully_comparable"])


if __name__ == "__main__":
    unittest.main()
