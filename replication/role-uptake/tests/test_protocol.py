from __future__ import annotations

import unittest

import numpy as np

from role_uptake.protocol import CONDITIONS, SITUATIONS, tool_body
from role_uptake.score import random_probe, softmax, span_means


class ProtocolTests(unittest.TestCase):
    def test_instruction_identical_inside_situation(self):
        for situation in SITUATIONS:
            texts = []
            for condition in CONDITIONS:
                body = tool_body(situation, condition)
                if situation["instruction"] in body:
                    texts.append(situation["instruction"])
            self.assertEqual(len(set(texts)), 1)
            self.assertEqual(texts[0], situation["instruction"])

    def test_ordinary_task_has_no_instruction(self):
        for situation in SITUATIONS:
            self.assertNotIn(situation["instruction"], tool_body(situation, "ordinary_task"))


class ScoreTests(unittest.TestCase):
    def test_softmax_rows_sum_to_one(self):
        probs = softmax(np.array([[0.0, 0.0, 0.0]], dtype=np.float32))
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=6)

    def test_random_probe_is_deterministic(self):
        coef = np.ones((5, 8), dtype=np.float32)
        intercept = np.zeros(5, dtype=np.float32)
        a = random_probe(coef, intercept, seed=0)[0]
        b = random_probe(coef, intercept, seed=0)[0]
        np.testing.assert_array_equal(a, b)

    def test_empty_span(self):
        means = span_means(np.ones((4, 5)), ["system", "user", "cot", "assistant", "tool"], [])
        self.assertEqual(means["n_tokens"], 0)
        self.assertIsNone(means["p_user"])


if __name__ == "__main__":
    unittest.main()
