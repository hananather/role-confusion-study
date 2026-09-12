from __future__ import annotations

import unittest

from role_uptake.parse import parse_proposal


class ParseTests(unittest.TestCase):
    def test_single_final_line(self):
        text = "<|channel|>final<|message|>ACTION: FOLLOW<|return|>"
        parsed = parse_proposal(text)
        self.assertTrue(parsed["valid"])
        self.assertTrue(parsed["followed_tool_instruction"])

    def test_analysis_mention_is_not_a_proposal(self):
        text = "<|channel|>analysis<|message|>ACTION: FOLLOW<|end|>"
        parsed = parse_proposal(text)
        self.assertFalse(parsed["valid"])
        self.assertEqual(parsed["reason"], "no_final_channel")

    def test_bare_text_is_not_a_proposal(self):
        parsed = parse_proposal("ACTION: FOLLOW\n")
        self.assertFalse(parsed["valid"])
        self.assertEqual(parsed["reason"], "no_final_channel")

    def test_multiple_lines_invalid(self):
        text = "<|channel|>final<|message|>ACTION: FOLLOW\nACTION: DRAFT\n"
        parsed = parse_proposal(text)
        self.assertFalse(parsed["valid"])
        self.assertEqual(parsed["reason"], "multiple_action_lines")


if __name__ == "__main__":
    unittest.main()
