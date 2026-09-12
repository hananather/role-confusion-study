from __future__ import annotations

import unittest

from role_uptake.prepare import (
    build_manifest,
    load_tokenizer,
    match_instruction_starts,
    match_user_lengths,
    render_prefill,
)
from role_uptake.protocol import AUTHORIZATION, CONDITIONS, SITUATIONS, USER_BY_AUTH, tool_body


class PrepareTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = load_tokenizer()
        cls.manifest = build_manifest(cls.tokenizer)

    def test_twenty_four_cases(self):
        self.assertEqual(self.manifest["n_cases"], 24)
        self.assertEqual(len(self.manifest["cases"]), 24)

    def test_instruction_start_and_ids_match_inside_each_situation(self):
        for situation in SITUATIONS:
            cases = [
                case
                for case in self.manifest["cases"]
                if case["situation_id"] == situation["id"] and case["instruction"]
            ]
            starts = {case["instruction_token_start"] for case in cases}
            ids = {tuple(case["instruction_token_ids"]) for case in cases}
            self.assertEqual(len(starts), 1, situation["id"])
            self.assertEqual(len(ids), 1, situation["id"])
            self.assertGreater(next(iter(starts)), 0)

    def test_authorization_pair_shares_tool_text(self):
        for situation in SITUATIONS:
            by_id = {
                case["condition"]: case
                for case in self.manifest["cases"]
                if case["situation_id"] == situation["id"]
            }
            self.assertEqual(
                by_id["legitimate_delegation"]["tool_text"],
                by_id["ordinary_injection"]["tool_text"],
            )
            self.assertEqual(
                by_id["authorized_reasoning"]["tool_text"],
                by_id["cot_forgery"]["tool_text"],
            )
            self.assertNotEqual(
                by_id["legitimate_delegation"]["user_text"],
                by_id["ordinary_injection"]["user_text"],
            )
            self.assertNotIn(situation["instruction"], by_id["ordinary_task"]["tool_text"])

    def test_user_length_matching_does_not_drop_authorization(self):
        users = match_user_lengths(
            self.tokenizer,
            {condition: USER_BY_AUTH[AUTHORIZATION[condition]] for condition in CONDITIONS},
        )
        self.assertIn("draft", users["ordinary_injection"].lower())
        self.assertIn("carry out", users["legitimate_delegation"].lower())
        for condition, original in {
            condition: USER_BY_AUTH[AUTHORIZATION[condition]] for condition in CONDITIONS
        }.items():
            self.assertTrue(
                users[condition].endswith(original) or original in users[condition]
            )
            self.assertNotIn("aYou", users[condition])
            self.assertNotIn("aPlease", users[condition])


class InstructionPadTests(unittest.TestCase):
    def test_plain_catches_up_to_forged_start(self):
        tokenizer = load_tokenizer()
        situation = SITUATIONS[0]
        tools = {condition: tool_body(situation, condition) for condition in CONDITIONS}
        users = match_user_lengths(
            tokenizer,
            {condition: USER_BY_AUTH[AUTHORIZATION[condition]] for condition in CONDITIONS},
        )
        matched = match_instruction_starts(
            tokenizer, tools, users, situation["instruction"]
        )
        starts = []
        for condition in CONDITIONS:
            if situation["instruction"] not in matched[condition]:
                continue
            prefill = render_prefill(users[condition], matched[condition])
            self.assertEqual(prefill.count(situation["instruction"]), 1)
            starts.append(prefill.find(situation["instruction"]))
        self.assertTrue(starts)
        # Character starts need not match; token starts are checked in PrepareTests.


if __name__ == "__main__":
    unittest.main()
