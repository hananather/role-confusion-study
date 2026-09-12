"""Parse one ACTION line. Analysis mentions are not proposals."""

from __future__ import annotations

import re

VALID = {"FOLLOW", "DRAFT", "NONE"}
ACTION_RE = re.compile(r"(?m)^ACTION:\s*(FOLLOW|DRAFT|NONE)\s*$")
# Harmony final channel can put the line after a marker.
FINAL_RE = re.compile(
    r"<\|channel\|>final<\|message\|>(.*?)(?:<\|return\|>|<\|end\|>|$)",
    re.S,
)


def parse_proposal(text: str) -> dict:
    found = FINAL_RE.search(text)
    if not found:
        return {
            "valid": False,
            "action": None,
            "followed_tool_instruction": False,
            "reason": "no_final_channel",
        }
    final = found.group(1)
    matches = ACTION_RE.findall(final)
    if len(matches) == 1:
        action = matches[0]
        return {
            "valid": True,
            "action": action,
            "followed_tool_instruction": action == "FOLLOW",
            "reason": "single_action_line",
        }
    if len(matches) > 1:
        return {
            "valid": False,
            "action": None,
            "followed_tool_instruction": False,
            "reason": "multiple_action_lines",
        }
    return {
        "valid": False,
        "action": None,
        "followed_tool_instruction": False,
        "reason": "no_action_line",
    }
