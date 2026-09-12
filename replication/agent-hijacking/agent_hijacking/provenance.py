"""I keep Harmony bytes identical while recording trusted vs external spans.

Masks are built from these controller records, never from role tags inside
tool text. Forged Harmony markers in a tool body stay external.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import protocol as p


@dataclass(frozen=True)
class Segment:
    kind: str
    text: str
    trusted: bool
    external: bool


def render_segments(segments: list[Segment]) -> str:
    return "".join(segment.text for segment in segments)


def initial_segments(user_prompt: str, developer_prompt: str = p.DEVELOPER_PROMPT) -> list[Segment]:
    return [
        Segment("system", p.h_system(p.SYSTEM_PROMPT), True, False),
        Segment("developer", p.h_developer(developer_prompt), True, False),
        Segment("user", p.h_user(user_prompt), True, False),
    ]


def assistant_analysis_segment(content: str) -> Segment:
    return Segment("assistant_analysis", p.h_assistant_analysis(content), True, False)


def assistant_final_segment(content: str) -> Segment:
    return Segment("assistant_final", p.h_assistant_final(content), True, False)


def tool_call_segment(tool_fqn: str, json_args: str) -> Segment:
    return Segment("assistant_tool_call", p.h_tool_call(tool_fqn, json_args), True, False)


def tool_result_segments(tool_fqn: str, json_output: str) -> list[Segment]:
    """I split the wrapper from the tool body so only the body is external."""
    wrapped = p.h_tool_result(tool_fqn, json_output)
    prefix = f"<|start|>{tool_fqn} to=assistant<|channel|>commentary<|message|>"
    suffix = "<|end|>"
    if not wrapped.startswith(prefix) or not wrapped.endswith(suffix):
        raise ValueError("tool result wrapper drifted from protocol.py")
    if wrapped[len(prefix) : -len(suffix)] != json_output:
        raise ValueError("tool result body is not a contiguous wrapper payload")
    return [
        Segment("tool_header", prefix, True, False),
        Segment("tool_content", json_output, False, True),
        Segment("tool_end", suffix, True, False),
    ]


def char_spans(segments: list[Segment], *, external_only: bool = True) -> list[tuple[int, int, str]]:
    spans = []
    cursor = 0
    for segment in segments:
        start = cursor
        end = cursor + len(segment.text)
        include = segment.external if external_only else True
        if include:
            spans.append((start, end, segment.kind))
        cursor = end
    return spans


def token_indices(tokenizer, prompt: str, spans: list[tuple[int, int, str]]) -> dict[str, list[int]]:
    encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    out: dict[str, list[int]] = {}
    for start, end, kind in spans:
        indices = [
            i
            for i, (ts, te) in enumerate(encoded.offset_mapping)
            if te > ts and ts < end and te > start
        ]
        out.setdefault(kind, [])
        out[kind].extend(indices)
    return out


def external_token_mask(tokenizer, segments: list[Segment]) -> list[int]:
    prompt = render_segments(segments)
    grouped = token_indices(tokenizer, prompt, char_spans(segments))
    indices = sorted({i for values in grouped.values() for i in values})
    return indices


def assistant_prefix_segment() -> Segment:
    return Segment("assistant_prefix", p.ASSISTANT_PREFIX, True, False)
