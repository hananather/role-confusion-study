"""I verify raw evidence, limits, and shutdown with a deterministic fake stream."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from agent_hijacking.backend import ContextLimitError, MLXBackend, has_repetition


class FakeTokenizer:
    pieces = {1: "hello", 2: " world", 3: "<|endoftext|>", 12: "<|call|>", 13: "<|return|>"}

    def encode(self, prompt, **kwargs):
        return list(range(len(prompt)))

    def decode(self, tokens, **kwargs):
        return "".join(self.pieces[token] for token in tokens)


def fake_backend(tokens=(1, 2, 12)):
    backend = MLXBackend.__new__(MLXBackend)
    backend._closed = False
    backend._model = object()
    backend._tokenizer = FakeTokenizer()
    backend._stop_ids = {12: "<|call|>", 13: "<|return|>"}
    backend.max_context_tokens = 100
    backend.prefill_step_size = 8
    backend.top_k = 50
    backend.top_p = 1.0
    backend._mx = SimpleNamespace(
        random=SimpleNamespace(seed=lambda _: None),
        reset_peak_memory=lambda: None,
        get_peak_memory=lambda: 12_000_000_000,
        clear_cache=lambda: None,
    )
    events = {"closed": False, "calls": 0}

    def stream(*args, **kwargs):
        events["calls"] += 1
        events["sampler"] = kwargs["sampler"]
        try:
            kwargs["prompt_progress_callback"](0, len(kwargs["prompt"]))
            kwargs["prompt_progress_callback"](len(kwargs["prompt"]), len(kwargs["prompt"]))
            for i, token in enumerate(tokens[:kwargs["max_tokens"]]):
                reason = "stop" if token in backend._stop_ids else (
                    "length" if i + 1 == kwargs["max_tokens"] else None
                )
                yield SimpleNamespace(token=token, prompt_tps=2, finish_reason=reason)
                if reason is not None:
                    break
        finally:
            events["closed"] = True

    backend._stream_generate = stream
    backend._make_sampler = lambda **kwargs: kwargs
    return backend, events


class BackendTests(unittest.TestCase):
    def test_final_stream_token_is_preserved_in_evidence_and_counts(self):
        backend, events = fake_backend()
        result = backend.generate("ab", seed=1234, max_new_tokens=4)
        self.assertEqual(result.text, "hello world<|call|>")
        self.assertEqual(result.token_ids, [1, 2, 12])
        self.assertEqual(result.generated_tokens, 3)
        self.assertEqual(result.finish_reason, "stop")
        self.assertEqual(result.stop_token, "<|call|>")
        self.assertEqual(result.prompt_elapsed_s, 1.0)
        self.assertEqual(events["sampler"], {"temp": 1.0, "top_k": 50, "top_p": 1.0})
        self.assertTrue(events["closed"])

    def test_endoftext_is_not_an_extra_stop_condition(self):
        backend, _ = fake_backend((1, 3, 2, 13))
        result = backend.generate("ab", seed=1, max_new_tokens=4)
        self.assertEqual(result.text, "hello<|endoftext|> world<|return|>")
        self.assertEqual(result.stop_token, "<|return|>")

    def test_budget_retains_last_token_exactly_once(self):
        backend, _ = fake_backend((1, 2, 12))
        result = backend.generate("ab", seed=1, max_new_tokens=2)
        self.assertEqual(result.token_ids, [1, 2])
        self.assertEqual(result.finish_reason, "length")
        self.assertIsNone(result.stop_token)

    def test_context_limit_rejects_without_starting_inference(self):
        backend, events = fake_backend()
        backend.max_context_tokens = 5
        with self.assertRaises(ContextLimitError):
            backend.generate("abcd", seed=1, max_new_tokens=2)
        self.assertEqual(events["calls"], 0)

    def test_prefill_deadline_returns_empty_timeout_and_closes_stream(self):
        backend, events = fake_backend()
        with patch("agent_hijacking.backend.time.monotonic", side_effect=[0, 3, 3]):
            result = backend.generate("ab", seed=1, max_new_tokens=4, timeout_s=2)
        self.assertEqual(result.finish_reason, "timeout")
        self.assertEqual(result.token_ids, [])
        self.assertIsNone(result.prompt_elapsed_s)
        self.assertTrue(events["closed"])

    def test_token_deadline_preserves_partial_generation(self):
        backend, events = fake_backend()
        with patch("agent_hijacking.backend.time.monotonic", side_effect=[0, 0, 0, 3, 3]):
            result = backend.generate("ab", seed=1, max_new_tokens=4, timeout_s=2)
        self.assertEqual(result.finish_reason, "timeout")
        self.assertEqual(result.text, "hello")
        self.assertEqual(result.token_ids, [1])
        self.assertTrue(events["closed"])

    def test_progress_preserves_raw_partial_output_for_process_watchdog(self):
        backend, _ = fake_backend()
        progress = []
        backend.generate("ab", seed=1, max_new_tokens=4, on_progress=progress.append)
        self.assertEqual(progress[0]["phase"], "prefill")
        self.assertEqual(progress[-1]["token_ids"], [1, 2, 12])
        self.assertEqual(progress[-1]["text"], "hello world<|call|>")

    def test_repetition_is_observational_not_a_stopping_rule(self):
        self.assertTrue(has_repetition(list(range(8)) * 4))
        self.assertFalse(has_repetition(list(range(32))))
        backend, _ = fake_backend([1, 2] * 32 + [12])
        result = backend.generate("ab", seed=1, max_new_tokens=70)
        self.assertTrue(result.repetition_detected)
        self.assertEqual(result.finish_reason, "stop")
        self.assertEqual(result.generated_tokens, 65)

    def test_close_is_idempotent_and_blocks_reuse(self):
        backend, _ = fake_backend()
        backend.close()
        backend.close()
        with self.assertRaises(RuntimeError):
            backend.generate("ab", seed=1)


if __name__ == "__main__":
    unittest.main()
