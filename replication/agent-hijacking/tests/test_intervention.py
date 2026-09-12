from __future__ import annotations

import unittest

import numpy as np

from agent_hijacking.intervention import STEER_LAYER, SourceLocalHook, SteeringSpec


class IdentityBlock:
    def __call__(self, x, mask=None, cache=None):
        return x


class InterventionTests(unittest.TestCase):
    def run_chunks(self, hook, widths, width_d=4):
        outputs = []
        for width in widths:
            chunk = np.zeros((1, width, width_d), dtype=np.float32)
            outputs.append(np.array(hook(chunk)))
        return outputs

    def test_zero_dose_is_identity(self):
        hook = SourceLocalHook(IdentityBlock(), layer=STEER_LAYER)
        hook.reset(SteeringSpec(
            dose=0.0, vector=[1, 0, 0, 0], external_indices=[0, 1], prompt_length=4,
        ))
        out = self.run_chunks(hook, [4])[0]
        self.assertEqual(hook.edits, 0)
        np.testing.assert_array_equal(out, np.zeros((1, 4, 4), dtype=np.float32))

    def test_one_addition_per_selected_token_across_512_boundary(self):
        hook = SourceLocalHook(IdentityBlock(), layer=STEER_LAYER)
        vec = [2.0, 0, 0, 0]
        mask = [511, 512]
        hook.reset(SteeringSpec(
            dose=1.0, vector=vec, external_indices=mask, prompt_length=513,
        ))
        first, second = self.run_chunks(hook, [512, 1])
        self.assertEqual(hook.chunk_sizes, [512, 1])
        self.assertEqual(hook.edits, 2)
        self.assertEqual(float(first[0, 511, 0]), 2.0)
        self.assertEqual(float(first[0, 510, 0]), 0.0)
        self.assertEqual(float(second[0, 0, 0]), 2.0)

    def test_last_prompt_token_and_generated_token(self):
        hook = SourceLocalHook(IdentityBlock(), layer=STEER_LAYER)
        hook.reset(SteeringSpec(
            dose=1.0, vector=[3.0, 0, 0, 0], external_indices=[9], prompt_length=10,
        ))
        head, last_prompt, generated = self.run_chunks(hook, [9, 1, 1])
        self.assertEqual(float(head[0, 0, 0]), 0.0)
        self.assertEqual(float(last_prompt[0, 0, 0]), 3.0)
        self.assertEqual(float(generated[0, 0, 0]), 0.0)
        self.assertEqual(hook.edits, 1)

    def test_trusted_indices_are_not_edited(self):
        hook = SourceLocalHook(IdentityBlock(), layer=STEER_LAYER)
        hook.reset(SteeringSpec(
            dose=1.0, vector=[1.0, 0, 0, 0], external_indices=[2], prompt_length=4,
        ))
        out = self.run_chunks(hook, [4])[0]
        self.assertEqual(float(out[0, 2, 0]), 1.0)
        self.assertEqual(float(out[0, 0, 0]), 0.0)
        self.assertEqual(float(out[0, 3, 0]), 0.0)


if __name__ == "__main__":
    unittest.main()
