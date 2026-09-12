"""I verify the new attribution cap and preserved baseline contracts without model weights."""
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import json
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
from replication.cloud.agent_steering.test_hf_backend import fake_backend
from prepare_inputs import QUESTION, FINAL_PREFIX, attribution_prompt, spans
from run_queue import validate_attribution

class Tests(unittest.TestCase):
    def generate(self, purpose, cap):
        b = fake_backend()
        with patch.dict("sys.modules", {"transformers": SimpleNamespace(StoppingCriteriaList=lambda x: x)}):
            g, stats = b.generate_steered("abcdef", {"page": [[1, 3]]}, arm_id="none", seed=101238,
                                        purpose=purpose, max_new_tokens=cap, temperature=1.0)
        return b, g, stats

    def test_attribution_200_keeps_sampling_and_no_hooks(self):
        b, g, stats = self.generate("attribution", 200)
        self.assertEqual(b.model.kwargs["max_new_tokens"], 200)
        self.assertEqual((b.model.kwargs["temperature"], b.model.kwargs["top_k"], b.model.kwargs["top_p"]), (1., 50, 1.))
        self.assertEqual(stats["purpose"], "attribution")
        self.assertEqual(stats["hook_calls"], 0)
        self.assertEqual(stats["edited_positions"], 0)
        self.assertEqual(g.generated_tokens, len(g.token_ids))

    def test_only_exact_purpose_cap_pairs_are_accepted(self):
        for purpose, cap in [("episode", 4096), ("engineering_pilot", 64), ("attribution", 200)]:
            self.generate(purpose, cap)
        for purpose, cap in [("episode", 200), ("episode", 64), ("engineering_pilot", 200),
                             ("attribution", 64), ("attribution", 201), ("attribution", 4096), ("other", 200)]:
            with self.subTest(purpose=purpose, cap=cap), self.assertRaises(ValueError): self.generate(purpose, cap)

    def test_prompt_adds_exact_new_user_question_at_tool_boundary(self):
        prefix = '<|start|>functions.bash to=assistant<|channel|>commentary<|message|>We need to consider policy<|end|>'
        saved = prefix + '<|start|>assistant'
        result = attribution_prompt(saved)
        self.assertEqual(result, prefix + '<|start|>user<|message|>' + QUESTION + '<|end|>' + FINAL_PREFIX)
        a, b = spans(result)["page"][0]
        self.assertEqual(result[a:b], 'We need to consider policy')
        self.assertLess(b, result.index(QUESTION))
        with self.assertRaises(ValueError): attribution_prompt(saved+'extra')

    def test_receipt_checks_seed_and_true_200_cap(self):
        _, g, stats = self.generate("attribution", 200)
        sample = {"seed": 101238, "prompt_sha256": stats["prompt_sha256"], "prompt_tokens": g.prompt_tokens}
        validate_attribution(g, stats, sample)
        for key, value in [("seed", 0), ("purpose", "engineering_pilot"), ("max_new_tokens", 64), ("hook_calls", 1)]:
            with self.subTest(key=key), self.assertRaises(ValueError): validate_attribution(g, {**stats, key: value}, sample)

    def test_registration_binds_sources_and_rejects_unclosed_predecessor(self):
        from register import register
        from run_queue import verify_packet
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); closure = root / 'closure.json'
            receipt = {"passed":True,"audit_passed":True,"sync_verified":True,"item4_completed":True,
                "previous_service_exit_verified":True,"previous_run_id":"agent-newpages-20260912T025000Z",
                "completed_jobs":40,"previous_worker_pid":12345}
            closure.write_text(json.dumps(receipt))
            args = SimpleNamespace(closure_receipt=closure, predecessor_pid=12345, out=root/'registration',
                remote_root=Path('/workspace/continuations/test-queue-a'), job_id='test-queue-a', local_results=root/'results')
            with patch('register.time.time', return_value=0): out=register(args)
            config=json.loads((out/'local-config.json').read_text())
            verify_packet(config)
            self.assertTrue(Path(json.loads((out/'launch.json').read_text())['local_argv'][0]).exists())
            from launch_successor import launch, PRECHECK, BOOTSTRAP, STATUS
            for program in (PRECHECK, BOOTSTRAP, STATUS): compile(program, '<fixed-remote-program>', 'exec')
            spec=json.loads((out/'launch.json').read_text());job=json.loads((out/'registration.json').read_text())
            metadata={k:job[k] for k in ('directions_file_sha256','probe_file_sha256')}
            ready={'guardian':{'guardian_pid':987,'worker_pid':988,'config_sha256':spec['service_config_sha256']},
                'service':{'pid':988,'backend':metadata},'job':{'pid':988,'backend':metadata,'job_file_sha256':config['registration_sha256']},'terminal':[None]}
            with patch('launch_successor.time.time',return_value=0), patch('launch_successor.remaining',return_value=3600), \
                    patch('launch_successor.ssh',side_effect=[{'passed':True},{'guardian_pid':987},ready]) as remote, \
                    patch('launch_successor.register_sync') as mirror, \
                    patch('launch_successor.subprocess.run') as copy, \
                    patch('launch_successor.subprocess.Popen',return_value=SimpleNamespace(pid=989)) as spawn, patch('builtins.print'):
                launch(out)
                self.assertEqual(remote.call_count,3)
                self.assertEqual(copy.call_count,2)
                spawn.assert_called_once()
                mirror.assert_called_once_with(config)
            self.assertEqual(json.loads((out/'launch-state/LAUNCHED.json').read_text())['local_runner_pid'],989)
            receipt['previous_service_exit_verified']=False;closure.write_text(json.dumps(receipt))
            args.out=root/'bad-registration'
            with patch('register.time.time', return_value=0), self.assertRaises(ValueError): register(args)

if __name__ == "__main__": unittest.main()
