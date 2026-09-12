"""I test E9 analysis with explicitly synthetic numbers, never model results."""
import copy
import csv
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


SPEC = importlib.util.spec_from_file_location("e9_analyze", Path(__file__).with_name("analyze.py"))
analyze = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = analyze
SPEC.loader.exec_module(analyze)


def synthetic_rows(families=analyze.FAMILIES, optional_zeros=False):
    """My fixtures deliberately use unequal token counts to detect pseudoreplication."""
    design = [(condition, "none", "zero") for condition in analyze.CONDITIONS]
    for family in families:
        design.extend(("tool_tagged", family, arm) for arm in (("zero",) if optional_zeros else ()) + analyze.ARMS)
    rows = []
    for conv_ix, conv in enumerate(("synthetic-A", "synthetic-B", "synthetic-C")):
        for condition, family, arm in design:
            for layer in analyze.LAYERS:
                for role in analyze.ROLES:
                    shift = {"zero": 0, "positive": .03, "negative": -.03,
                             "random_0": .005, "random_1": -.004, "random_2": .001}[arm]
                    condition_shift = {"tagged": .05, "untagged": .02, "tool_tagged": 0}[condition]
                    user = .2 + .05 * conv_ix + layer * .001 + shift + condition_shift
                    assistant = .3 + .01 * conv_ix
                    rows.append(dict(conv_id=conv, condition=condition, family=family, arm=arm,
                                     layer_ix=layer, original_role=role, n_tokens=(10, 100, 1000)[conv_ix],
                                     correct_fraction=(.1, .4, .9)[conv_ix] + shift,
                                     p_user=user, p_assistant=assistant, p_tool=1 - user - assistant))
    return rows


def write_fixture(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=analyze.REQUIRED)
        writer.writeheader()
        writer.writerows(rows)


class PairedAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = synthetic_rows()
        cls.data = analyze.validate_rows(cls.rows)
        cls.aggregate, cls.deltas, cls.indices = analyze.paired_statistics(cls.data)

    def find(self, rows, family="none", arm="zero", metric="p_user", condition="tool_tagged", layer=0, role="user"):
        return next(r for r in rows if (r["condition"], r["family"], r["arm"], r["layer_ix"], r["original_role"], r["metric"])
                    == (condition, family, arm, layer, role, metric))

    def test_complete_24_layer_design_and_equal_conversation_weights(self):
        self.assertEqual(self.data.values.shape, (3, 864, 4))
        row = self.find(self.aggregate, metric="correct_fraction")
        self.assertAlmostEqual(row["mean"], (.1 + .4 + .9) / 3)
        self.assertNotAlmostEqual(row["mean"], np.average([.1, .4, .9], weights=[10, 100, 1000]))
        self.assertEqual(row["n_conversations"], 3)
        self.assertEqual(row["n_tokens_total"], 1110)

    def test_bootstrap_matches_direct_shared_conversation_resamples(self):
        expected_indices = np.random.default_rng(123).integers(0, 3, size=(2000, 3))
        np.testing.assert_array_equal(self.indices, expected_indices)
        for cell in [("tool_tagged", "none", "zero", 0, "user"),
                     ("tool_tagged", "historical", "positive", 23, "assistant")]:
            j = self.data.cells.index(cell)
            for k, metric in enumerate(analyze.METRICS):
                direct = self.data.values[self.indices, j, k].mean(axis=1)
                row = self.find(self.aggregate, condition=cell[0], family=cell[1], arm=cell[2], layer=cell[3], role=cell[4], metric=metric)
                np.testing.assert_allclose([row["ci_lo"], row["ci_hi"]], np.percentile(direct, [2.5, 97.5]), atol=1e-14)

    def test_paired_difference_preserves_constant_shift_and_zero_reference(self):
        for family in analyze.FAMILIES:
            row = self.find(self.deltas, family=family, arm="positive")
            np.testing.assert_allclose([row["mean_delta"], row["ci_lo"], row["ci_hi"]], .03, atol=1e-14)
        reference = self.find(self.deltas)
        np.testing.assert_array_equal([reference["mean_delta"], reference["ci_lo"], reference["ci_hi"]], [0, 0, 0])

    def test_missing_cell_layer_role_arm_and_cohort_fail(self):
        variants = [self.rows[1:],
                    [r for r in self.rows if r["layer_ix"] != 23],
                    [r for r in self.rows if r["original_role"] != "assistant"],
                    [r for r in self.rows if r["arm"] != "random_2"],
                    [r for r in self.rows if not (r["conv_id"] == "synthetic-C" and r["family"] == "historical")]]
        for rows in variants:
            with self.subTest(length=len(rows)), self.assertRaisesRegex(ValueError, "missing cell"):
                analyze.validate_rows(rows)

    def test_unknown_family_duplicate_and_token_count_change_fail(self):
        with self.assertRaisesRegex(ValueError, "duplicates"):
            analyze.validate_rows(self.rows + [self.rows[0]])
        rows = copy.deepcopy(self.rows)
        rows[0]["family"] = "unknown"
        with self.assertRaisesRegex(ValueError, "unknown family"):
            analyze.validate_rows(rows)
        rows = copy.deepcopy(self.rows)
        rows[-1]["n_tokens"] += 1
        with self.assertRaisesRegex(ValueError, "token count changes"):
            analyze.validate_rows(rows)

    def test_invalid_metrics_normalization_and_token_count_fail(self):
        for field, value, message in [("p_user", float("nan"), "nonfinite"),
                                      ("p_tool", float("inf"), "nonfinite"),
                                      ("correct_fraction", 1.01, "out-of-range"),
                                      ("p_user", .5, "must equal one"),
                                      ("n_tokens", 0, "no scored"),
                                      ("layer_ix", "1.5", "non-integer")]:
            rows = copy.deepcopy(self.rows)
            rows[0][field] = value
            with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, message):
                analyze.validate_rows(rows)

    def test_optional_family_zero_is_complete_and_equal(self):
        rows = synthetic_rows(optional_zeros=True)
        data = analyze.validate_rows(rows)
        self.assertEqual(data.optional_zero_families, analyze.FAMILIES)
        bad = copy.deepcopy(rows)
        target = next(r for r in bad if r["family"] == "historical" and r["arm"] == "zero")
        target["correct_fraction"] += .01
        with self.assertRaisesRegex(ValueError, "zero differs"):
            analyze.validate_rows(bad)

    def test_output_tables_provenance_and_nonempty_directory_guard(self):
        with tempfile.TemporaryDirectory(prefix="e9-synthetic-") as directory:
            base = Path(directory)
            source = base / "synthetic.csv"
            write_fixture(source, self.rows)
            out = base / "analysis"
            metadata = analyze.run_analysis(source, out, no_plots=True, title="SYNTHETIC TEST DATA")
            self.assertEqual(metadata["bootstrap"]["n_boot"], 2000)
            self.assertEqual(metadata["bootstrap"]["seed"], 123)
            self.assertEqual(metadata["conversation_order"], list(self.data.conversations))
            np.testing.assert_array_equal(np.load(out / "bootstrap-conversation-indices.npy"), self.indices)
            with (out / "paired-deltas.csv").open() as stream:
                saved = list(csv.DictReader(stream))
            self.assertEqual(len(saved), 864 * 4)
            with self.assertRaisesRegex(ValueError, "nonempty"):
                analyze.run_analysis(source, out, no_plots=True)

    def test_source_only_explicit_selection(self):
        rows = synthetic_rows(families=())
        data = analyze.validate_rows(rows, families=())
        self.assertEqual(data.values.shape, (3, 144, 4))

    @unittest.skipUnless(analyze.DEFAULT_FONT.is_file(), "I need the requested TeX Gyre Termes font for this rendering check")
    def test_synthetic_rendering_uses_condition_colors_and_no_ribbons(self):
        from unittest.mock import patch
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot  # I initialize its decorators before mocking Axes methods.
        import matplotlib.axes
        captured = []
        original = matplotlib.axes.Axes.plot

        def remember(axis, *args, **kwargs):
            captured.append(kwargs)
            return original(axis, *args, **kwargs)

        with tempfile.TemporaryDirectory(prefix="e9-synthetic-render-") as directory:
            with patch.object(matplotlib.axes.Axes, "plot", remember), \
                 patch.object(matplotlib.axes.Axes, "fill_between", side_effect=AssertionError("I do not shade confidence intervals")):
                paths, family = analyze.render_figures(self.aggregate, directory, title="SYNTHETIC TEST DATA — NO MODEL RUN")
            self.assertEqual(len(paths), 14)
            self.assertIn("Termes", family)
            self.assertTrue(all(p.is_file() and p.stat().st_size > 1000 for p in paths))
            self.assertEqual({line["color"] for line in captured}, set(analyze.COLORS.values()))
            self.assertTrue(any(line.get("label") == "Random 2" for line in captured))


if __name__ == "__main__":
    unittest.main()
