import json
import unittest
from pathlib import Path

import numpy as np

from plan_core import _fits_bay
from plan_lr import solve_plan_lr
from rc_variants import discrepant_plan, reduced_cost_options


class ReducedCostVariantTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        path = root / "instances" / "train" / "prob_27.json"
        cls.prob = json.loads(path.read_text())
        cls.state = {}
        cls.plan, cls.final_lam = solve_plan_lr(
            cls.prob, alpha=0.72, timelimit=1e9, iters=8,
            rc_state=cls.state)
        cls.ranked = reduced_cost_options(
            cls.prob, cls.plan, cls.state["lam"], top_k=4)

    def test_captured_multiplier_shape(self):
        self.assertEqual(self.state["lam"].shape, self.final_lam.shape)
        self.assertEqual(self.state["horizon"], self.final_lam.shape[1])
        self.assertTrue(np.isfinite(self.state["lam"]).all())

    def test_options_are_sorted_and_valid(self):
        self.assertEqual(len(self.ranked), len(self.prob["blocks"]))
        horizon = self.state["horizon"]
        for i, row in self.ranked.items():
            costs = [option["cost"] for option in row["options"]]
            self.assertEqual(costs, sorted(costs))
            block = self.prob["blocks"][i]
            for option in row["options"]:
                self.assertTrue(
                    _fits_bay(block, self.prob["bays"][option["bay_id"]]))
                self.assertGreaterEqual(
                    option["start"], block["release_time"])
                self.assertLessEqual(
                    option["start"] + block["processing_time"], horizon)

    def test_one_bounded_discrepancy(self):
        variant, meta = discrepant_plan(
            self.prob, self.plan, self.ranked, {}, 0)
        self.assertTrue(meta["changed"])
        self.assertEqual(len(meta["chosen"]), 1)
        selected = meta["chosen"][0]["block_id"]
        forced = [i for i, row in variant.items() if row["bay_id"] >= 0]
        self.assertEqual(forced, [selected])
        self.assertNotEqual(
            (variant[selected]["bay_id"], variant[selected]["start"]),
            (self.plan[selected]["bay_id"], self.plan[selected]["start"]))
        for i in self.plan:
            if i != selected:
                self.assertEqual(
                    variant[i]["start"], self.plan[i]["start"])


if __name__ == "__main__":
    unittest.main()
