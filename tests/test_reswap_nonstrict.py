import unittest
from unittest.mock import patch

import reswap as reswap_module


def _assignment(bay):
    return {
        0: {
            "bay_id": bay,
            "x": 0,
            "y": 0,
            "orient_idx": 0,
            "entry_time": 0,
            "exit_time": 1,
        }
    }


class ReswapNonstrictTest(unittest.TestCase):
    def setUp(self):
        self.problem = {
            "weights": {"w1": 1.0, "w2": 1.0, "w3": 1.0},
            "bays": [
                {"width": 10, "height": 10},
                {"width": 10, "height": 10},
            ],
            "blocks": [{
                "release_time": 0,
                "due_date": 2,
                "processing_time": 1,
                "workload": 1.0,
                "bay_preferences": [1.0, 0.0],
                "shape": [{"layers": [[[0, 0], [1, 0], [1, 1]]]}],
            }],
        }

    @staticmethod
    def _objective(_prob, assign):
        objective = 100.0 if assign[0]["bay_id"] == 0 else 105.0
        return objective, (0.0, 0.0, objective)

    @staticmethod
    def _construct(_prob, **kwargs):
        return _assignment(kwargs["plan"][0]["bay_id"]), {}

    @staticmethod
    def _retime(_prob, assign, **_kwargs):
        return None, {
            "assignments": {
                bid: dict(row) for bid, row in assign.items()
            }
        }

    def _run(self, gap, return_terminal=False):
        with (
            patch.object(
                reswap_module, "_objective_of",
                side_effect=self._objective),
            patch.object(
                reswap_module, "construct",
                side_effect=self._construct),
            patch.object(
                reswap_module, "retime",
                side_effect=self._retime),
        ):
            return reswap_module.reswap(
                self.problem, _assignment(0), timelimit=20.0,
                parallel=False, quality_gap=gap,
                return_terminal=return_terminal)

    def test_zero_gap_keeps_historical_strict_round_loop(self):
        result, stats = self._run(0.0)
        self.assertEqual(result[0]["bay_id"], 0)
        self.assertFalse(stats["nonstrict"])
        self.assertEqual(stats["accepted_rounds"], 0)
        self.assertEqual(stats["worsening_rounds"], 0)

    def test_positive_gap_can_traverse_worse_but_returns_protected_best(self):
        result, stats, terminal = self._run(
            0.10, return_terminal=True)
        self.assertEqual(result[0]["bay_id"], 0)
        self.assertEqual(terminal[0]["bay_id"], 1)
        self.assertTrue(stats["nonstrict"])
        self.assertEqual(stats["accepted_rounds"], 1)
        self.assertEqual(stats["worsening_rounds"], 1)
        self.assertEqual(stats["improved_rounds"], 0)
        self.assertEqual(stats["obj"], 100.0)
        self.assertEqual(stats["terminal_obj"], 105.0)


if __name__ == "__main__":
    unittest.main()
