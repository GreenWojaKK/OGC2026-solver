import unittest
from unittest.mock import patch

import alap_batch
import winrep


def _row(bay):
    return {
        "bay_id": int(bay),
        "x": 0,
        "y": 0,
        "orient_idx": 0,
        "entry_time": 0,
        "exit_time": 1,
    }


class AlapBatchWalkTest(unittest.TestCase):
    def setUp(self):
        block = {
            "release_time": 0,
            "due_date": 3,
            "processing_time": 1,
            "workload": 1.0,
            "bay_preferences": [1.0, 0.0],
            "shape": [{
                "layers": [[[0, 0], [1, 0], [1, 1], [0, 1]]],
            }],
        }
        self.problem = {
            "weights": {"w1": 1.0, "w2": 1.0, "w3": 1.0},
            "bays": [
                {"width": 10, "height": 10},
                {"width": 10, "height": 10},
            ],
            "blocks": [dict(block), dict(block)],
        }
        self.start = {0: _row(0), 1: _row(0)}
        self.latest_calls = 0
        self.exact_calls = []

    @staticmethod
    def _objective(_prob, assignments):
        state = tuple(assignments[bid]["bay_id"] for bid in (0, 1))
        objective = {
            (0, 0): 100.0,
            (1, 0): 105.0,
            (0, 1): 110.0,
            (1, 1): 90.0,
        }[state]
        return objective, (0.0, 0.0, objective)

    @staticmethod
    def _construct(_prob, **kwargs):
        result = {
            int(bid): dict(row)
            for bid, row in kwargs["fixed_assignments"].items()
        }
        for bid in kwargs["insert_ids"]:
            planned = kwargs["plan"][bid]["bay_id"]
            result[int(bid)] = _row(0 if planned < 0 else planned)
        return result, {}

    def _retime(self, _prob, assignments, **kwargs):
        if kwargs.get("timing") == "latest":
            self.latest_calls += 1
        return None, {
            "assignments": {
                int(bid): dict(row) for bid, row in assignments.items()
            }
        }

    def _exact_repair(
            self, _lib, _prob, scaffold, off_rows, _budget_left,
            **kwargs):
        forced = {int(bid): int(bay) for bid, bay in off_rows}
        members = tuple(int(bid) for bid in kwargs["explicit_window_ids"])
        self.exact_calls.append({
            "forced": forced,
            "members": members,
            "workers": kwargs["solver_workers"],
            "quality_limit": kwargs["quality_limit"],
            "novelty": kwargs["novelty"],
        })
        result = {
            int(bid): dict(row) for bid, row in scaffold.items()
        }
        for bid in members:
            if bid in forced:
                result[bid] = _row(forced[bid])
        return result

    def _run(
            self, quality_gap, latest_scaffold=True,
            repair_mode="construct"):
        with (
            patch.object(
                alap_batch, "_objective_of",
                side_effect=self._objective),
            patch.object(
                alap_batch, "construct",
                side_effect=self._construct),
            patch.object(
                alap_batch, "retime",
                side_effect=self._retime),
            patch.object(
                winrep, "_rust_lib",
                return_value=object()),
            patch.object(
                winrep, "_repair_once",
                side_effect=self._exact_repair),
        ):
            return alap_batch.alap_batch_walk(
                self.problem,
                self.start,
                timelimit=5.0,
                quality_gap=quality_gap,
                batch_size=1,
                anchors=1,
                max_rounds=3,
                max_proposals_per_round=4,
                candidate_timelimit=0.2,
                latest_scaffold=latest_scaffold,
                repair_mode=repair_mode,
            )

    def test_nonstrict_walk_crosses_worse_state_to_strict_gain(self):
        protected, stats, terminal = self._run(0.10)
        self.assertEqual(
            tuple(protected[bid]["bay_id"] for bid in (0, 1)),
            (1, 1))
        self.assertEqual(
            tuple(terminal[bid]["bay_id"] for bid in (0, 1)),
            (1, 1))
        self.assertEqual(stats["path_objectives"][:3],
                         [100.0, 105.0, 90.0])
        self.assertEqual(stats["worsening_rounds"], 1)
        self.assertEqual(stats["improved_rounds"], 1)
        self.assertGreaterEqual(self.latest_calls, 2)

    def test_zero_gap_does_not_accept_the_bridge(self):
        protected, stats, terminal = self._run(0.0)
        self.assertEqual(protected, self.start)
        self.assertEqual(terminal, self.start)
        self.assertEqual(stats["accepted_rounds"], 0)
        self.assertEqual(stats["protected_objective"], 100.0)

    def test_asap_ablation_does_not_request_latest_schedule(self):
        protected, stats, _ = self._run(
            0.10, latest_scaffold=False)
        self.assertEqual(
            tuple(protected[bid]["bay_id"] for bid in (0, 1)),
            (1, 1))
        self.assertFalse(stats["latest_scaffold"])
        self.assertEqual(stats["alap_rounds"], 0)
        self.assertEqual(self.latest_calls, 0)

    def test_cpsat_repair_uses_explicit_cohort_and_keeps_bridge(self):
        protected, stats, terminal = self._run(
            0.10, repair_mode="cpsat")
        self.assertEqual(
            tuple(protected[bid]["bay_id"] for bid in (0, 1)),
            (1, 1))
        self.assertEqual(
            tuple(terminal[bid]["bay_id"] for bid in (0, 1)),
            (1, 1))
        self.assertEqual(stats["path_objectives"][:3],
                         [100.0, 105.0, 90.0])
        self.assertEqual(stats["repair_mode"], "cpsat")
        self.assertGreaterEqual(stats["cpsat_calls"], 2)
        self.assertEqual(
            stats["cpsat_solved"], stats["cpsat_calls"])
        self.assertEqual(stats["cpsat_calls"], len(self.exact_calls))
        self.assertTrue(all(
            call["workers"] == 1 for call in self.exact_calls))
        self.assertIn((0,), [
            call["members"] for call in self.exact_calls])
        self.assertIn((1,), [
            call["members"] for call in self.exact_calls])

    def test_cpsat_novelty_alternates_exact_and_bridge_rounds(self):
        protected, stats, _ = self._run(
            0.10, repair_mode="cpsat_novelty")
        self.assertEqual(
            tuple(protected[bid]["bay_id"] for bid in (0, 1)),
            (1, 1))
        self.assertEqual(stats["repair_mode"], "cpsat_novelty")
        self.assertGreater(stats["cpsat_novelty_calls"], 0)
        self.assertEqual(
            stats["cpsat_novelty_calls"],
            sum(int(call["novelty"]) for call in self.exact_calls),
        )
        self.assertFalse(self.exact_calls[0]["novelty"])
        self.assertTrue(any(
            call["novelty"] for call in self.exact_calls))
        self.assertTrue(all(
            abs(call["quality_limit"] - 110.0) <= 1e-9
            or abs(call["quality_limit"] - 99.0) <= 1e-9
            for call in self.exact_calls))

if __name__ == "__main__":
    unittest.main()
