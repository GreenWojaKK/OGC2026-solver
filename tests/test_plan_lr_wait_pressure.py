import unittest

import numpy as np

from plan_lr import (
    _schedule_zobrist,
    _schedule_zobrist_token,
    solve_plan_lr,
)


def _block(release, due):
    return {
        "release_time": int(release),
        "due_date": int(due),
        "processing_time": 10,
        "workload": 1,
        "bay_preferences": [0],
        "shape": [{
            "orientation": 0,
            "layers": [[
                [0.0, 0.0],
                [10.0, 0.0],
                [10.0, 6.0],
                [0.0, 6.0],
            ]],
        }],
    }


def _problem():
    # Each block occupies area 60 in a capacity-100 bay.  Their relaxed
    # release-time targets overlap, so capacity repair must reject at least
    # one target window and produce a bay/day blocked signal.
    return {
        "weights": {"w1": 1, "w2": 0, "w3": 0},
        "bays": [{"width": 10, "height": 10}],
        "blocks": [_block(0, 10), _block(5, 15)],
    }


class PlanLrWaitPressureTest(unittest.TestCase):
    def test_schedule_zobrist_is_order_independent_and_xor_updatable(self):
        plan = {
            0: {"bay_id": 0, "start": 2},
            1: {"bay_id": 1, "start": 7},
        }
        reversed_plan = {
            1: {"bay_id": 1, "start": 7},
            0: {"bay_id": 0, "start": 2},
        }
        moved = {
            0: {"bay_id": 0, "start": 2},
            1: {"bay_id": 1, "start": 8},
        }
        original_hash = _schedule_zobrist(plan)
        moved_hash = _schedule_zobrist(moved)
        self.assertEqual(original_hash, _schedule_zobrist(reversed_plan))
        self.assertNotEqual(original_hash, moved_hash)
        for lane in range(2):
            incrementally_moved = (
                original_hash[lane]
                ^ _schedule_zobrist_token(1, 1, 7, lane)
                ^ _schedule_zobrist_token(1, 1, 8, lane)
            )
            self.assertEqual(incrementally_moved, moved_hash[lane])

    def test_wait_dispatch_collects_target_window_blocked_pressure(self):
        stats = {}
        plan, lam = solve_plan_lr(
            _problem(),
            alpha=1.0,
            timelimit=1e9,
            iters=3,
            stats=stats,
            relative_dispatch=True,
            blocked_weight=0.15,
            zobrist_dedup=True,
        )
        self.assertEqual(len(plan), 2)
        self.assertTrue(np.isfinite(lam).all())
        self.assertEqual(stats["dispatch"], "wait_from_release")
        self.assertEqual(stats["blocked_weight"], 0.15)
        self.assertTrue(stats["zobrist_dedup"])
        self.assertGreaterEqual(stats["blocked_unique_states"], 1)
        self.assertGreater(stats["blocked_signal_days"], 0)
        self.assertGreater(stats["blocked_signal_area"], 0.0)

    def test_disabled_options_preserve_the_baseline_path(self):
        baseline, baseline_lam = solve_plan_lr(
            _problem(), alpha=1.0, timelimit=1e9, iters=3)
        explicit, explicit_lam = solve_plan_lr(
            _problem(),
            alpha=1.0,
            timelimit=1e9,
            iters=3,
            relative_dispatch=False,
            blocked_weight=0.0,
            zobrist_dedup=False,
        )
        self.assertEqual(baseline, explicit)
        np.testing.assert_array_equal(baseline_lam, explicit_lam)


if __name__ == "__main__":
    unittest.main()
