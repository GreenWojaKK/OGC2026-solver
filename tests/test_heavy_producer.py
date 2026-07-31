import os
import unittest
from unittest.mock import patch

from myalgorithm import _heavy_portfolio_arm_configs


class HeavyProducerTests(unittest.TestCase):
    def _legacy_arms(self):
        common = {
            "OGC_WORK_SCALE": "0.90",
            "OGC_ARM_THREADS": "1",
        }
        return [
            dict(common, OGC_ARM="0"),
            dict(common, OGC_ARM="1", OGC_PLAN_BUCKET="5"),
            dict(common, OGC_ARM="2", OGC_AEFF_SCALE="0.92"),
            dict(
                common,
                OGC_ARM="3",
                OGC_ARM_SEED="1",
                OGC_NO_KARM="1",
            ),
        ]

    def test_production_menu_keeps_lr_axes_and_disables_primal(self):
        source = self._legacy_arms()
        for arm in source:
            arm.update({
                "OGC_PRIMAL_REALIZATION": "1",
                "OGC_PRIMAL_JITTER_DAYS": "4",
                "OGC_PRIMAL_SEED": "109",
                "OGC_PRIMAL_WINDOW": "9",
                "OGC_PRIMAL_BAY_WEIGHT": "0.30",
            })
        with patch.dict(os.environ, {}, clear=True):
            arms = _heavy_portfolio_arm_configs(
                source, menu="lr_qd_lns4")

        self.assertEqual(len(arms), 4)
        self.assertTrue(all(
            arm["OGC_PRIMAL_REALIZATION"] == "0" for arm in arms))
        for key in (
                "OGC_PRIMAL_JITTER_DAYS",
                "OGC_PRIMAL_SEED",
                "OGC_PRIMAL_WINDOW",
                "OGC_PRIMAL_BAY_WEIGHT"):
            self.assertTrue(all(key not in arm for arm in arms))
        self.assertEqual(arms[1]["OGC_PLAN_BUCKET"], "5")
        self.assertEqual(arms[2]["OGC_AEFF_SCALE"], "0.92")
        self.assertEqual(arms[3]["OGC_NO_KARM"], "1")

if __name__ == "__main__":
    unittest.main()
