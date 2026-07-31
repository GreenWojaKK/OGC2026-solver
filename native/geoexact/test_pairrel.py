"""test_pairrel.py -- parity gate for the gpr_core native intersects screen.

For each instance: one exact construct() pass gives a realistic layout;
every same-bay pair is then classified twice -- native screen armed vs the
pure-GEOS reference (screen disarmed in-process) -- and the relation tuples
must be identical, FREE pairs included.  Also reports the screen's exact-
compute/memo counters.

Usage:  PKG=<dir> python3 test_pairrel.py prob_12 prob_21 prob_30 prob_39
"""
import json
import os
import sys
import time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
PKG = os.environ.get("PKG", os.path.join(ROOT, "solver"))
sys.path.insert(0, PKG)

import gpr_core  # noqa: E402
from constructor import construct, _NATIVE_LIB  # noqa: E402
from utils import Block  # noqa: E402

assert _NATIVE_LIB is not None, "native library did not load (need linux)"
assert gpr_core._NAT_H is not None, "gpr_core screen is not armed"

for name in sys.argv[1:]:
    with open(
            os.path.join(ROOT, "instances", "train", f"{name}.json")) as f:
        prob = json.load(f)
    assign, _stats = construct(prob, verbose=False)

    by_bay = defaultdict(list)
    for bid, a in assign.items():
        by_bay[a["bay_id"]].append(bid)
    placed = {bid: Block(block_id=bid, block_data=prob["blocks"][bid],
                         x=int(a["x"]), y=int(a["y"]),
                         orient_idx=a["orient_idx"])
              for bid, a in assign.items()}

    pairs = [(i, j) for ids in by_bay.values()
             for p, i in enumerate(sorted(ids))
             for j in sorted(ids)[p + 1:]]

    gpr_core._NAT_CNT[:] = 0
    gpr_core._REL_MEMO.clear()      # construct() warmed it; recompute fresh
    t0 = time.time()
    nat = [gpr_core.pair_relation(placed[i], placed[j]) for i, j in pairs]
    t_nat = time.time() - t0

    h_save = gpr_core._NAT_H
    gpr_core._NAT_H = None
    gpr_core._REL_MEMO.clear()      # force the pure-GEOS recompute
    t0 = time.time()
    ref = [gpr_core.pair_relation(placed[i], placed[j]) for i, j in pairs]
    t_ref = time.time() - t0
    gpr_core._NAT_H = h_save

    bad = [(pairs[k], nat[k], ref[k]) for k in range(len(pairs))
           if nat[k] != ref[k]]
    exact, memo = int(gpr_core._NAT_CNT[0]), int(gpr_core._NAT_CNT[1])
    print(f"{name}: pairs={len(pairs)} mismatches={len(bad)} "
          f"nat={t_nat:.2f}s ref={t_ref:.2f}s "
          f"screen exact={exact} memo={memo}", flush=True)
    if bad:
        for pr, a, b in bad[:5]:
            print(f"  MISMATCH {pr}: nat={a} ref={b}")
        sys.exit(1)
print("PAIRREL PARITY OK")
