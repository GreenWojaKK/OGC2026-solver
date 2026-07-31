"""
test_predicates.py -- cross-check geoexact.so against shapely/GEOS.

Every verdict the engine can produce is compared with
shapely.intersects(translate(A), translate(B)) on the SAME float64 inputs.
Two suites:

  1. synthetic adversarial: shared edges, vertex touches, collinear
     overlaps, hole containment, multipolygon parts, decimal knife-edges.
  2. real data: block-orientation unions from train instances at random
     integer offsets PLUS the AABB-contact offsets (ceil/floor of bbox
     differences) that Phase A actually enumerates -- the decision
     boundary the constructor explores.

Run inside WSL/linux:  python3 test_predicates.py [n_random_per_inst]
Exit 0 iff zero mismatches.
"""
import ctypes
from ctypes import POINTER, c_double, c_int32, c_int64, c_void_p
import json
import math
import os
import random
import sys

import numpy as np
import shapely
from shapely.affinity import translate as shp_translate
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

lib = ctypes.CDLL(os.path.join(HERE, "geoexact.so"))
lib.geo_new.restype = c_void_p
lib.geo_new.argtypes = []
lib.geo_del.argtypes = [c_void_p]
lib.geo_add.restype = c_int32
lib.geo_add.argtypes = [c_void_p, POINTER(c_double), c_int32,
                        POINTER(c_int32), c_int32, POINTER(c_int32), c_int32]
lib.geo_pair.restype = c_int32
lib.geo_pair.argtypes = [c_void_p, c_int32, c_int64, c_int64,
                         c_int32, c_int64, c_int64, POINTER(c_int64)]
lib.geo_isect_raw.restype = c_int32
lib.geo_isect_raw.argtypes = [c_void_p, c_int32, c_double, c_double,
                              c_int32, c_double, c_double]

H = lib.geo_new()


def register(union):
    parts = ([union] if union.geom_type == "Polygon"
             else list(shapely.get_parts(union)))
    coords, ring_ptr, test_idx = [], [0], []
    for p in parts:
        test_idx.append(ring_ptr[-1])
        for ring in [p.exterior, *p.interiors]:
            c = shapely.get_coordinates(ring)
            coords.append(c)
            ring_ptr.append(ring_ptr[-1] + len(c))
    xy = np.ascontiguousarray(np.concatenate(coords), dtype=np.float64)
    rp = np.array(ring_ptr, dtype=np.int32)
    tv = np.array(test_idx, dtype=np.int32)
    return lib.geo_add(H, xy.ctypes.data_as(POINTER(c_double)), len(xy),
                       rp.ctypes.data_as(POINTER(c_int32)), len(rp) - 1,
                       tv.ctypes.data_as(POINTER(c_int32)), len(tv))


mismatches = []
n_checked = 0


def check(gid_a, ua, ax, ay, gid_b, ub, bx, by, tag):
    global n_checked
    n_checked += 1
    mine = lib.geo_isect_raw(H, gid_a, float(ax), float(ay),
                             gid_b, float(bx), float(by))
    ref = int(shapely.intersects(shp_translate(ua, float(ax), float(ay)),
                                 shp_translate(ub, float(bx), float(by))))
    if mine != ref:
        mismatches.append((tag, gid_a, ax, ay, gid_b, bx, by, mine, ref))
        if len(mismatches) <= 10:
            print(f"MISMATCH [{tag}] mine={mine} geos={ref} "
                  f"A@({ax},{ay}) B@({bx},{by})")
            print("  A:", shp_translate(ua, float(ax), float(ay)).wkt[:160])
            print("  B:", shp_translate(ub, float(bx), float(by)).wkt[:160])


# --------------------------------------------------------------- synthetic
def synthetic():
    sq = box(0, 0, 1, 1)
    big = box(0, 0, 10, 10)
    holed = Polygon(box(0, 0, 10, 10).exterior.coords,
                    [list(box(3, 3, 7, 7).exterior.coords)])
    tri = Polygon([(0, 0), (4, 0), (0, 4)])
    dec = Polygon([(0.3357, 0.0), (3.9166, 0.1123), (4.8284, 2.77),
                   (1.5, 3.9999), (0.3357, 0.0)])
    mp = unary_union([box(0, 0, 2, 2), box(5, 0, 7, 2)])
    inner = box(4, 4, 6, 6)          # sits inside holed's hole
    hole_fill = box(3, 3, 7, 7)      # exactly fills the hole
    cases = [sq, big, holed, tri, dec, mp, inner, hole_fill]
    gids = [register(u) for u in cases]

    # pairwise at structured offsets: shared edges, vertex touch, containment
    offs = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (10, 0), (-1, -1),
            (3, 3), (4, 4), (5, 5), (7, 7), (-10, 3), (6, 2), (2, 6)]
    for i, ui in enumerate(cases):
        for j, uj in enumerate(cases):
            for ox, oy in offs:
                check(gids[i], ui, 0, 0, gids[j], uj, ox, oy,
                      f"syn{i}-{j}")

    # collinear edge overlaps at decimal coordinates
    a = Polygon([(0.1234, 0), (5.5678, 0), (5.5678, 2), (0.1234, 2)])
    b = Polygon([(2.9, -3), (8.25, -3), (8.25, 0), (2.9, 0)])   # touches y=0
    ga, gb = register(a), register(b)
    for dx in range(-12, 13):
        check(ga, a, 0, 0, gb, b, dx, 0, "collinear")
        check(ga, a, 0, 0, gb, b, dx, 1, "overlap1")
        check(ga, a, 7, 3, gb, b, dx, 3, "shifted")

    # random small polygons, dense offset sweep (touch-rich)
    rng = random.Random(42)
    for t in range(60):
        pts = [(rng.randint(0, 6) + rng.choice([0.0, 0.5, 0.25, 0.3357]),
                rng.randint(0, 6) + rng.choice([0.0, 0.5, 0.75, 0.1123]))
               for _ in range(8)]
        p = Polygon(pts).buffer(0)
        if p.is_empty or p.geom_type not in ("Polygon", "MultiPolygon"):
            continue
        gp = register(p)
        for dx in range(-8, 9, 2):
            for dy in range(-8, 9, 2):
                check(gp, p, 0, 0, ga, a, dx, dy, f"rand{t}")


# --------------------------------------------------------------- real data
def _poly_from_verts(verts):
    if len(verts) < 3:
        return None
    p = Polygon(verts)
    if not p.is_valid or p.area <= 0:
        p = p.buffer(0)
        if p.is_empty:
            return None
    return p


def real_instance(name, n_random):
    with open(
            os.path.join(ROOT, "instances", "train", f"{name}.json")) as f:
        prob = json.load(f)
    blocks = prob["blocks"]
    if isinstance(blocks, dict):
        blocks = list(blocks.values())
    unions = []
    for b in blocks:
        for oi, s in enumerate(b["shape"]):
            layers = s["layers"]
            polys = [q for q in (_poly_from_verts(l) for l in layers)
                     if q is not None]
            if not polys:
                continue
            u = unary_union(polys)
            if u.is_empty or u.geom_type not in ("Polygon", "MultiPolygon"):
                continue
            unions.append(u)
    rng = random.Random(1234)
    sample = rng.sample(unions, min(120, len(unions)))
    gids = [register(u) for u in sample]
    print(f"  {name}: {len(unions)} unions, testing {len(sample)}")

    # AABB-contact offsets: the ceil/floor candidates Phase A enumerates
    n_contact = 0
    for _ in range(n_random // 2):
        i, j = rng.randrange(len(sample)), rng.randrange(len(sample))
        ua, ub = sample[i], sample[j]
        ax0, ay0, ax1, ay1 = ua.bounds
        bx0, by0, bx1, by1 = ub.bounds
        for dx in {math.ceil(ax1 - bx0), math.floor(ax0 - bx1),
                   math.ceil(ax0 - bx0)}:
            dy = rng.randint(math.floor(ay0 - by1) - 1,
                             math.ceil(ay1 - by0) + 1)
            check(gids[i], ua, 0, 0, gids[j], ub, dx, dy,
                  f"{name}-contact")
            n_contact += 1

    # random near-overlap offsets
    for _ in range(n_random):
        i, j = rng.randrange(len(sample)), rng.randrange(len(sample))
        ua, ub = sample[i], sample[j]
        ax0, ay0, ax1, ay1 = ua.bounds
        bx0, by0, bx1, by1 = ub.bounds
        dx = rng.randint(math.floor(ax0 - bx1) - 2, math.ceil(ax1 - bx0) + 2)
        dy = rng.randint(math.floor(ay0 - by1) - 2, math.ceil(ay1 - by0) + 2)
        ax, ay = rng.randint(0, 50), rng.randint(0, 20)
        check(gids[i], ua, ax, ay, gids[j], ub, ax + dx, ay + dy,
              f"{name}-rand")
    print(f"    contact cases: {n_contact}")


# ------------------------------------------------------------- memo parity
def memo_check():
    """geo_pair (memoised) must agree with geo_isect_raw and be stable."""
    counters = (c_int64 * 2)(0, 0)
    sq = box(0, 0, 3, 3)
    g1, g2 = register(sq), register(box(0, 0, 2, 5))
    rng = random.Random(7)
    for _ in range(2000):
        ax, ay = rng.randint(0, 30), rng.randint(0, 30)
        bx, by = rng.randint(0, 30), rng.randint(0, 30)
        v1 = lib.geo_pair(H, g1, ax, ay, g2, bx, by, counters)
        v2 = lib.geo_isect_raw(H, g1, float(ax), float(ay),
                               g2, float(bx), float(by))
        v3 = lib.geo_pair(H, g1, ax, ay, g2, bx, by, counters)  # memo hit
        assert v1 == v2 == v3, (ax, ay, bx, by, v1, v2, v3)
    print(f"  memo: confirms={counters[0]} hits={counters[1]}")
    assert counters[1] >= counters[0]  # repeats + offset collisions hit


def main():
    n_random = int(sys.argv[1]) if len(sys.argv) > 1 else 30000
    print("synthetic suite ...")
    synthetic()
    print(f"  checked so far: {n_checked}")
    print("memo suite ...")
    memo_check()
    print("real-data suite ...")
    for name in ("prob_1", "prob_21", "prob_38"):
        real_instance(name, n_random)
    print(f"TOTAL checked: {n_checked}, mismatches: {len(mismatches)}")
    if mismatches:
        sys.exit(1)
    print("OK -- native verdicts identical to GEOS")


if __name__ == "__main__":
    main()
