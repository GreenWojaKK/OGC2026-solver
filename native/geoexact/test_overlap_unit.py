"""test_overlap_unit.py -- hand-built semantics cases for geo_pair_ov
(exact interior overlap, touch-exclusive).  Run on WSL after build.sh.
"""
import ctypes
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SO = os.path.join(HERE, "geoexact.so")
lib = ctypes.CDLL(SO)
PD = ctypes.POINTER(ctypes.c_double)
PI32 = ctypes.POINTER(ctypes.c_int32)
PI64 = ctypes.POINTER(ctypes.c_int64)
lib.geo_new.restype = ctypes.c_void_p
lib.geo_add.restype = ctypes.c_int32
lib.geo_add.argtypes = [ctypes.c_void_p, PD, ctypes.c_int32, PI32,
                        ctypes.c_int32, PI32, ctypes.c_int32]
lib.geo_pair_ov.restype = ctypes.c_int32
lib.geo_pair_ov.argtypes = [ctypes.c_void_p, ctypes.c_int32,
                            ctypes.c_int64, ctypes.c_int64, ctypes.c_int32,
                            ctypes.c_int64, ctypes.c_int64, PI64]

h = lib.geo_new()
cnt = np.zeros(2, dtype=np.int64)
cnt_p = cnt.ctypes.data_as(PI64)


def add(verts):
    xy = np.ascontiguousarray(np.asarray(verts, dtype=np.float64))
    if not np.array_equal(xy[0], xy[-1]):
        xy = np.vstack((xy, xy[:1]))
    rp = np.array([0, len(xy)], dtype=np.int32)
    tv = np.array([0], dtype=np.int32)
    return lib.geo_add(h, xy.ctypes.data_as(PD), len(xy),
                       rp.ctypes.data_as(PI32), 1,
                       tv.ctypes.data_as(PI32), 1)


def ov(ga, ax, ay, gb, bx, by):
    return lib.geo_pair_ov(h, ga, ax, ay, gb, bx, by, cnt_p)


sq = add([[0, 0], [1, 0], [1, 1], [0, 1]])                # unit square
sq_cw = add([[0, 0], [0, 1], [1, 1], [1, 0]])             # same, CW ring
tri_chord = add([[0, 0], [4, 4], [0, 4]])                 # chord diagonal
big = add([[0, 0], [4, 0], [4, 4], [0, 4]])               # 4x4 square
small = add([[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]])
tri_bot = add([[0, 0], [4, 0], [0, 4]])                   # shares bottom edge

CASES = [
    ("edge-share touch",      ov(sq, 0, 0, sq, 1, 0), 0),
    ("vertex touch",          ov(sq, 0, 0, sq, 1, 1), 0),
    ("disjoint",              ov(sq, 0, 0, sq, 3, 0), 0),
    ("half overlap",          ov(sq, 0, 0, sq, 0, 0) and 1, 1),
    ("identical",             ov(sq, 0, 0, sq, 0, 0), 1),
    ("identical cw-vs-ccw",   ov(sq, 0, 0, sq_cw, 0, 0), 1),
    ("proper overlap",        ov(big, 0, 0, big, 2, 2), 1),
    ("containment",           ov(big, 0, 0, small, 0, 0), 1),
    ("chord through interior", ov(big, 0, 0, tri_chord, 0, 0), 1),
    ("shared-edge triangle",  ov(big, 0, 0, tri_bot, 0, 0), 1),
    ("stacked touch",         ov(sq, 0, 0, sq, 0, 1), 0),
    ("offset far",            ov(sq, 0, 0, big, 10, 10), 0),
    ("contain w/ shared edge", ov(big, 0, 0, sq, 0, 0), 1),
    ("corner-inside overlap", ov(big, 0, 0, big, 3, 3), 1),
]
bad = 0
for name, got, want in CASES:
    ok = int(bool(got)) == want
    bad += not ok
    print(f"{'OK ' if ok else 'BAD'} {name}: got={int(bool(got))} want={want}")
print("UNIT", "FAIL" if bad else "OK", flush=True)
sys.exit(1 if bad else 0)
