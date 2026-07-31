"""test_oob_guard.py -- v5.9.5 hardening regression: out-of-range gids
must return the conservative verdict (1 / blocked) instead of indexing
h->geoms out of bounds (the 2026-07-20 rollover-race crash class).

Usage: python3 test_oob_guard.py   (loads ./geoexact.so)
"""
import ctypes
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PD = ctypes.POINTER(ctypes.c_double)
PI32 = ctypes.POINTER(ctypes.c_int32)
PI64 = ctypes.POINTER(ctypes.c_int64)
PU8 = ctypes.POINTER(ctypes.c_uint8)

C = ctypes.CDLL(os.path.join(HERE, "geoexact.so"))
C.geo_new.restype = ctypes.c_void_p
C.geo_add.restype = ctypes.c_int32
C.geo_add.argtypes = [ctypes.c_void_p, PD, ctypes.c_int32, PI32,
                      ctypes.c_int32, PI32, ctypes.c_int32]
C.geo_pair.restype = ctypes.c_int32
C.geo_pair.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_int64,
                       ctypes.c_int64, ctypes.c_int32, ctypes.c_int64,
                       ctypes.c_int64, PI64]
C.geo_pair_ov.restype = ctypes.c_int32
C.geo_pair_ov.argtypes = list(C.geo_pair.argtypes)
C.geo_isect_raw.restype = ctypes.c_int32
C.geo_isect_raw.argtypes = [ctypes.c_void_p, ctypes.c_int32,
                            ctypes.c_double, ctypes.c_double,
                            ctypes.c_int32, ctypes.c_double,
                            ctypes.c_double]
C.geo_resolve.restype = None
C.geo_resolve.argtypes = [
    ctypes.c_void_p, ctypes.c_int32, PI64, ctypes.c_int32, PI64,
    ctypes.c_int32, PI32, PI64, PI64, PI32, PI32, PU8, PI64]

h = C.geo_new()
sq = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0],
               [0.0, 0.0]])
rp = np.array([0, 5], dtype=np.int32)
tv = np.array([0], dtype=np.int32)
g0 = C.geo_add(h, np.ascontiguousarray(sq).ctypes.data_as(PD), 5,
               rp.ctypes.data_as(PI32), 1, tv.ctypes.data_as(PI32), 1)
cnt = np.zeros(2, dtype=np.int64)
cp = cnt.ctypes.data_as(PI64)

fails = 0
for bad in (-1, 1, 9999, 16383, 20000):
    for name, call in (
            ("geo_pair", lambda b: C.geo_pair(h, g0, 0, 0, b, 1, 1, cp)),
            ("geo_pair(a)", lambda b: C.geo_pair(h, b, 0, 0, g0, 1, 1,
                                                 cp)),
            ("geo_pair_ov", lambda b: C.geo_pair_ov(h, g0, 0, 0, b, 1, 1,
                                                    cp)),
            ("geo_isect_raw", lambda b: C.geo_isect_raw(h, b, 0.0, 0.0,
                                                        g0, 1.0, 1.0))):
        v = call(bad)
        if v != 1:
            print(f"FAIL {name} gid={bad}: got {v}, want 1")
            fails += 1

xs = np.array([0], dtype=np.int64)
ys = np.array([0], dtype=np.int64)
act_gid = np.array([g0], dtype=np.int32)
act_x = np.array([0], dtype=np.int64)
act_y = np.array([0], dtype=np.int64)
res_ptr = np.array([0, 1], dtype=np.int32)
res_flat = np.array([0], dtype=np.int32)
res_ok = np.zeros(1, dtype=np.uint8)
C.geo_resolve(h, 9999, xs.ctypes.data_as(PI64), 1,
              ys.ctypes.data_as(PI64), 1, act_gid.ctypes.data_as(PI32),
              act_x.ctypes.data_as(PI64), act_y.ctypes.data_as(PI64),
              res_ptr.ctypes.data_as(PI32), res_flat.ctypes.data_as(PI32),
              res_ok.ctypes.data_as(PU8), cp)
if res_ok[0] != 1:
    print(f"FAIL geo_resolve oob cand: res_ok={res_ok[0]}, want 1")
    fails += 1
bad_act = np.array([12345], dtype=np.int32)
res_ok[0] = 0
C.geo_resolve(h, g0, xs.ctypes.data_as(PI64), 1,
              ys.ctypes.data_as(PI64), 1, bad_act.ctypes.data_as(PI32),
              act_x.ctypes.data_as(PI64), act_y.ctypes.data_as(PI64),
              res_ptr.ctypes.data_as(PI32), res_flat.ctypes.data_as(PI32),
              res_ok.ctypes.data_as(PU8), cp)
if res_ok[0] != 1:
    print(f"FAIL geo_resolve oob act: res_ok={res_ok[0]}, want 1")
    fails += 1

# sanity: valid gids still work
g1 = C.geo_add(h, np.ascontiguousarray(sq + 10.0).ctypes.data_as(PD), 5,
               rp.ctypes.data_as(PI32), 1, tv.ctypes.data_as(PI32), 1)
if C.geo_pair(h, g0, 0, 0, g1, 0, 0, cp) != 0:
    print("FAIL disjoint valid pair: want 0")
    fails += 1
if C.geo_pair_ov(h, g0, 0, 0, g1, -10, -10, cp) != 1:
    print("FAIL overlapping valid pair: want 1")
    fails += 1

print("OOB GUARD OK" if fails == 0 else f"{fails} FAILURES")
raise SystemExit(1 if fails else 0)
