from __future__ import annotations
import ctypes
import math
import os
import random
import sys
import time
import numpy as np
import shapely
from shapely.affinity import translate as shp_translate
from shapely.ops import unary_union
from shapely.prepared import prep
from utils import Bay, Block, _poly_from_verts, _bounding_box, _resolve_layers
import gpr_core
from gpr_core import pair_relation, FREE, EXCLUSIVE
_PD = ctypes.POINTER(ctypes.c_double)
_PI32 = ctypes.POINTER(ctypes.c_int32)
_PI64 = ctypes.POINTER(ctypes.c_int64)
_PU8 = ctypes.POINTER(ctypes.c_uint8)
_NATIVE_LIB = None
if sys.platform == 'linux' and True:
    _so_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'geoexact.so')
    if os.path.exists(_so_path):
        _NATIVE_LIB = ctypes.CDLL(_so_path)
        _NATIVE_LIB.geo_new.restype = ctypes.c_void_p
        _NATIVE_LIB.geo_new.argtypes = []
        _NATIVE_LIB.geo_add.restype = ctypes.c_int32
        _NATIVE_LIB.geo_add.argtypes = [ctypes.c_void_p, _PD, ctypes.c_int32, _PI32, ctypes.c_int32, _PI32, ctypes.c_int32]
        _NATIVE_LIB.geo_pair.restype = ctypes.c_int32
        _NATIVE_LIB.geo_pair.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_int64, ctypes.c_int64, ctypes.c_int32, ctypes.c_int64, ctypes.c_int64, _PI64]
        _NATIVE_LIB.geo_resolve.restype = None
        _NATIVE_LIB.geo_resolve.argtypes = [ctypes.c_void_p, ctypes.c_int32, _PI64, ctypes.c_int32, _PI64, ctypes.c_int32, _PI32, _PI64, _PI64, _PI32, _PI32, _PU8, _PI64]
        gpr_core._set_native(_NATIVE_LIB)
_RUST_LIB = None
if sys.platform == 'linux' and True:
    for _rc_path in (os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rustcore.so'), os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'dev_rust', 'rustcore.so')):
        if os.path.exists(_rc_path):
            _RUST_LIB = ctypes.CDLL(_rc_path)
            _RUST_LIB.rc_eng_new.restype = ctypes.c_void_p
            _RUST_LIB.rc_eng_new.argtypes = [ctypes.c_int64, _PI64, _PI64, ctypes.c_int32]
            _RUST_LIB.rc_eng_add_geo.restype = ctypes.c_int32
            _RUST_LIB.rc_eng_add_geo.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, _PD, ctypes.c_int32, _PI32, ctypes.c_int32, _PI32, ctypes.c_int32, _PD]
            _RUST_LIB.rc_eng_add_layer.restype = None
            _RUST_LIB.rc_eng_add_layer.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, _PD, ctypes.c_int32]
            _RUST_LIB.rc_eng_run.restype = ctypes.c_int32
            _RUST_LIB.rc_eng_run.argtypes = [ctypes.c_void_p, ctypes.c_int32, _PI64, _PI64, _PI64, _PD, _PD, ctypes.c_int32, ctypes.c_double, _PD, _PI32, _PI32, _PI32, _PI32, ctypes.c_int32, ctypes.c_int32, _PI64, ctypes.c_int32, ctypes.c_int32, ctypes.c_double, _PI64, _PI32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, _PI64]
            break
_HAVE_NUMBA = False
try:
    from numba import njit as _njit, prange as _prange
    _HAVE_NUMBA = True
except Exception:
    _HAVE_NUMBA = False

def _rasterize(union, bbox) -> tuple[np.ndarray, np.ndarray, int, int]:
    lx0, ly0, lx1, ly1 = bbox
    ox = math.floor(lx0 * 2)
    oy = math.floor(ly0 * 2)
    w = max(1, math.ceil(lx1 * 2) - ox)
    h = max(1, math.ceil(ly1 * 2) - oy)
    cs = 1.0 / 2
    xs = (np.arange(w) + ox) * cs
    ys = (np.arange(h) + oy) * cs
    X, Y = np.meshgrid(xs, ys)
    cells = shapely.box(X.ravel(), Y.ravel(), X.ravel() + cs, Y.ravel() + cs)
    mask = shapely.intersects(union, cells).reshape(h, w)
    amask = shapely.contains_properly(union, cells).reshape(h, w)
    return (mask, amask, ox, oy)

class _Geo:
    __slots__ = ('union', 'bbox', 'mask', 'amask', 'ox', 'oy', 'cells', 'tmpl', 'gid')

    def __init__(self, blk_data: dict, oi: int):
        layers = blk_data['shape'][oi]['layers']
        polys = [p for p in (_poly_from_verts(l) for l in layers) if p is not None]
        self.union = unary_union(polys) if polys else None
        self.bbox = _bounding_box([v for l in layers for v in l])
        if self.union is not None:
            self.mask, self.amask, self.ox, self.oy = _rasterize(self.union, self.bbox)
        else:
            z = np.zeros((1, 1), dtype=bool)
            self.mask, self.amask, self.ox, self.oy = (z, z, 0, 0)
        self.cells = None
        self.tmpl = _bulk_template(self.union)
        self.gid = None

def _bulk_template(union):
    if union is None:
        return None
    if union.geom_type == 'Polygon':
        parts = [union]
    elif union.geom_type == 'MultiPolygon':
        parts = list(shapely.get_parts(union))
    else:
        return None
    coords, ridx = ([], [])
    for k, p in enumerate(parts):
        if shapely.get_num_interior_rings(p) > 0:
            return None
        c = shapely.get_coordinates(p.exterior)
        coords.append(c)
        ridx.append(np.full(c.shape[0], k, dtype=np.int64))
    return (np.ascontiguousarray(np.concatenate(coords)), np.concatenate(ridx), len(parts))

def _bulk_translate(tmpl, offs):
    coords, ridx, M = tmpl
    K = offs.shape[0]
    flat = (coords[None, :, :] + offs[:, None, :]).reshape(-1, 2)
    fridx = (np.arange(K, dtype=np.int64)[:, None] * M + ridx[None, :]).reshape(-1)
    return shapely.polygons(shapely.linearrings(flat, indices=fridx))

class _Placed:
    __slots__ = ('bid', 'x', 'y', 'oi', 'S', 'F', 'block', 'union', 'prep', 'bbox', 'mask', 'amask', 'geo')

    def __init__(self, bid, x, y, oi, S, F, block, union, geo, bay_w, bay_h):
        self.bid = bid
        self.x, self.y, self.oi = (x, y, oi)
        self.S, self.F = (S, F)
        self.block = block
        self.union = union
        self.geo = geo
        shapely.prepare(union)
        self.prep = prep(union)
        self.bbox = block.bounding_rect()
        m = np.zeros((bay_h * 2, bay_w * 2), dtype=bool)
        a = np.zeros_like(m)
        hh, ww = geo.mask.shape
        r0, c0 = (y * 2 + geo.oy, x * 2 + geo.ox)
        m[r0:r0 + hh, c0:c0 + ww] = geo.mask
        a[r0:r0 + hh, c0:c0 + ww] = geo.amask
        self.mask = m
        self.amask = a

def _overlap_exact(e: '_Placed', g: '_Geo', bid: int, oi: int, x: int, y: int, memo: dict, bu_cell: list, stats: dict, native: '_Native | None'=None) -> bool:
    r0, c0 = (y * 2 + g.oy, x * 2 + g.ox)
    if r0 >= 0 and c0 >= 0:
        hh, ww = g.mask.shape
        sub = e.mask[r0:r0 + hh, c0:c0 + ww]
        if sub.shape == g.mask.shape:
            if not (sub & g.mask).any():
                return False
            if (e.amask[r0:r0 + hh, c0:c0 + ww] & g.amask).any():
                stats['n_agg_confirm'] += 1
                return True
    if native is not None:
        return native.pair(e, g, bid, oi, x, y, stats)
    key = (e.bid, e.oi, bid, oi, e.x, e.y, x, y)
    v = memo.get(key)
    if v is None:
        stats['n_prep_confirm'] += 1
        if bu_cell[0] is None:
            bu_cell[0] = shp_translate(g.union, x, y)
        v = bool(e.prep.intersects(bu_cell[0]))
        if len(memo) >= 262144:
            memo.clear()
        memo[key] = v
    else:
        stats['n_memo_hit'] += 1
    return v

def _resolve_geos(g: '_Geo', bid: int, oi: int, actives: list, xs: list, ys: list, res_ptr, res_flat, pair_memo: dict, stats: dict):
    res_ok = np.zeros(res_flat.shape[0], dtype=bool)
    nxl = len(xs)
    pend: dict = {}
    pkey: list = []
    pact: list = []
    ppos: list = []
    pos_id: dict = {}
    offs: list = []
    for p_ in np.nonzero(np.diff(res_ptr) > 0)[0].tolist():
        y_ = ys[p_ // nxl]
        x_ = xs[p_ % nxl]
        for ti in range(res_ptr[p_], res_ptr[p_ + 1]):
            a = int(res_flat[ti])
            e = actives[a]
            mkey = (e.bid, e.oi, bid, oi, e.x, e.y, x_, y_)
            v = pair_memo.get(mkey)
            if v is None:
                w = pend.get(mkey)
                if w is None:
                    stats['n_prep_confirm'] += 1
                    pend[mkey] = [ti]
                    sl = pos_id.get(p_)
                    if sl is None:
                        sl = pos_id[p_] = len(offs)
                        offs.append((x_, y_))
                    pkey.append(mkey)
                    pact.append(e)
                    ppos.append(sl)
                else:
                    stats['n_memo_hit'] += 1
                    w.append(ti)
            else:
                stats['n_memo_hit'] += 1
                if v:
                    res_ok[ti] = True
    if pkey:
        if g.tmpl is not None:
            M = g.tmpl[2]
            polys = _bulk_translate(g.tmpl, np.array(offs, dtype=float))
            pp = np.array(ppos, dtype=np.int64)
            lefts = np.repeat(np.array([e.union for e in pact], dtype=object), M)
            rights = polys[(pp[:, None] * M + np.arange(M)).ravel()]
            vs = shapely.intersects(lefts, rights).reshape(-1, M).any(1)
        else:
            bus = [None] * len(offs)
            vs = np.empty(len(pkey), dtype=bool)
            for si in range(len(pkey)):
                sl = ppos[si]
                bu = bus[sl]
                if bu is None:
                    bu = bus[sl] = shp_translate(g.union, *offs[sl])
                vs[si] = bool(pact[si].prep.intersects(bu))
        for si, mkey in enumerate(pkey):
            v = bool(vs[si])
            if len(pair_memo) >= 262144:
                pair_memo.clear()
            pair_memo[mkey] = v
            if v:
                for ti in pend[mkey]:
                    res_ok[ti] = True
    return res_ok

class _Native:
    __slots__ = ('h', 'cnt', 'cnt_p', 'verify', 'shadow')

    def __init__(self):
        self.h = _NATIVE_LIB.geo_new()
        self.cnt = np.zeros(2, dtype=np.int64)
        self.cnt_p = self.cnt.ctypes.data_as(_PI64)
        self.verify = False
        self.shadow = {} if self.verify else None

    def gid_of(self, g: '_Geo') -> int:
        gid = g.gid
        if gid is None:
            u = g.union
            parts = [u] if u.geom_type == 'Polygon' else list(shapely.get_parts(u))
            coords, ring_ptr, test_idx = ([], [0], [])
            for p in parts:
                test_idx.append(ring_ptr[-1])
                for ring in [p.exterior, *p.interiors]:
                    c = shapely.get_coordinates(ring)
                    coords.append(c)
                    ring_ptr.append(ring_ptr[-1] + len(c))
            xy = np.ascontiguousarray(np.concatenate(coords))
            rp = np.array(ring_ptr, dtype=np.int32)
            tv = np.array(test_idx, dtype=np.int32)
            gid = g.gid = _NATIVE_LIB.geo_add(self.h, xy.ctypes.data_as(_PD), len(xy), rp.ctypes.data_as(_PI32), len(rp) - 1, tv.ctypes.data_as(_PI32), len(tv))
        return gid

    def resolve(self, g: '_Geo', bid: int, oi: int, actives: list, xs: list, ys: list, act_gid, act_gx, act_gy, xs_a, ys_a, res_ptr, res_flat, stats: dict):
        res_ok8 = np.zeros(res_flat.shape[0], dtype=np.uint8)
        c = self.cnt
        c[0] = 0
        c[1] = 0
        _NATIVE_LIB.geo_resolve(self.h, self.gid_of(g), xs_a.ctypes.data_as(_PI64), len(xs_a), ys_a.ctypes.data_as(_PI64), len(ys_a), act_gid.ctypes.data_as(_PI32), act_gx.ctypes.data_as(_PI64), act_gy.ctypes.data_as(_PI64), res_ptr.ctypes.data_as(_PI32), res_flat.ctypes.data_as(_PI32), res_ok8.ctypes.data_as(_PU8), self.cnt_p)
        stats['n_prep_confirm'] += int(c[0])
        stats['n_memo_hit'] += int(c[1])
        res_ok = res_ok8.view(np.bool_)
        if self.verify:
            ref = _resolve_geos(g, bid, oi, actives, xs, ys, res_ptr, res_flat, self.shadow, {'n_prep_confirm': 0, 'n_memo_hit': 0})
            if not np.array_equal(res_ok, ref):
                bad = np.nonzero(res_ok != ref)[0]
                raise AssertionError(f'native/GEOS resolve mismatch: {len(bad)} of {res_ok.shape[0]} entries, first ti={int(bad[0])}')
        return res_ok

    def pair(self, e: '_Placed', g: '_Geo', bid: int, oi: int, x: int, y: int, stats: dict) -> bool:
        c = self.cnt
        c[0] = 0
        c[1] = 0
        v = bool(_NATIVE_LIB.geo_pair(self.h, self.gid_of(e.geo), e.x, e.y, self.gid_of(g), x, y, self.cnt_p))
        stats['n_prep_confirm'] += int(c[0])
        stats['n_memo_hit'] += int(c[1])
        if self.verify:
            ref = bool(e.prep.intersects(shp_translate(g.union, x, y)))
            if v != ref:
                raise AssertionError(f'native/GEOS pair mismatch: {(e.bid, e.oi, bid, oi, e.x, e.y, x, y)} native={v} geos={ref}')
        return v

def _resolved_top_b(top_b: int) -> int:
    if int(top_b) != 8:
        return int(top_b)
    raw = os.environ.get('OGC_TOP_B', '')
    if not raw:
        return int(top_b)
    try:
        value = int(raw)
    except ValueError:
        return int(top_b)
    return max(1, value)

def _resolved_nest_lane(nest_lane: int) -> int:
    if int(nest_lane) <= 0:
        return int(nest_lane)
    raw = os.environ.get('OGC_NEST_LANE', '')
    if not raw:
        return int(nest_lane)
    try:
        value = int(raw)
    except ValueError:
        return int(nest_lane)
    return max(0, value)

def _earliest_gap(R_b: int, P_b: int, intervals: list) -> int:
    t = R_b
    for S_e, F_e in sorted(intervals):
        if t + P_b <= S_e:
            break
        t = max(t, F_e)
    return t
if _HAVE_NUMBA:

    @_njit(cache=True, parallel=True)
    def _k_tiers(occ, agg, bbox, xs, ys, res, gox, goy, ci, cj, ai, aj, lx0, ly0, lx1, ly1):
        A = occ.shape[0]
        ny, nx = (ys.shape[0], xs.shape[0])
        P = ny * nx
        cnt_c = np.empty(P, np.int32)
        cnt_r = np.empty(P, np.int32)
        tmp_c = np.empty((P, A), np.int32)
        tmp_r = np.empty((P, A), np.int32)
        n_mask = 0
        nc = ci.shape[0]
        na = ai.shape[0]
        for p in _prange(P):
            y = ys[p // nx]
            x = xs[p % nx]
            r0 = y * res + goy
            c0 = x * res + gox
            wy0 = y + ly0
            wy1 = y + ly1
            wx0 = x + lx0
            wx1 = x + lx1
            ncf_l = 0
            nrs_l = 0
            seen = False
            for a in range(A):
                if not (wx0 < bbox[a, 2] and bbox[a, 0] < wx1 and (wy0 < bbox[a, 3]) and (bbox[a, 1] < wy1)):
                    continue
                if not seen:
                    seen = True
                hit = False
                for t in range(nc):
                    if occ[a, r0 + ci[t], c0 + cj[t]]:
                        hit = True
                        break
                if not hit:
                    continue
                ahit = False
                for t in range(na):
                    if agg[a, r0 + ai[t], c0 + aj[t]]:
                        ahit = True
                        break
                if ahit:
                    tmp_c[p, ncf_l] = a
                    ncf_l += 1
                else:
                    tmp_r[p, nrs_l] = a
                    nrs_l += 1
            cnt_c[p] = ncf_l
            cnt_r[p] = nrs_l
            if seen:
                n_mask += 1
        conf_ptr = np.empty(P + 1, np.int32)
        res_ptr = np.empty(P + 1, np.int32)
        conf_ptr[0] = 0
        res_ptr[0] = 0
        for p in range(P):
            conf_ptr[p + 1] = conf_ptr[p] + cnt_c[p]
            res_ptr[p + 1] = res_ptr[p] + cnt_r[p]
        ncf = conf_ptr[P]
        nrs = res_ptr[P]
        conf_flat = np.empty(ncf, np.int32)
        res_flat = np.empty(nrs, np.int32)
        for p in range(P):
            b = conf_ptr[p]
            for k in range(cnt_c[p]):
                conf_flat[b + k] = tmp_c[p, k]
            b = res_ptr[p]
            for k in range(cnt_r[p]):
                res_flat[b + k] = tmp_r[p, k]
        return (conf_ptr, conf_flat, res_ptr, res_flat, n_mask, ncf)

    @_njit(cache=True, parallel=True)
    def _k_score(conf_ptr, conf_flat, res_ptr, res_flat, res_ok, act_S, act_F, xs, ys, R_b, P_b, D_b, base_score, w1, ly1, nest, tgt, ptw):
        ny, nx = (ys.shape[0], xs.shape[0])
        P = ny * nx
        scores = np.empty(P, np.float64)
        scores_opt = np.empty(P, np.float64)
        for p in _prange(P):
            y = ys[p // nx]
            c0, c1 = (conf_ptr[p], conf_ptr[p + 1])
            r0_, r1_ = (res_ptr[p], res_ptr[p + 1])
            need = c1 - c0 + (r1_ - r0_)
            keys = np.empty(need, np.int64)
            hard = np.empty(need, np.int64)
            n_iv = 0
            for i in range(c0, c1):
                a = conf_flat[i]
                keys[n_iv] = act_S[a] << 20 | act_F[a]
                n_iv += 1
            for i in range(r0_, r1_):
                if res_ok[i]:
                    a = res_flat[i]
                    keys[n_iv] = act_S[a] << 20 | act_F[a]
                    n_iv += 1
            for i in range(1, n_iv):
                k = keys[i]
                j = i - 1
                while j >= 0 and keys[j] > k:
                    keys[j + 1] = keys[j]
                    j -= 1
                keys[j + 1] = k
            t = R_b
            for i in range(n_iv):
                S_e = keys[i] >> 20
                F_e = keys[i] & 1048575
                if t + P_b <= S_e:
                    break
                if F_e > t:
                    t = F_e
            entry = t
            tard = entry + P_b - D_b
            if tard < 0:
                tard = 0
            dtg = entry - tgt
            if dtg < 0:
                dtg = -dtg
            sc = base_score + w1 * tard + 0.0001 * (y + ly1) + 1e-06 * entry + ptw * dtg
            sco = sc
            if nest and n_iv > 0:
                nh = 0
                RPb = R_b + P_b
                for i in range(n_iv):
                    S_e = keys[i] >> 20
                    F_e = keys[i] & 1048575
                    if not (S_e <= R_b and RPb <= F_e):
                        hard[nh] = keys[i]
                        nh += 1
                if nh < n_iv:
                    t = R_b
                    for i in range(nh):
                        S_e = hard[i] >> 20
                        F_e = hard[i] & 1048575
                        if t + P_b <= S_e:
                            break
                        if F_e > t:
                            t = F_e
                    t_opt = t + P_b - D_b
                    if t_opt < 0:
                        t_opt = 0
                    dtg = t - tgt
                    if dtg < 0:
                        dtg = -dtg
                    sco = base_score + w1 * t_opt + 0.0001 * (y + ly1) + 1e-06 * t + ptw * dtg
            scores[p] = sc
            scores_opt[p] = sco
        return (scores, scores_opt)
_NUMBA_READY = None
_RV_IN = [False]

def _numba_warmup() -> bool:
    global _NUMBA_READY, _HAVE_NUMBA
    if _NUMBA_READY is not None:
        return _NUMBA_READY
    if not _HAVE_NUMBA:
        _NUMBA_READY = False
        return False
    try:
        from numba import set_num_threads, get_num_threads
        set_num_threads(min(4, get_num_threads()))
        occ = np.zeros((1, 8, 8), dtype=bool)
        occ[0, 2:5, 2:5] = True
        bbox = np.array([[1.0, 1.0, 3.0, 3.0]])
        xs = np.array([0, 1], dtype=np.int64)
        ys = np.array([0], dtype=np.int64)
        idx = np.nonzero(np.ones((2, 2), dtype=bool))
        cp, cf, rp, rf, _, _ = _k_tiers(occ, occ, bbox, xs, ys, 2, 0, 0, idx[0], idx[1], idx[0], idx[1], 0.0, 0.0, 2.0, 2.0)
        _k_score(cp, cf, rp, rf, np.zeros(len(rf), dtype=bool), np.zeros(1, np.int64), np.ones(1, np.int64), xs, ys, 0, 1, 2, 0.0, 1.0, 2.0, 1, 0, 0.0)
        _NUMBA_READY = True
    except Exception:
        _HAVE_NUMBA = False
        _NUMBA_READY = False
    return _NUMBA_READY

def _insertion_order(blocks_data: list, order_key: str, seed) -> list[int]:
    n = len(blocks_data)

    def bbox_area(i):
        vs = [v for l in blocks_data[i]['shape'][0]['layers'] for v in l]
        xs = [v[0] for v in vs]
        ys = [v[1] for v in vs]
        return (max(xs) - min(xs)) * (max(ys) - min(ys))
    area = [bbox_area(i) for i in range(n)]
    bd = blocks_data
    if order_key == 'edd':
        key = lambda i: (bd[i]['due_date'], bd[i]['release_time'], bd[i]['processing_time'])
    elif order_key == 'edd_big':
        key = lambda i: (bd[i]['due_date'], -area[i], bd[i]['processing_time'])
    elif order_key == 'edd_long':
        key = lambda i: (bd[i]['due_date'], -bd[i]['processing_time'], -area[i])
    elif order_key == 'release':
        key = lambda i: (bd[i]['release_time'], bd[i]['due_date'], -area[i])
    elif order_key == 'shuffle':
        rnd = random.Random(seed)
        noise = [rnd.uniform(-2.5, 2.5) for _ in range(n)]
        key = lambda i: (bd[i]['due_date'] + noise[i], -area[i])
    else:
        raise ValueError(f'unknown order_key {order_key!r}')
    return sorted(range(n), key=key)

def _slide_to_contact(x: int, y: int, g: '_Geo', actives: list, entry: int, bid: int, oi: int, pair_memo: dict, stats: dict, max_steps: int=80, native: '_Native | None'=None) -> tuple[int, int]:
    lx0, ly0, lx1, ly1 = g.bbox
    px_lo, py_lo = (math.ceil(-lx0), math.ceil(-ly0))
    obs = [e for e in actives if e.F > entry]

    def ok(nx, ny):
        wbb = (nx + lx0, ny + ly0, nx + lx1, ny + ly1)
        bu_cell = [None]
        for e in obs:
            if not (wbb[0] < e.bbox[2] and e.bbox[0] < wbb[2] and (wbb[1] < e.bbox[3]) and (e.bbox[1] < wbb[3])):
                continue
            if _overlap_exact(e, g, bid, oi, nx, ny, pair_memo, bu_cell, stats, native):
                return False
        return True
    steps = 0
    moved = True
    while moved and steps < max_steps:
        moved = False
        while y > py_lo and steps < max_steps and ok(x, y - 1):
            y -= 1
            steps += 1
            moved = True
        while x > px_lo and steps < max_steps and ok(x - 1, y):
            x -= 1
            steps += 1
            moved = True
    return (x, y)

def _earliest_slot(R_b: int, P_b: int, conflicts: list) -> tuple[int, int]:
    if not conflicts:
        return (R_b, R_b + P_b)

    def feasible(t: int):
        F_b = t + P_b
        for _ in range(len(conflicts) + 2):
            changed = False
            for S_e, F_e, kind in conflicts:
                if F_e <= t:
                    continue
                if F_b <= S_e:
                    continue
                if kind == 'b_inside' and S_e <= t and (F_b <= F_e):
                    continue
                if kind == 'b_enclose' and t <= S_e:
                    if F_b < F_e:
                        F_b = F_e
                        changed = True
                    continue
                return None
            if not changed:
                return F_b
        return None
    t_after = max(R_b, max((F_e for _, F_e, _ in conflicts)))
    cands = {R_b}
    for S_e, F_e, kind in conflicts:
        if F_e > R_b:
            cands.add(F_e)
        if kind == 'b_inside' and S_e > R_b:
            cands.add(S_e)
    best = (t_after, t_after + P_b)
    for t in sorted(cands):
        if t >= t_after:
            break
        F_b = feasible(t)
        if F_b is not None:
            if F_b < best[1] or (F_b == best[1] and t < best[0]):
                best = (t, F_b)
            break
    return best

def _construct_rust(prob_info, timelimit, max_xs, max_ys, top_b, order_key, seed, nest_lane, slide, geo_cache, fixed_assignments, insert_ids, insert_order, plan, plan_bays, plan_bucket, plan_target_w, bay_prior, bay_prior_strength):
    t0 = time.time()
    blocks_data = prob_info['blocks']
    st = geo_cache.get('_rust')
    if st is None:
        bays_data = prob_info['bays']
        n, m = (len(blocks_data), len(bays_data))
        w = prob_info.get('weights', {})
        bay_w = np.array([int(b['width']) for b in bays_data], dtype=np.int64)
        bay_h = np.array([int(b['height']) for b in bays_data], dtype=np.int64)
        eng = _RUST_LIB.rc_eng_new(2, bay_w.ctypes.data_as(_PI64), bay_h.ctypes.data_as(_PI64), m)
        dd = np.zeros(2, dtype=np.float64)
        di = np.zeros(2, dtype=np.int32)
        db = np.zeros(4, dtype=np.float64)
        for bid, blk in enumerate(blocks_data):
            for oi in range(len(blk['shape'])):
                layers = _resolve_layers(blk['shape'][oi]['layers'])
                polys = [p for p in (_poly_from_verts(l) for l in layers) if p is not None]
                union = unary_union(polys) if polys else None
                if union is None:
                    _RUST_LIB.rc_eng_add_geo(eng, bid, oi, 0, dd.ctypes.data_as(_PD), 0, di.ctypes.data_as(_PI32), 0, di.ctypes.data_as(_PI32), 0, db.ctypes.data_as(_PD))
                else:
                    parts = [union] if union.geom_type == 'Polygon' else list(shapely.get_parts(union))
                    coords, ring_ptr, test_idx = ([], [0], [])
                    for pt in parts:
                        test_idx.append(ring_ptr[-1])
                        for ring in [pt.exterior, *pt.interiors]:
                            c = shapely.get_coordinates(ring)
                            coords.append(c)
                            ring_ptr.append(ring_ptr[-1] + len(c))
                    xy = np.ascontiguousarray(np.concatenate(coords))
                    rp = np.array(ring_ptr, dtype=np.int32)
                    tv = np.array(test_idx, dtype=np.int32)
                    bb = np.ascontiguousarray(np.array(_bounding_box([v for l in layers for v in l]), dtype=np.float64))
                    _RUST_LIB.rc_eng_add_geo(eng, bid, oi, 1, xy.ctypes.data_as(_PD), len(xy), rp.ctypes.data_as(_PI32), len(rp) - 1, tv.ctypes.data_as(_PI32), len(tv), bb.ctypes.data_as(_PD))
                for verts in layers:
                    if len(verts) < 3:
                        _RUST_LIB.rc_eng_add_layer(eng, bid, oi, dd.ctypes.data_as(_PD), 0)
                    else:
                        a = np.ascontiguousarray(np.asarray(verts, dtype=np.float64))
                        if not np.array_equal(a[0], a[-1]):
                            a = np.vstack((a, a[:1]))
                        _RUST_LIB.rc_eng_add_layer(eng, bid, oi, a.ctypes.data_as(_PD), len(a))
        oi_ptr, oi_flat = ([0], [])
        for blk in blocks_data:
            seen_sig = {}
            for oi in range(len(blk['shape'])):
                sig = tuple((tuple((tuple(v) for v in layer)) for layer in blk['shape'][oi]['layers']))
                if sig not in seen_sig:
                    seen_sig[sig] = oi
                    oi_flat.append(oi)
            oi_ptr.append(len(oi_flat))
        st = geo_cache['_rust'] = {'eng': eng, 'n': n, 'm': m, 'w': (w.get('w1', 1.0), w.get('w2', 1.0), w.get('w3', 1.0)), 'r': np.array([b['release_time'] for b in blocks_data], dtype=np.int64), 'd': np.array([b['due_date'] for b in blocks_data], dtype=np.int64), 'p': np.array([b['processing_time'] for b in blocks_data], dtype=np.int64), 'wl': np.array([b['workload'] for b in blocks_data], dtype=np.float64), 'prefs': np.ascontiguousarray(np.array([b['bay_preferences'] for b in blocks_data], dtype=np.float64)).ravel(), 'n_or': np.array([len(b['shape']) for b in blocks_data], dtype=np.int32), 'oi_ptr': np.array(oi_ptr, dtype=np.int32), 'oi_flat': np.array(oi_flat, dtype=np.int32)}
    n = st['n']
    if insert_ids is not None:
        if insert_order == 'given':
            order = list(insert_ids)
        else:
            order = sorted(insert_ids, key=lambda i: (blocks_data[i]['due_date'], blocks_data[i]['release_time'], blocks_data[i]['processing_time']))
    elif plan is not None:
        if plan_bucket > 0:

            def _ba(i):
                vs = [v for l in blocks_data[i]['shape'][0]['layers'] for v in l]
                xs_ = [v[0] for v in vs]
                ys_ = [v[1] for v in vs]
                return (max(xs_) - min(xs_)) * (max(ys_) - min(ys_))
            order = sorted(range(len(blocks_data)), key=lambda i: (plan[i].get('start', 10 ** 9) // plan_bucket if i in plan else 10 ** 9, -_ba(i), blocks_data[i]['due_date'], blocks_data[i]['release_time']))
        else:
            order = sorted(range(len(blocks_data)), key=lambda i: (plan[i].get('start', 10 ** 9) if i in plan else 10 ** 9, blocks_data[i]['due_date'], blocks_data[i]['release_time']))
    else:
        order = _insertion_order(blocks_data, order_key, seed)
    order_a = np.ascontiguousarray(np.array(order, dtype=np.int32))
    if fixed_assignments:
        ff = np.array([[fb, a['bay_id'], a['x'], a['y'], a['orient_idx'], a['entry_time'], a['exit_time']] for fb, a in fixed_assignments.items()], dtype=np.int64).ravel()
        nf = len(fixed_assignments)
    else:
        ff = np.zeros(7, dtype=np.int64)
        nf = 0
    if plan is not None:
        ps = np.full(n, np.iinfo(np.int64).min, dtype=np.int64)
        pb = np.full(n, -1, dtype=np.int32)
        for b_, e_ in plan.items():
            ps[b_] = e_.get('start', np.iinfo(np.int64).min)
            pb[b_] = e_['bay_id']
        plan_active = 1
    else:
        ps = np.zeros(n, dtype=np.int64)
        pb = np.zeros(n, dtype=np.int32)
        plan_active = 0
    prior_strength = max(0.0, float(bay_prior_strength))
    prior_active = int(bay_prior is not None and prior_strength > 0.0)
    if prior_active:
        bp = np.ascontiguousarray(np.asarray(bay_prior, dtype=np.float64))
        if bp.shape != (n, st['m']):
            raise ValueError('bay_prior must have one value per block and bay')
        bp = bp.ravel()
    else:
        bp = np.zeros(max(1, n * st['m']), dtype=np.float64)
    out = np.zeros(n * 6, dtype=np.int64)
    w1, w2, w3 = st['w']
    rc = _RUST_LIB.rc_eng_run(st['eng'], n, st['r'].ctypes.data_as(_PI64), st['d'].ctypes.data_as(_PI64), st['p'].ctypes.data_as(_PI64), st['wl'].ctypes.data_as(_PD), st['prefs'].ctypes.data_as(_PD), prior_active, prior_strength, bp.ctypes.data_as(_PD), st['n_or'].ctypes.data_as(_PI32), st['oi_ptr'].ctypes.data_as(_PI32), st['oi_flat'].ctypes.data_as(_PI32), order_a.ctypes.data_as(_PI32), len(order_a), nf, ff.ctypes.data_as(_PI64), plan_active, 1 if plan_bays else 0, float(plan_target_w), ps.ctypes.data_as(_PI64), pb.ctypes.data_as(_PI32), int(max_xs), int(max_ys), int(top_b), int(nest_lane), 1 if slide else 0, w1, w2, w3, float(timelimit), out.ctypes.data_as(_PI64))
    if rc != 0:
        raise ValueError(f'block {-rc - 1}: no valid placement in any bay')
    assignments: dict[int, dict] = {}
    if fixed_assignments:
        for fb, a in fixed_assignments.items():
            assignments[fb] = dict(a)
    for bid in order:
        o = out[6 * bid:6 * bid + 6]
        assignments[bid] = {'block_id': bid, 'bay_id': int(o[0]), 'x': int(o[1]), 'y': int(o[2]), 'orient_idx': int(o[3]), 'entry_time': int(o[4]), 'exit_time': int(o[5])}
    stats = {'n_inside': 0, 'n_enclose': 0, 'n_mask_tests': 0, 'n_prep_confirm': 0, 'n_classify': 0, 'n_pos_evaluated': 0, 'n_slide_steps': 0, 'n_agg_confirm': 0, 'n_memo_hit': 0, 'n_rel_hit': 0, 'degraded': False, 'wall_s': round(time.time() - t0, 1)}
    return (assignments, stats)


def construct(prob_info: dict, timelimit: float=120.0, max_xs: int=24, max_ys: int=16, top_b: int=8, order_key: str='edd', seed=None, nest_lane: int=4, slide: bool=True, geo_cache: dict | None=None, screen: str='exact', fixed_assignments: dict | None=None, insert_ids: list | None=None, insert_order: str='edd', placed_cache: dict | None=None, pair_memo: dict | None=None, rel_memo: dict | None=None, plan: dict | None=None, plan_bays: bool=True, plan_bucket: int=0, plan_target_w: float=0.0, bay_prior: list | None=None, bay_prior_strength: float=0.0, verbose: bool=True):
    if geo_cache is None:
        geo_cache = {}
    if pair_memo is None:
        pair_memo = {}
    if rel_memo is None:
        rel_memo = {}
    nest_lane = _resolved_nest_lane(nest_lane)
    top_b = _resolved_top_b(top_b)
    if _RUST_LIB is not None and screen == 'exact' and (not _RV_IN[0]):
        r_res = _construct_rust(prob_info, timelimit, max_xs, max_ys, top_b, order_key, seed, nest_lane, slide, geo_cache, fixed_assignments, insert_ids, insert_order, plan, plan_bays, plan_bucket, plan_target_w, bay_prior, bay_prior_strength)
        return r_res
    use_kernel = screen == 'exact' and _numba_warmup()
    t0 = time.time()
    bays_data = prob_info['bays']
    blocks_data = prob_info['blocks']
    m, n = (len(bays_data), len(blocks_data))
    w1 = prob_info.get('weights', {}).get('w1', 1.0)
    w2 = prob_info.get('weights', {}).get('w2', 1.0)
    w3 = prob_info.get('weights', {}).get('w3', 1.0)
    prior_strength = max(0.0, float(bay_prior_strength))
    prior_active = bool(bay_prior is not None and prior_strength > 0.0)
    if prior_active:
        if len(bay_prior) != n or any((len(row) != m for row in bay_prior)):
            raise ValueError('bay_prior must have one value per block and bay')
    bays = [Bay.from_dict(d, i) for i, d in enumerate(bays_data)]
    areas = [b.width * b.height for b in bays]
    u = [sum(areas) / m / a for a in areas]
    native = None
    if _NATIVE_LIB is not None:
        native = geo_cache.get('_native')
        if native is None:
            native = geo_cache['_native'] = _Native()

    def geo_of(bid: int, oi: int) -> _Geo:
        g = geo_cache.get((bid, oi))
        if g is None:
            g = geo_cache[bid, oi] = _Geo(blocks_data[bid], oi)
        return g
    placed: list[list[_Placed]] = [[] for _ in range(m)]
    loads = [0.0] * m
    assignments: dict[int, dict] = {}
    if fixed_assignments:
        for fbid, a in fixed_assignments.items():
            fblk = blocks_data[fbid]
            bay_id = a['bay_id']
            key = (fbid, bay_id, a['x'], a['y'], a['orient_idx'], a['entry_time'], a['exit_time'])
            p = placed_cache.get(key) if placed_cache is not None else None
            if p is None:
                g = geo_of(fbid, a['orient_idx'])
                blk_obj = Block(block_id=fbid, block_data=fblk, x=int(a['x']), y=int(a['y']), orient_idx=a['orient_idx'])
                p = _Placed(fbid, int(a['x']), int(a['y']), a['orient_idx'], a['entry_time'], a['exit_time'], blk_obj, shp_translate(g.union, int(a['x']), int(a['y'])), g, bays[bay_id].width, bays[bay_id].height)
                if placed_cache is not None:
                    if len(placed_cache) > 4096:
                        placed_cache.clear()
                    placed_cache[key] = p
            placed[bay_id].append(p)
            loads[bay_id] += fblk['workload']
            assignments[fbid] = dict(a)
    if insert_ids is not None:
        if insert_order == 'given':
            order = list(insert_ids)
        else:
            order = sorted(insert_ids, key=lambda i: (blocks_data[i]['due_date'], blocks_data[i]['release_time'], blocks_data[i]['processing_time']))
    elif plan is not None:
        if plan_bucket > 0:

            def bbox_area(i):
                vs = [v for l in blocks_data[i]['shape'][0]['layers'] for v in l]
                xs_ = [v[0] for v in vs]
                ys_ = [v[1] for v in vs]
                return (max(xs_) - min(xs_)) * (max(ys_) - min(ys_))
            order = sorted(range(len(blocks_data)), key=lambda i: (plan[i].get('start', 10 ** 9) // plan_bucket if i in plan else 10 ** 9, -bbox_area(i), blocks_data[i]['due_date'], blocks_data[i]['release_time']))
        else:
            order = sorted(range(len(blocks_data)), key=lambda i: (plan[i].get('start', 10 ** 9) if i in plan else 10 ** 9, blocks_data[i]['due_date'], blocks_data[i]['release_time']))
    else:
        order = _insertion_order(blocks_data, order_key, seed)
    n = len(order) if order else 1
    stats = {'n_inside': 0, 'n_enclose': 0, 'n_mask_tests': 0, 'n_prep_confirm': 0, 'n_classify': 0, 'n_pos_evaluated': 0, 'n_slide_steps': 0, 'n_agg_confirm': 0, 'n_memo_hit': 0, 'n_rel_hit': 0, 'degraded': False}
    uniq_cache: dict[int, list] = {}
    log_every = max(1, n // 10)
    cur_xs, cur_ys, cur_b = (max_xs, max_ys, top_b)
    cur_nest = nest_lane
    orient_cap = 99
    for rank, bid in enumerate(order):
        frac_used = (time.time() - t0) / timelimit
        frac_done = rank / n
        if frac_used > 0.85:
            cur_xs, cur_ys, cur_b, orient_cap, cur_nest = (4, 3, 2, 2, 0)
            stats['degraded'] = True
        elif frac_used > frac_done + 0.12:
            cur_xs = max(8, cur_xs * 2 // 3)
            cur_ys = max(6, cur_ys * 2 // 3)
            orient_cap = 4
            stats['degraded'] = True
        blk = blocks_data[bid]
        R_b, D_b, P_b = (blk['release_time'], blk['due_date'], blk['processing_time'])
        wl, prefs = (blk['workload'], blk['bay_preferences'])
        if plan is not None and plan_target_w > 0.0 and (bid in plan) and ('start' in plan[bid]):
            tgt_b, ptw_b = (int(plan[bid]['start']), float(plan_target_w))
        else:
            tgt_b, ptw_b = (0, 0.0)
        smax = max(prefs)
        n_orient = len(blk['shape'])
        oi_list = uniq_cache.get(bid)
        if oi_list is None:
            seen_sig: dict = {}
            oi_list = []
            for oi in range(n_orient):
                sig = tuple((tuple((tuple(v) for v in layer)) for layer in blk['shape'][oi]['layers']))
                if sig not in seen_sig:
                    seen_sig[sig] = oi
                    oi_list.append(oi)
            uniq_cache[bid] = oi_list
        if len(oi_list) > orient_cap:
            step = len(oi_list) / orient_cap
            oi_list = [oi_list[int(i * step)] for i in range(orient_cap)]
        cands_all: list[tuple] = []
        grids: list[tuple] = []
        if prior_active:
            prior_row = bay_prior[bid]
            prior_penalties = [w1 * prior_strength * float(value) for value in prior_row]
            bay_order = sorted(range(m), key=lambda j: (w3 * (smax - prefs[j]) + prior_penalties[j], j))
        else:
            prior_penalties = [0.0] * m
            bay_order = sorted(range(m), key=lambda j: prefs[j], reverse=True)
        if plan is not None and plan_bays and (bid in plan) and (plan[bid].get('bay_id', -1) >= 0):
            bay_order = [plan[bid]['bay_id']]
        best_cons = 1e309
        for bay_id in bay_order:
            if w3 * (smax - prefs[bay_id]) + prior_penalties[bay_id] > best_cons:
                continue
            bay = bays[bay_id]
            actives = [e for e in placed[bay_id] if e.F > R_b]
            A = len(actives)
            if A:
                occ_stack = np.stack([e.mask for e in actives])
                agg_stack = np.stack([e.amask for e in actives])
                act_bbox = np.array([e.bbox for e in actives])
            if use_kernel:
                act_S = np.array([e.S for e in actives], dtype=np.int64)
                act_F = np.array([e.F for e in actives], dtype=np.int64)
                if native is not None:
                    act_gid = np.array([native.gid_of(e.geo) for e in actives], dtype=np.int32)
                    act_gx = np.array([e.x for e in actives], dtype=np.int64)
                    act_gy = np.array([e.y for e in actives], dtype=np.int64)
            if m > 1:
                new_load = loads[bay_id] + wl
                d2 = max((abs(u[bay_id] * new_load - u[j] * loads[j]) for j in range(m) if j != bay_id))
            else:
                d2 = 0.0
            base_score = w2 * d2 + w3 * (smax - prefs[bay_id]) + prior_penalties[bay_id]
            for oi in oi_list:
                g = geo_of(bid, oi)
                if g.union is None:
                    continue
                lx0, ly0, lx1, ly1 = g.bbox
                px_lo, px_hi = (math.ceil(-lx0), math.floor(bay.width - lx1))
                py_lo, py_hi = (math.ceil(-ly0), math.floor(bay.height - ly1))
                if px_lo > px_hi or py_lo > py_hi:
                    continue
                xs, ys = ({px_lo, px_hi}, {py_lo, py_hi})
                for e in actives:
                    ex0, ey0, ex1, ey1 = e.bbox
                    xs.update((math.ceil(ex1 - lx0), math.floor(ex0 - lx1), math.ceil(ex0 - lx0)))
                    ys.update((math.ceil(ey1 - ly0), math.floor(ey0 - ly1), math.ceil(ey0 - ly0)))
                xs = sorted((x for x in xs if px_lo <= x <= px_hi))
                ys = sorted((y for y in ys if py_lo <= y <= py_hi))
                if len(xs) > cur_xs:
                    idx = np.linspace(0, len(xs) - 1, cur_xs).round().astype(int)
                    xs = [xs[i] for i in dict.fromkeys(idx.tolist())]
                if len(ys) > cur_ys:
                    idx = np.linspace(0, len(ys) - 1, cur_ys).round().astype(int)
                    ys = [ys[i] for i in dict.fromkeys(idx.tolist())]
                if not xs or not ys:
                    continue
                if use_kernel:
                    xs_a = np.array(xs, dtype=np.int64)
                    ys_a = np.array(ys, dtype=np.int64)
                    P_ = len(xs) * len(ys)
                    stats['n_pos_evaluated'] += P_
                    if A:
                        cells = g.cells
                        if cells is None:
                            ci_, cj_ = np.nonzero(g.mask)
                            ai_, aj_ = np.nonzero(g.amask)
                            cells = g.cells = (ci_, cj_, ai_, aj_)
                        conf_ptr, conf_flat, res_ptr, res_flat, nmask, nagg = _k_tiers(occ_stack, agg_stack, act_bbox, xs_a, ys_a, 2, g.ox, g.oy, cells[0], cells[1], cells[2], cells[3], lx0, ly0, lx1, ly1)
                        stats['n_mask_tests'] += int(nmask)
                        stats['n_agg_confirm'] += int(nagg)
                        if res_flat.shape[0] and native is not None:
                            res_ok = native.resolve(g, bid, oi, actives, xs, ys, act_gid, act_gx, act_gy, xs_a, ys_a, res_ptr, res_flat, stats)
                        elif res_flat.shape[0]:
                            res_ok = _resolve_geos(g, bid, oi, actives, xs, ys, res_ptr, res_flat, pair_memo, stats)
                        else:
                            res_ok = np.zeros(0, dtype=bool)
                    else:
                        zp = np.zeros(P_ + 1, dtype=np.int32)
                        conf_ptr = res_ptr = zp
                        conf_flat = res_flat = np.empty(0, dtype=np.int32)
                        res_ok = np.empty(0, dtype=bool)
                    scores, scores_opt = _k_score(conf_ptr, conf_flat, res_ptr, res_flat, res_ok, act_S, act_F, xs_a, ys_a, R_b, P_b, D_b, base_score, w1, ly1, 1 if cur_nest else 0, tgt_b, ptw_b)
                    mn = float(scores.min())
                    if mn < best_cons:
                        best_cons = mn
                    grids.append((bay_id, oi, xs, ys, scores, scores_opt, conf_ptr, conf_flat, res_ptr, res_flat, res_ok))
                    continue
                pos = [(x, y) for y in ys for x in xs]
                stats['n_pos_evaluated'] += len(pos)
                gm, gam, gox, goy = (g.mask, g.amask, g.ox, g.oy)
                ghh, gww = gm.shape
                if A:
                    parr = np.array(pos, dtype=float)
                    hitm = (parr[:, None, 0] + lx0 < act_bbox[None, :, 2]) & (act_bbox[None, :, 0] < parr[:, None, 0] + lx1) & (parr[:, None, 1] + ly0 < act_bbox[None, :, 3]) & (act_bbox[None, :, 1] < parr[:, None, 1] + ly1)
                for pi, (x, y) in enumerate(pos):
                    conf_idx = []
                    if A:
                        idxs = np.nonzero(hitm[pi])[0]
                        if len(idxs):
                            r0, c0 = (y * 2 + goy, x * 2 + gox)
                            stats['n_mask_tests'] += 1
                            sub = occ_stack[idxs, r0:r0 + ghh, c0:c0 + gww]
                            mh = np.nonzero((sub & gm).any(axis=(1, 2)))[0]
                            if screen == 'mask':
                                conf_idx = [int(idxs[k_]) for k_ in mh]
                            elif len(mh):
                                sel = idxs[mh]
                                suba = agg_stack[sel, r0:r0 + ghh, c0:c0 + gww]
                                agg_hit = (suba & gam).any(axis=(1, 2))
                                bu = None
                                for j_, k in enumerate(sel.tolist()):
                                    if agg_hit[j_]:
                                        stats['n_agg_confirm'] += 1
                                        conf_idx.append(k)
                                        continue
                                    e = actives[k]
                                    mkey = (e.bid, e.oi, bid, oi, e.x, e.y, x, y)
                                    v = pair_memo.get(mkey)
                                    if v is None:
                                        stats['n_prep_confirm'] += 1
                                        if bu is None:
                                            bu = shp_translate(g.union, x, y)
                                        v = bool(e.prep.intersects(bu))
                                        if len(pair_memo) >= 262144:
                                            pair_memo.clear()
                                        pair_memo[mkey] = v
                                    else:
                                        stats['n_memo_hit'] += 1
                                    if v:
                                        conf_idx.append(k)
                    ivals = [(actives[k].S, actives[k].F) for k in conf_idx]
                    entry = _earliest_gap(R_b, P_b, ivals)
                    tard = max(0, entry + P_b - D_b)
                    score = base_score + w1 * tard + 0.0001 * (y + ly1) + 1e-06 * entry + ptw_b * abs(entry - tgt_b)
                    score_opt = score
                    if cur_nest and conf_idx:
                        hard = [(S_e, F_e) for S_e, F_e in ivals if not (S_e <= R_b and R_b + P_b <= F_e)]
                        if len(hard) < len(ivals):
                            e_opt = _earliest_gap(R_b, P_b, hard)
                            t_opt = max(0, e_opt + P_b - D_b)
                            score_opt = base_score + w1 * t_opt + 0.0001 * (y + ly1) + 1e-06 * e_opt + ptw_b * abs(e_opt - tgt_b)
                    if score < best_cons:
                        best_cons = score
                    cands_all.append((score, score_opt, bay_id, oi, x, y, conf_idx))
        if not cands_all and (not grids):
            fallback_bays = bay_order if prior_active else sorted(range(m), key=lambda j: prefs[j], reverse=True)
            for bay_id in fallback_bays:
                bay = bays[bay_id]
                actives = [e for e in placed[bay_id] if e.F > R_b]
                for oi in range(n_orient):
                    g = geo_of(bid, oi)
                    if g.union is None:
                        continue
                    lx0, ly0, lx1, ly1 = g.bbox
                    px_lo = math.ceil(-lx0)
                    py_lo = math.ceil(-ly0)
                    if px_lo > math.floor(bay.width - lx1) or py_lo > math.floor(bay.height - ly1):
                        continue
                    wbb = (px_lo + lx0, py_lo + ly0, px_lo + lx1, py_lo + ly1)
                    conf_idx = [k for k, e in enumerate(actives) if wbb[0] < e.bbox[2] and e.bbox[0] < wbb[2] and (wbb[1] < e.bbox[3]) and (e.bbox[1] < wbb[3])]
                    cands_all.append((1e309, 1e309, bay_id, oi, px_lo, py_lo, conf_idx))
        if not cands_all and (not grids):
            raise ValueError(f'block {bid}: no valid placement in any bay')
        if grids:
            sc_all = np.concatenate([gr[4] for gr in grids])
            so_all = np.concatenate([gr[5] for gr in grids])
            order = np.argsort(sc_all, kind='stable')
            sel_idx = order[:cur_b].tolist()
            if cur_nest:
                nest_seq = order[so_all[order] < sc_all[order] - 1e-09]
                nest_seq = nest_seq[np.argsort(so_all[nest_seq], kind='stable')]
                for idx in nest_seq[:cur_nest].tolist():
                    if idx not in sel_idx:
                        sel_idx.append(idx)
            gstart = np.cumsum([0] + [gr[4].shape[0] for gr in grids])
            selected = []
            for idx in sel_idx:
                gi = int(np.searchsorted(gstart, idx, side='right')) - 1
                bay_id_g, oi_g, xs_g, ys_g, sc_g, so_g, cptr, cflat, rptr, rflat, rok = grids[gi]
                p_ = idx - int(gstart[gi])
                nxl = len(xs_g)
                x_ = xs_g[p_ % nxl]
                y_ = ys_g[p_ // nxl]
                ca = [int(v) for v in cflat[cptr[p_]:cptr[p_ + 1]]]
                ra = [int(rflat[ti]) for ti in range(rptr[p_], rptr[p_ + 1]) if rok[ti]]
                selected.append((float(sc_g[p_]), float(so_g[p_]), bay_id_g, oi_g, x_, y_, sorted(ca + ra)))
        else:
            cands_all.sort(key=lambda c: c[0])
            selected = list(cands_all[:cur_b])
            if cur_nest:
                nesty = [c for c in cands_all if c[1] < c[0] - 1e-09]
                nesty.sort(key=lambda c: c[1])
                for c in nesty[:cur_nest]:
                    if c not in selected:
                        selected.append(c)
        best = None
        best_score = 1e309
        for score_cons, score_opt, bay_id, oi, x, y, conf_idx in selected:
            actives = [e for e in placed[bay_id] if e.F > R_b]
            g = geo_of(bid, oi)
            cand_block = None
            conflicts = []
            for k in conf_idx:
                e = actives[k]
                rkey = (e.bid, e.oi, bid, oi, e.x, e.y, x, y)
                rv = rel_memo.get(rkey)
                if rv is None:
                    stats['n_classify'] += 1
                    if cand_block is None:
                        cand_block = Block(block_id=bid, block_data=blk, x=x, y=y, orient_idx=oi)
                    rel = pair_relation(e.block, cand_block)
                    rv = (rel[0], len(rel) > 1 and rel[1] == bid)
                    if len(rel_memo) >= 262144:
                        rel_memo.clear()
                    rel_memo[rkey] = rv
                else:
                    stats['n_rel_hit'] += 1
                rk, under_is_b = rv
                if rk == FREE:
                    continue
                kind = 'excl' if rk == EXCLUSIVE else 'b_enclose' if under_is_b else 'b_inside'
                conflicts.append((e.S, e.F, kind))
            entry, exit_t = _earliest_slot(R_b, P_b, conflicts)
            if slide and all((F_e <= entry for _, F_e, _ in conflicts)):
                nx, ny = _slide_to_contact(x, y, g, actives, entry, bid, oi, pair_memo, stats, native=native)
                stats['n_slide_steps'] += abs(nx - x) + abs(ny - y)
                x, y = (nx, ny)
            tard = max(0, exit_t - D_b)
            if m > 1:
                new_load = loads[bay_id] + wl
                d2 = max((abs(u[bay_id] * new_load - u[j] * loads[j]) for j in range(m) if j != bay_id))
            else:
                d2 = 0.0
            _, _, _, ly1g = g.bbox
            score = w1 * tard + w2 * d2 + w3 * (smax - prefs[bay_id]) + prior_penalties[bay_id] + 0.0001 * (y + ly1g) + 1e-06 * entry + ptw_b * abs(entry - tgt_b)
            if score < best_score:
                best_score = score
                best = (bay_id, x, y, oi, entry, exit_t, conflicts)
        bay_id, x, y, oi, entry, exit_t, conflicts = best
        g = geo_of(bid, oi)
        blk_obj = Block(block_id=bid, block_data=blk, x=x, y=y, orient_idx=oi)
        placed[bay_id].append(_Placed(bid, x, y, oi, entry, exit_t, blk_obj, shp_translate(g.union, x, y), g, bays[bay_id].width, bays[bay_id].height))
        loads[bay_id] += wl
        assignments[bid] = {'block_id': bid, 'bay_id': bay_id, 'x': int(x), 'y': int(y), 'orient_idx': int(oi), 'entry_time': int(entry), 'exit_time': int(exit_t)}
        for S_e, F_e, kind in conflicts:
            if S_e < exit_t and entry < F_e:
                if kind == 'b_inside':
                    stats['n_inside'] += 1
                elif kind == 'b_enclose':
                    stats['n_enclose'] += 1
        if verbose and ((rank + 1) % log_every == 0 or rank + 1 == n):
            el = time.time() - t0
            print(f'[Con]  {rank + 1:4d}/{n}  block{bid:<4d} -> bay{bay_id} ({x},{y}) oi={oi} t=[{entry},{exit_t})  grid={cur_xs}x{cur_ys}  {el:.1f}s')
    stats['wall_s'] = round(time.time() - t0, 1)
    return (assignments, stats)

def _objective_of(prob_info: dict, assignments: dict):
    bays_data = prob_info['bays']
    blocks_data = prob_info['blocks']
    m = len(bays_data)
    w = prob_info.get('weights', {})
    w1, w2, w3 = (w.get('w1', 1.0), w.get('w2', 1.0), w.get('w3', 1.0))
    obj1 = 0.0
    obj3 = 0.0
    loads = [0.0] * m
    for bid, a in assignments.items():
        blk = blocks_data[bid]
        obj1 += max(0.0, a['exit_time'] - blk['due_date'])
        loads[a['bay_id']] += blk['workload']
        obj3 += max(blk['bay_preferences']) - blk['bay_preferences'][a['bay_id']]
    if m > 1:
        areas = [b['width'] * b['height'] for b in bays_data]
        avg = sum(areas) / m
        u = [avg / x for x in areas]
        obj2 = float(math.floor(max((abs(u[i] * loads[i] - u[j] * loads[j]) for i in range(m) for j in range(m) if i != j))))
    else:
        obj2 = 0.0
    return (w1 * obj1 + w2 * obj2 + w3 * obj3, (obj1, obj2, obj3))

def _assignment_hash(assignments: dict) -> str:
    import hashlib
    import struct
    digest = hashlib.blake2b(digest_size=8, person=b'OGC-v512-mseed')
    for bid, a in sorted(assignments.items()):
        digest.update(struct.pack('<7q', int(bid), int(a['bay_id']), int(a['x']), int(a['y']), int(a['orient_idx']), int(a['entry_time']), int(a['exit_time'])))
    return digest.hexdigest()

def construct_multi(prob_info: dict, timelimit: float=120.0, verbose: bool=True, use_legacy: bool=True, scout_frac: float=0.5, refine_top: int=3, scout_skip: tuple | None=None, max_passes: int | None=None, pass_tl: float=30.0, var_offset: int=0, var_slice: tuple | None=None, collect_top: dict | None=None, **kw):
    t0 = time.time()
    fast = {'screen': 'exact'}
    scout_vars = [('edd', None), ('edd_big', None), ('edd_long', None), ('release', None)]
    scout_vars += [('shuffle', s) for s in range(1 + var_offset, 400)]
    if scout_skip is not None:
        scout_vars = [v for v in scout_vars if v != tuple(scout_skip)]
    if var_slice is not None:
        scout_vars = scout_vars[var_slice[0]::var_slice[1]]
    shared_geo = kw.pop('geo_cache', None)
    if shared_geo is None:
        shared_geo = {}
    kw.setdefault('pair_memo', {})
    kw.setdefault('rel_memo', {})
    best_assign = None
    best_stats: dict = {}
    best_obj = 1e309
    best_parts = None
    best_var = '-'
    passes = 0
    scouted: list[tuple] = []

    def run_pass(ok_, seed, extra, tag_suffix=''):
        nonlocal best_assign, best_stats, best_obj, best_parts, best_var, passes
        remaining = pass_tl if max_passes is not None else timelimit - (time.time() - t0)
        assign, stats = construct(prob_info, timelimit=remaining, order_key=ok_, seed=seed, geo_cache=shared_geo, verbose=False, **{**kw, **extra})
        obj, parts = _objective_of(prob_info, assign)
        passes += 1
        tag = (ok_ if seed is None else f'{ok_}#{seed}') + tag_suffix
        if verbose:
            marker = ' *' if obj < best_obj else ''
            print(f'[Multi] pass {passes:2d} {tag:16s} obj={obj:>14,.0f} (o1={parts[0]:.0f} o2={parts[1]:.0f} o3={parts[2]:.0f})  {stats['wall_s']:.1f}s{marker}')
        if obj < best_obj:
            best_obj, best_assign, best_parts = (obj, assign, parts)
            best_stats, best_var = (stats, tag)
        if collect_top is not None:
            _h = _assignment_hash(assign)
            collect_top['rows'].append((obj, passes, _h))
            stream_q = collect_top.get('stream_queue')
            if stream_q is not None:
                stream_q.put(('candidate', int(collect_top.get('stream_worker', -1)), int(passes), float(obj), _h, assign))
            _ca = collect_top['assigns']
            _ca.append((obj, passes, _h, assign))
            if len(_ca) > collect_top.get('k', 8):
                _ca.sort(key=lambda t: (t[0], t[1]))
                del _ca[collect_top.get('k', 8):]
        return (obj, stats['wall_s'])
    longest_exact = 0.0
    if use_legacy:
        _, dur = run_pass('edd', None, {'slide': False, 'nest_lane': 0, 'screen': 'exact'}, '(legacy)')
        longest_exact = dur
    explore_frac = scout_frac if fast.get('screen') != 'exact' else 1.0
    longest_fast = 0.0
    n_explore = 0
    for ok_, seed in scout_vars:
        stop_event = collect_top.get('stop_event') if collect_top is not None else None
        if stop_event is not None and stop_event.is_set():
            break
        elapsed = time.time() - t0
        if max_passes is not None:
            if n_explore >= max_passes or elapsed > timelimit - 2.0:
                break
        else:
            if elapsed > timelimit * explore_frac:
                break
            if best_assign is not None and timelimit - elapsed < max(5.0, 1.05 * max(longest_fast, 1.0)):
                break
        obj, dur = run_pass(ok_, seed, fast)
        n_explore += 1
        longest_fast = max(longest_fast, dur)
        scouted.append((obj, ok_, seed))
    if fast.get('screen') != 'exact':
        scouted.sort()
        exact_est = max(longest_exact, 3.0 * longest_fast, 1.0)
        for obj_s, ok_, seed in scouted[:max(0, refine_top)]:
            remaining = timelimit - (time.time() - t0)
            if remaining < max(5.0, 0.9 * exact_est):
                break
            _, dur = run_pass(ok_, seed, {'screen': 'exact'}, '(exact)')
            exact_est = max(exact_est, dur)
    best_stats = dict(best_stats, passes=passes, best_variant=best_var, wall_s=round(time.time() - t0, 1))
    if verbose:
        print(f'[Multi] best={best_var}  obj={best_obj:,.0f} (o1={best_parts[0]:.0f})  passes={passes}')
    return (best_assign, best_stats)
