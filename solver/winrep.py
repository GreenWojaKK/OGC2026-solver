import ctypes
import math
import os
import time
import numpy as np
from utils import Bay, check_feasibility, _bounding_box, _resolve_layers
from gpr_core import retime
_CONF_KIND = {0: 'EXCL', 1: 'B_UNDER', 2: 'B_OVER'}
_PAIR_KIND = {0: 'EXCL', 1: 'U12', 2: 'U21'}
_PD = ctypes.POINTER(ctypes.c_double)
_PI64 = ctypes.POINTER(ctypes.c_int64)
_PI32 = ctypes.POINTER(ctypes.c_int32)
_PU8 = ctypes.POINTER(ctypes.c_uint8)
_RLIB = None

def _rust_lib():
    global _RLIB
    if _RLIB is not None:
        return _RLIB or None
    here = os.path.dirname(os.path.abspath(__file__))
    for p in (os.path.join(here, 'rustcore.so'), os.path.join(here, '..', 'dev_rust', 'rustcore.so')):
        if os.path.exists(p):
            try:
                lib = ctypes.CDLL(p)
                lib.rc_eng_new.restype = ctypes.c_void_p
                lib.rc_eng_new.argtypes = [ctypes.c_int64, _PI64, _PI64, ctypes.c_int32]
                lib.rc_eng_del.restype = None
                lib.rc_eng_del.argtypes = [ctypes.c_void_p]
                lib.rc_eng_add_geo.restype = ctypes.c_int32
                lib.rc_eng_add_geo.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, _PD, ctypes.c_int32, _PI32, ctypes.c_int32, _PI32, ctypes.c_int32, _PD]
                lib.rc_eng_add_layer.restype = None
                lib.rc_eng_add_layer.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, _PD, ctypes.c_int32]
                lib.rc_win_enum.restype = ctypes.c_void_p
                lib.rc_win_enum.argtypes = [ctypes.c_void_p, ctypes.c_int32, _PI32, _PI32, _PI64, _PI64, _PI32, _PI64, _PI64, ctypes.c_int32, _PI32, _PI64, _PI64, _PI64, _PU8, _PI32, _PI64, _PI64, _PI32, _PI32, _PI32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
                lib.rc_win_counts.restype = None
                lib.rc_win_counts.argtypes = [ctypes.c_void_p, _PI64]
                lib.rc_win_fetch.restype = None
                lib.rc_win_fetch.argtypes = [ctypes.c_void_p, _PI64, _PI64, _PI64]
                lib.rc_win_del.restype = None
                lib.rc_win_del.argtypes = [ctypes.c_void_p]
                _RLIB = lib
                return lib
            except Exception:
                continue
    _RLIB = False
    return None

def _reg_geo(lib, eng, bid, oi, layers):
    dd = np.zeros(2, dtype=np.float64)
    di = np.zeros(2, dtype=np.int32)
    bb = np.ascontiguousarray(np.array(_bounding_box([v for l in layers for v in l]), dtype=np.float64))
    lib.rc_eng_add_geo(eng, bid, oi, 0, dd.ctypes.data_as(_PD), 0, di.ctypes.data_as(_PI32), 0, di.ctypes.data_as(_PI32), 0, bb.ctypes.data_as(_PD))
    for verts in layers:
        if len(verts) < 3:
            lib.rc_eng_add_layer(eng, bid, oi, dd.ctypes.data_as(_PD), 0)
        else:
            a = np.ascontiguousarray(np.asarray(verts, dtype=np.float64))
            if not np.array_equal(a[0], a[-1]):
                a = np.vstack((a, a[:1]))
            lib.rc_eng_add_layer(eng, bid, oi, a.ctypes.data_as(_PD), len(a))

def _allowed_bays(bid, off, cur_bay, prefs, m):
    if bid in off:
        return [off[bid]]
    alts = sorted((j for j in range(m) if j != cur_bay), key=lambda j: -prefs[j])[:2]
    return [cur_bay] + alts

def _enum_windows(lib, prob, asgn, off, cand_m, grid_nx, grid_ny, max_window, explicit_window_ids=None):
    blocks = prob['blocks']
    bays = [Bay.from_dict(d, i) for i, d in enumerate(prob['bays'])]
    m = len(bays)
    window = dict(off)
    if explicit_window_ids is None:
        residents = []
        for bid, want in off.items():
            b = blocks[bid]
            lo, hi = (b['release_time'], b['due_date'])
            for j_bid, a in asgn.items():
                if a['bay_id'] != want or j_bid in off:
                    continue
                ov = min(hi, a['exit_time']) - max(lo, a['entry_time'])
                if ov > 0:
                    residents.append((ov, j_bid))
        residents.sort(reverse=True)
        for ov, j_bid in residents:
            if len(window) >= max_window:
                break
            window.setdefault(j_bid, asgn[j_bid]['bay_id'])
    else:
        for j_bid in sorted({int(bid) for bid in explicit_window_ids}):
            if j_bid not in asgn:
                continue
            if j_bid not in window and len(window) >= max_window:
                break
            window.setdefault(j_bid, asgn[j_bid]['bay_id'])
    win_ids = sorted(window)
    frozen_ids = sorted((b for b in asgn if b not in window))
    eng = lib.rc_eng_new(1, np.ascontiguousarray([b.width for b in bays], dtype=np.int64).ctypes.data_as(_PI64), np.ascontiguousarray([b.height for b in bays], dtype=np.int64).ctypes.data_as(_PI64), m)
    try:
        for fid in frozen_ids:
            oi = int(asgn[fid]['orient_idx'])
            _reg_geo(lib, eng, fid, oi, _resolve_layers(blocks[fid]['shape'][oi]['layers']))
        for bid in win_ids:
            for oi in range(len(blocks[bid]['shape'])):
                _reg_geo(lib, eng, bid, oi, _resolve_layers(blocks[bid]['shape'][oi]['layers']))
        nf = len(frozen_ids)
        f32 = lambda k: np.ascontiguousarray([asgn[f][k] for f in frozen_ids], dtype=np.int32)
        f64 = lambda k: np.ascontiguousarray([asgn[f][k] for f in frozen_ids], dtype=np.int64)
        fz_bid = np.ascontiguousarray(frozen_ids, dtype=np.int32)
        fz_oi, fz_bay = (f32('orient_idx'), f32('bay_id'))
        fz_x, fz_y = (f64('x'), f64('y'))
        fz_s, fz_f = (f64('entry_time'), f64('exit_time'))
        nw = len(win_ids)
        wn_bid = np.ascontiguousarray(win_ids, dtype=np.int32)
        wn_r = np.ascontiguousarray([blocks[b]['release_time'] for b in win_ids], dtype=np.int64)
        wn_p = np.ascontiguousarray([blocks[b]['processing_time'] for b in win_ids], dtype=np.int64)
        wn_d = np.ascontiguousarray([blocks[b]['due_date'] for b in win_ids], dtype=np.int64)
        wn_is_off = np.ascontiguousarray([1 if b in off and off[b] != asgn[b]['bay_id'] else 0 for b in win_ids], dtype=np.uint8)
        wn_cur_bay = np.ascontiguousarray([asgn[b]['bay_id'] for b in win_ids], dtype=np.int32)
        wn_cur_x = np.ascontiguousarray([asgn[b]['x'] for b in win_ids], dtype=np.int64)
        wn_cur_y = np.ascontiguousarray([asgn[b]['y'] for b in win_ids], dtype=np.int64)
        wn_cur_oi = np.ascontiguousarray([asgn[b]['orient_idx'] for b in win_ids], dtype=np.int32)
        ap, af = ([0], [])
        for bid in win_ids:
            al = _allowed_bays(bid, off, asgn[bid]['bay_id'], blocks[bid]['bay_preferences'], m)
            af.extend(al)
            ap.append(len(af))
        ap = np.ascontiguousarray(ap, dtype=np.int32)
        af = np.ascontiguousarray(af, dtype=np.int32)
        res = lib.rc_win_enum(eng, nf, fz_bid.ctypes.data_as(_PI32), fz_oi.ctypes.data_as(_PI32), fz_x.ctypes.data_as(_PI64), fz_y.ctypes.data_as(_PI64), fz_bay.ctypes.data_as(_PI32), fz_s.ctypes.data_as(_PI64), fz_f.ctypes.data_as(_PI64), nw, wn_bid.ctypes.data_as(_PI32), wn_r.ctypes.data_as(_PI64), wn_p.ctypes.data_as(_PI64), wn_d.ctypes.data_as(_PI64), wn_is_off.ctypes.data_as(_PU8), wn_cur_bay.ctypes.data_as(_PI32), wn_cur_x.ctypes.data_as(_PI64), wn_cur_y.ctypes.data_as(_PI64), wn_cur_oi.ctypes.data_as(_PI32), ap.ctypes.data_as(_PI32), af.ctypes.data_as(_PI32), grid_nx, grid_ny, cand_m, 0)
        cnt = np.zeros(3, dtype=np.int64)
        lib.rc_win_counts(res, cnt.ctypes.data_as(_PI64))
        c_buf = np.zeros(cnt[0] * 8, dtype=np.int64)
        f_buf = np.zeros(max(1, cnt[1] * 3), dtype=np.int64)
        p_buf = np.zeros(max(1, cnt[2] * 5), dtype=np.int64)
        lib.rc_win_fetch(res, c_buf.ctypes.data_as(_PI64), f_buf.ctypes.data_as(_PI64), p_buf.ctypes.data_as(_PI64))
        lib.rc_win_del(res)
    finally:
        lib.rc_eng_del(eng)
    entries = {}
    marshal = []
    cand_geo = {}
    for bid in win_ids:
        b = blocks[bid]
        cur = asgn[bid]
        e = {'id': bid, 'R': int(b['release_time']), 'P': int(b['processing_time']), 'D': int(b['due_date']), 'wl': b['workload'], 'cands': [], 'hint_cand': 0, 'current_bay': int(cur['bay_id']), 'hint_S': int(cur['entry_time']), 'hint_F': int(cur['exit_time'])}
        entries[bid] = e
        marshal.append(e)
    f_rows = f_buf[:cnt[1] * 3].reshape(-1, 3)
    for row in c_buf.reshape(-1, 8):
        slot, bay, x, y, oi, is_cur, c_off, c_len = (int(v) for v in row)
        bid = win_ids[slot]
        prefs = prob['blocks'][bid]['bay_preferences']
        e = entries[bid]
        ci = len(e['cands'])
        e['cands'].append({'bay': bay, 'pref_pen': max(prefs) - prefs[bay], 'conflicts': [[int(s), int(f), _CONF_KIND[int(k)]] for s, f, k in f_rows[c_off:c_off + c_len]]})
        cand_geo[bid, ci] = (bay, x, y, oi)
        if is_cur:
            e['hint_cand'] = ci
    pairs = [{'b1': win_ids[r[0]], 'c1': int(r[1]), 'b2': win_ids[r[2]], 'c2': int(r[3]), 'kind': _PAIR_KIND[int(r[4])]} for r in p_buf[:cnt[2] * 5].reshape(-1, 5)]
    return (marshal, cand_geo, pairs)

def _solve(prob, marshal, pairs, frozen_loads, det_units, wall_cap, workers=4, quality_limit=None, frozen_objective_constant=0.0, novelty=False):
    from ortools.sat.python import cp_model
    bays = prob['bays']
    m = len(bays)
    areas = [b['width'] * b['height'] for b in bays]
    u = [sum(areas) / m / a for a in areas]
    w = prob['weights']
    horizon = max((max(b['due_date'], b['release_time'] + b['processing_time']) for b in prob['blocks']))
    horizon += sum((e['P'] for e in marshal)) + 8
    model = cp_model.CpModel()
    S, F, T, Y = ({}, {}, {}, {})
    for b in marshal:
        i = b['id']
        S[i] = model.NewIntVar(b['R'], horizon, f's{i}')
        F[i] = model.NewIntVar(b['R'] + b['P'], horizon, f'f{i}')
        T[i] = model.NewIntVar(0, horizon, f't{i}')
        model.Add(F[i] >= S[i] + b['P'])
        model.Add(T[i] >= F[i] - b['D'])
        ys = [model.NewBoolVar(f'y{i}_{c}') for c in range(len(b['cands']))]
        model.AddExactlyOne(ys)
        Y[i] = ys
        model.AddHint(ys[b['hint_cand']], 1)
        model.AddHint(S[i], b['hint_S'])
        model.AddHint(F[i], b['hint_F'])
        for c, cand in enumerate(b['cands']):
            for S_e, F_e, kind in cand['conflicts']:
                opts = []
                o_a = model.NewBoolVar('')
                model.Add(S[i] >= F_e).OnlyEnforceIf(o_a)
                opts.append(o_a)
                o_b = model.NewBoolVar('')
                model.Add(F[i] <= S_e).OnlyEnforceIf(o_b)
                opts.append(o_b)
                if kind == 'B_UNDER':
                    o_c = model.NewBoolVar('')
                    model.Add(S[i] <= S_e).OnlyEnforceIf(o_c)
                    model.Add(F[i] >= F_e).OnlyEnforceIf(o_c)
                    opts.append(o_c)
                elif kind == 'B_OVER':
                    o_c = model.NewBoolVar('')
                    model.Add(S[i] >= S_e).OnlyEnforceIf(o_c)
                    model.Add(F[i] <= F_e).OnlyEnforceIf(o_c)
                    opts.append(o_c)
                model.AddBoolOr([Y[i][c].Not()] + opts)
    for p in pairs:
        i, c1, j, c2 = (p['b1'], p['c1'], p['b2'], p['c2'])
        guard = [Y[i][c1].Not(), Y[j][c2].Not()]
        o_ij = model.NewBoolVar('')
        model.Add(F[i] <= S[j]).OnlyEnforceIf(o_ij)
        o_ji = model.NewBoolVar('')
        model.Add(F[j] <= S[i]).OnlyEnforceIf(o_ji)
        opts = [o_ij, o_ji]
        if p['kind'] == 'U12':
            o_c = model.NewBoolVar('')
            model.Add(S[i] <= S[j]).OnlyEnforceIf(o_c)
            model.Add(F[i] >= F[j]).OnlyEnforceIf(o_c)
            opts.append(o_c)
        elif p['kind'] == 'U21':
            o_c = model.NewBoolVar('')
            model.Add(S[j] <= S[i]).OnlyEnforceIf(o_c)
            model.Add(F[j] >= F[i]).OnlyEnforceIf(o_c)
            opts.append(o_c)
        model.AddBoolOr(guard + opts)
    loads = []
    for j in range(m):
        terms = [int(round(b['wl'] * u[j] * 1000)) * Y[b['id']][c] for b in marshal for c, cand in enumerate(b['cands']) if cand['bay'] == j]
        lj = model.NewIntVar(0, 10 ** 12, f'L{j}')
        model.Add(lj == int(round(frozen_loads[j] * u[j] * 1000)) + (sum(terms) if terms else 0))
        loads.append(lj)
    B = model.NewIntVar(0, 10 ** 12, 'bal')
    for a in range(m):
        for bb in range(m):
            if a != bb:
                model.Add(B >= loads[a] - loads[bb])
    pref = sum((int(round(cand['pref_pen'] * 1000)) * Y[b['id']][c] for b in marshal for c, cand in enumerate(b['cands'])))
    if novelty:
        if quality_limit is None:
            raise ValueError('novelty solve requires a quality limit')
        official_scaled = int(round(w['w1'])) * 1000 * sum(T.values()) + int(round(w['w2'])) * B + int(round(w['w3'])) * pref + int(round(float(frozen_objective_constant) * 1000))
        quality_scaled = int(math.floor(float(quality_limit) * 1000 + 1e-09))
        model.Add(official_scaled <= quality_scaled)
        changed_terms = [Y[b['id']][c] for b in marshal for c, cand in enumerate(b['cands']) if int(cand['bay']) != int(b['current_bay'])]
        changed = sum(changed_terms) if changed_terms else 0
        novelty_unit = max(1, quality_scaled + 1)
        model.Maximize(novelty_unit * changed - official_scaled)
    else:
        model.Minimize(int(round(10 ** 6 * 1000)) * sum(T.values()) + int(round(w['w2'])) * B + int(round(w['w3'])) * pref)
    solve_workers = max(1, int(workers))

    def run(interleave):
        solver = cp_model.CpSolver()
        solver.parameters.max_deterministic_time = float(det_units)
        solver.parameters.max_time_in_seconds = float(wall_cap)
        solver.parameters.random_seed = 0
        solver.parameters.num_search_workers = solve_workers if interleave else 1
        if interleave:
            solver.parameters.interleave_search = True
        st = solver.Solve(model)
        return (solver, st)
    try:
        solver, st = run(solve_workers > 1)
    except Exception:
        solver, st = run(False)
    finally:
        pass
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None
    out = {}
    for b in marshal:
        i = b['id']
        ci = next((c for c, yv in enumerate(Y[i]) if solver.Value(yv)))
        out[i] = (ci, solver.Value(S[i]), solver.Value(F[i]))
    return out

def _repair_once(lib, prob, asgn, off_rows, budget_left, explicit_window_ids=None, solver_wall_cap=None, solver_det_units=None, solver_workers=4, quality_limit=None, novelty=False):
    off = {int(r[0]): int(r[1]) for r in off_rows}
    if not off:
        return None
    marshal, cand_geo, pairs = _enum_windows(lib, prob, asgn, off, cand_m=12, grid_nx=18, grid_ny=14, max_window=80, explicit_window_ids=explicit_window_ids)
    if not marshal:
        return None
    win_set = {e['id'] for e in marshal}
    m = len(prob['bays'])
    frozen_loads = [sum((prob['blocks'][bid]['workload'] for bid, a in asgn.items() if a['bay_id'] == j and bid not in win_set)) for j in range(m)]
    frozen_objective_constant = 0.0
    if novelty:
        weights = prob.get('weights', {})
        w1 = float(weights.get('w1', 1.0))
        w3 = float(weights.get('w3', 1.0))
        for bid, current in asgn.items():
            if bid in win_set:
                continue
            block = prob['blocks'][bid]
            frozen_objective_constant += w1 * max(0.0, float(current['exit_time']) - float(block['due_date'])) + w3 * (max(block['bay_preferences']) - block['bay_preferences'][int(current['bay_id'])])
    det_units = max(4, min(32, len(pairs) // 2200)) if solver_det_units is None else max(0.05, float(solver_det_units))
    wall_cap = max(8.0, min(40.0, budget_left - 4.0)) if solver_wall_cap is None else max(0.05, float(solver_wall_cap))
    res = _solve(prob, marshal, pairs, frozen_loads, det_units, wall_cap, workers=solver_workers, quality_limit=quality_limit, frozen_objective_constant=frozen_objective_constant, novelty=novelty)
    if res is None:
        return None
    new_asgn = {bid: dict(a) for bid, a in asgn.items()}
    for bid, (ci, s, f) in res.items():
        j, x, y, k = cand_geo[bid, ci]
        new_asgn[bid] = {'bay_id': j, 'x': x, 'y': y, 'orient_idx': k, 'entry_time': s, 'exit_time': f}
    return new_asgn

def _peak_rows(prob, asgn, movers_k=8):
    blocks = prob['blocks']
    m = len(prob['bays'])
    areas = [b['width'] * b['height'] for b in prob['bays']]
    u = [sum(areas) / m / a for a in areas]
    w = prob['weights']
    L = [0.0] * m
    for bid, a in asgn.items():
        L[a['bay_id']] += blocks[bid]['workload']
    wl = [u[j] * L[j] for j in range(m)]
    jmin = min(range(m), key=lambda j: wl[j])
    jmax = max(range(m), key=lambda j: wl[j])
    out = []
    tardy = sorted(((a['exit_time'] - blocks[bid]['due_date'], bid) for bid, a in asgn.items() if a['exit_time'] > blocks[bid]['due_date']), reverse=True)
    if tardy:
        rows = [[bid, jmin] for _, bid in tardy[:movers_k] if asgn[bid]['bay_id'] != jmin]
        if rows:
            out.append(rows)
    pen = sorted(((max(blocks[bid]['bay_preferences']) - blocks[bid]['bay_preferences'][a['bay_id']], bid, int(np.argmax(blocks[bid]['bay_preferences']))) for bid, a in asgn.items()), reverse=True)
    pen = [(p, bid, want) for p, bid, want in pen if p > 0]
    rows2 = [[b, jmin] for b in sorted((b for b, a in asgn.items() if a['bay_id'] == jmax), key=lambda b: -blocks[b]['workload'])[:movers_k]]
    rows3 = [[bid, want] for _, bid, want in pen[:movers_k]]
    pot2 = float(w['w2']) * (wl[jmax] - wl[jmin])
    pot3 = float(w['w3']) * sum((p for p, _, _ in pen[:movers_k]))
    ordered = sorted(((pot2, rows2), (pot3, rows3)), key=lambda t: -t[0])
    out.extend((rows for _, rows in ordered if rows))
    return out

def window_stage(prob_info, base_assign, base_obj, base_o1, budget_fn, chart=True, max_windows=4, debug=False, seed_rows=None):
    lib = _rust_lib()
    if lib is None:
        return None
    t0 = time.time()
    from constructor import _objective_of

    def left():
        return budget_fn()
    stats = {'windows': 0, 'accepted': 0, 'modes': [], 'b0': round(left(), 1)}
    cur = base_assign
    cur_fast, (_, _o2f, _o3f) = _objective_of(prob_info, base_assign)
    cur_o1 = base_o1
    improved_any = False
    queue = []
    if seed_rows:
        rows = [[int(b_), int(j_)] for b_, j_ in seed_rows if int(b_) in base_assign]
        if rows:
            queue.append(rows[:24])
            stats['modes'].append('slip')
    if chart and left() > 30.0:
        try:
            from plan_core import solve_seat_plan
            w = prob_info['weights']
            target, st = solve_seat_plan(prob_info, timelimit=6.0, incumbent=cur, workers=4)
            inc23 = float(w['w2']) * _o2f + float(w['w3']) * _o3f
            if target and st['obj23'] is not None and (st['obj23'] < inc23 - 1e-09):
                rows = [[bid, target[bid]] for bid in target if bid in cur and cur[bid]['bay_id'] != target[bid]]
                if rows:
                    queue.append(rows[:24])
                    stats['modes'].append('S')
        except Exception:
            pass
    while stats['windows'] < max_windows and left() > 24.0:
        if not queue:
            rows_list = _peak_rows(prob_info, cur)
            if not rows_list:
                break
            queue = rows_list
            stats['modes'].append('P')
        rows = queue.pop(0)
        stats['windows'] += 1
        try:
            new_asgn = _repair_once(lib, prob_info, cur, rows, left())
        except Exception:
            new_asgn = None
        if new_asgn is None:
            continue
        f_obj, (f_o1, _, _) = _objective_of(prob_info, new_asgn)
        if f_o1 <= cur_o1 + 1e-09 and f_obj < cur_fast - 1e-09:
            cur, cur_fast, cur_o1 = (new_asgn, f_obj, f_o1)
            improved_any = True
            stats['accepted'] += 1
            queue = []
    out = None
    if improved_any and left() > 2.0:
        sol, _ = retime(prob_info, cur, verbose=False)
        chk = check_feasibility(prob_info, sol)
        if chk.get('feasible') and chk['obj1'] <= base_o1 + 1e-09 and (chk['objective'] < base_obj - 1e-09):
            out = (sol, chk)
    if debug:
        print(f'[WINREP] b0={stats['b0']}s windows={stats['windows']} accepted={stats['accepted']} modes={stats['modes']} fast={cur_fast:,.0f} out={('hit' if out else 'none')} wall={time.time() - t0:.1f}s', flush=True)
    return out

def base_lateness(prob_info, base_assign):
    blocks = prob_info['blocks']
    return float(sum((max(0, a['exit_time'] - blocks[bid]['due_date']) for bid, a in base_assign.items())))
