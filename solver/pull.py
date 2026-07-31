import time
from constructor import construct, _objective_of
from gpr_core import PositiveCycleError, retime



def _sol_key(assign):
    return tuple(sorted(((b, a['bay_id'], a['orient_idx'], round(float(a['x']), 6), round(float(a['y']), 6), round(float(a['entry_time']), 6), round(float(a['exit_time']), 6)) for b, a in assign.items())))

def pull_forward(prob_info: dict, assignments: dict, timelimit: float=120.0, max_rounds: int=8, pref: bool=False, geo_cache: dict | None=None, pair_memo: dict | None=None, rel_memo: dict | None=None, try_cost: float | None=None, verbose: bool=False, alap_when_ontime: bool=False, tie_open: bool=False):
    t0 = time.time()
    if geo_cache is None:
        geo_cache = {}
    if pair_memo is None:
        pair_memo = {}
    if rel_memo is None:
        rel_memo = {}
    placed_cache: dict = {}
    blocks = prob_info['blocks']
    cur = {b: dict(a) for b, a in assignments.items()}
    cur_obj = _objective_of(prob_info, cur)[0]
    n_moved = n_tried = rounds = 0
    _tie_moved: set = set()
    n_drift = 0
    _tie_cap = 128 if tie_open else 0
    _tabu = {_sol_key(cur)} if tie_open else None

    def _sweep_retime(assign):
        if alap_when_ontime and _objective_of(prob_info, assign)[1][0] <= 1e-09:
            try:
                return retime(prob_info, assign, verbose=False, timing='latest')
            except (PositiveCycleError, ValueError, RuntimeError):
                pass
        return retime(prob_info, assign, verbose=False)

    def _admission_retime(assign):
        try:
            _, info = retime(prob_info, assign, verbose=False, timing='earliest')
        except (PositiveCycleError, ValueError, RuntimeError):
            return None
        return {b: dict(a) for b, a in info['assignments'].items()}
    _try_cap = max(15, int(timelimit / (try_cost if try_cost else 0.08 + len(blocks) / 600.0)))
    for rnd in range(max_rounds):
        rounds = rnd + 1
        _, info = _sweep_retime(cur)
        rt = info['assignments']
        rt_obj = _objective_of(prob_info, rt)[0]
        if rt_obj <= cur_obj:
            cur, cur_obj = ({b: dict(a) for b, a in rt.items()}, rt_obj)
        tardy = sorted(((cur[b]['exit_time'] - blocks[b]['due_date'], b) for b in cur if cur[b]['exit_time'] > blocks[b]['due_date']), reverse=True)
        targets = [b for _, b in tardy]
        if pref:
            seen = set(targets)
            pen = sorted(((max(blocks[b]['bay_preferences']) - blocks[b]['bay_preferences'][cur[b]['bay_id']], b) for b in cur if b not in seen), key=lambda t: (-t[0], t[1]))
            targets += [b for p, b in pen if p > 0]
        moved = 0
        drift_round = 0
        for b in targets:
            if n_tried >= _try_cap or time.time() - t0 > 2.0 * timelimit:
                break
            n_tried += 1
            fixed = {k: v for k, v in cur.items() if k != b}
            cand, _ = construct(prob_info, timelimit=25.0, fixed_assignments=fixed, insert_ids=[b], geo_cache=geo_cache, placed_cache=placed_cache, pair_memo=pair_memo, rel_memo=rel_memo, max_xs=32, max_ys=20, top_b=8, verbose=False)
            cand = _admission_retime(cand)
            if cand is None:
                continue
            o = _objective_of(prob_info, cand)[0]
            if o < cur_obj - 1e-09:
                cur, cur_obj = (cand, o)
                moved += 1
                if tie_open:
                    _tabu.add(_sol_key(cand))
            elif tie_open and n_drift < _tie_cap and (o <= cur_obj + 1e-06):
                _k = _sol_key(cand)
                if _k not in _tabu:
                    _tabu.add(_k)
                    cur, cur_obj = (cand, o)
                    n_drift += 1
                    drift_round += 1
        n_moved += moved
        if verbose:
            print(f'[Pull] round {rnd + 1}: moved={moved} tie={0} obj={cur_obj:,.0f}  {time.time() - t0:.0f}s', flush=True)
        if moved == 0 and drift_round == 0 and True:
            break
        if n_tried >= _try_cap or time.time() - t0 > 2.0 * timelimit:
            break
    _, info = _sweep_retime(cur)
    rt = info['assignments']
    if _objective_of(prob_info, rt)[0] <= cur_obj:
        cur = {b: dict(a) for b, a in rt.items()}
        cur_obj = _objective_of(prob_info, cur)[0]
    return (cur, {'rounds': rounds, 'moved': n_moved, 'tried': n_tried, 'drift': n_drift, 'obj': cur_obj, 'wall_s': round(time.time() - t0, 1)})

def seat_move(prob_info: dict, assignments: dict, target: dict, timelimit: float=30.0, geo_cache: dict | None=None, pair_memo: dict | None=None, rel_memo: dict | None=None, try_cost: float | None=None, drain_cost: float | None=None, verbose: bool=False):
    t0 = time.time()
    blocks = prob_info['blocks']
    bays = prob_info['bays']
    m = len(bays)
    w3 = float(prob_info.get('weights', {}).get('w3', 1.0))
    areas = [b['width'] * b['height'] for b in bays]
    from plan_core import _layer0_area
    a0 = [_layer0_area(b) for b in blocks]
    T = max((max(b['due_date'], b['release_time'] + b['processing_time']) for b in blocks)) + 1

    def _core(i):
        lo = blocks[i]['due_date'] - blocks[i]['processing_time']
        hi = blocks[i]['release_time'] + blocks[i]['processing_time']
        return (lo, hi) if lo < hi else None

    def _profile(cur):
        free = [[float(areas[j])] * T for j in range(m)]
        for i, a in cur.items():
            w = _core(i)
            if w:
                row = free[a['bay_id']]
                for t in range(max(0, w[0]), min(T, w[1])):
                    row[t] -= a0[i]
        return free

    def _margin(i, tb, free):
        R = blocks[i]['release_time']
        D = blocks[i]['due_date']
        P = blocks[i]['processing_time']
        row = free[tb]
        best = None
        for t in range(R, max(R, D - P) + 1):
            end = min(T, t + P)
            if t >= end:
                break
            mg = min(row[t:end]) - a0[i]
            if best is None or mg > best:
                best = mg
        return best if best is not None else -a0[i]
    cur = {b: dict(a) for b, a in assignments.items()}
    cur_obj = _objective_of(prob_info, cur)[0]
    n_mov = tried = accepted = escal = 0

    def _admission_retime(assign):
        try:
            _, info = retime(prob_info, assign, verbose=False, timing='earliest')
        except (PositiveCycleError, ValueError, RuntimeError):
            return None
        return {b: dict(a) for b, a in info['assignments'].items()}
    _tie_moved = set()
    _try_cap = max(10, int(timelimit / (try_cost if try_cost else 0.1 + len(blocks) / 400.0)))
    for pass_ in range(8):
        free = _profile(cur)
        cands = []
        for i, a in cur.items():
            tb = target.get(i, a['bay_id'])
            if tb == a['bay_id']:
                continue
            if pass_ == 0:
                n_mov += 1
            gain = w3 * (blocks[i]['bay_preferences'][tb] - blocks[i]['bay_preferences'][a['bay_id']])
            mg = _margin(i, tb, free)
            if mg <= 0.0:
                continue
            cands.append((gain * mg / areas[tb], i, tb))
        cands.sort(reverse=True)
        acc_before = accepted
        for _, i, tb in cands:
            if tried >= _try_cap or time.time() - t0 > 2.0 * timelimit:
                break
            tried += 1
            fixed = {k: dict(v) for k, v in cur.items() if k != i}
            cand, _ = construct(prob_info, timelimit=20.0, fixed_assignments=fixed, insert_ids=[i], plan={i: {'bay_id': tb}}, plan_bays=True, geo_cache=geo_cache, pair_memo=pair_memo, rel_memo=rel_memo, max_xs=32, max_ys=20, top_b=8, verbose=False)
            cand = _admission_retime(cand)
            o = _objective_of(prob_info, cand)[0] if cand is not None else 1e309
            if o >= cur_obj - 1e-09 and tried <= _try_cap // 2:
                escal += 1
                R_, D_ = (blocks[i]['release_time'], blocks[i]['due_date'])
                ov = {}
                for j, a2 in cur.items():
                    if j != i and a2['bay_id'] == tb and (a2['entry_time'] < D_) and (a2['exit_time'] > R_):
                        ov[j] = min(a2['exit_time'], D_) - max(a2['entry_time'], R_)
                blockers = sorted(ov, key=lambda j: (-ov[j], j))[:6]
                absent = {i} | set(blockers)
                fixed = {k: dict(v) for k, v in cur.items() if k not in absent}
                cand, _ = construct(prob_info, timelimit=20.0, fixed_assignments=fixed, insert_ids=[i], plan={i: {'bay_id': tb}}, plan_bays=True, geo_cache=geo_cache, pair_memo=pair_memo, rel_memo=rel_memo, max_xs=32, max_ys=20, top_b=8, verbose=False)
                if blockers:
                    cand, _ = construct(prob_info, timelimit=20.0, fixed_assignments=cand, insert_ids=blockers, geo_cache=geo_cache, pair_memo=pair_memo, rel_memo=rel_memo, max_xs=32, max_ys=20, top_b=8, verbose=False)
                cand = _admission_retime(cand)
                o = _objective_of(prob_info, cand)[0] if cand is not None else 1e309
                tardy = [k for k in [i] + blockers if cand[k]['exit_time'] > blocks[k]['due_date']] if cand is not None else []
                if tardy and True:
                    fixed2 = {k: dict(v) for k, v in cand.items() if k not in set(tardy)}
                    cand2, _ = construct(prob_info, timelimit=20.0, fixed_assignments=fixed2, insert_ids=tardy, geo_cache=geo_cache, pair_memo=pair_memo, rel_memo=rel_memo, max_xs=32, max_ys=20, top_b=8, verbose=False)
                    cand2 = _admission_retime(cand2)
                    if cand2 is not None and _objective_of(prob_info, cand2)[0] < o:
                        cand = cand2
                        o = _objective_of(prob_info, cand2)[0]
            if o < cur_obj - 1e-09:
                cur = {b: dict(a) for b, a in cand.items()}
                cur_obj = o
                accepted += 1
                free = _profile(cur)
                if verbose:
                    print(f'[Seat] pass {pass_ + 1} blk {i} -> bay {tb} ACC obj={cur_obj:,.0f}  {time.time() - t0:.0f}s', flush=True)
        if accepted == acc_before and True:
            break
    drain_tl = 0.35 * timelimit
    if verbose:
        print(f'[Seat] movers={n_mov} tried={tried} acc={accepted} esc={escal} obj={cur_obj:,.0f}  {time.time() - t0:.0f}s', flush=True)
    return (cur, {'movers': n_mov, 'tried': tried, 'accepted': accepted, 'tie_accepted': 0, 'escalated': escal, 'obj': cur_obj, 'wall_s': round(time.time() - t0, 1)})
