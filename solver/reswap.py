import math
import sys
import time
from collections import deque
from constructor import construct, _objective_of
from gpr_core import PositiveCycleError, retime
_RW_PROB = None
_RW_SCAFFOLD = None
_RW_W1 = 1.0
_RW_CUR_OBJ = 1e309
_RW_QUALITY_LIMIT = 1e309
_RW_NONSTRICT = False
_RW_GEO_CACHE = None
_RW_PLACED_CACHE = None
_RW_PAIR_MEMO = None
_RW_REL_MEMO = None

def _reswap_worker_init(prob_info, scaffold, w1, cur_obj, quality_limit=None, nonstrict=False):
    global _RW_PROB, _RW_SCAFFOLD, _RW_W1, _RW_CUR_OBJ
    global _RW_QUALITY_LIMIT, _RW_NONSTRICT
    global _RW_GEO_CACHE, _RW_PLACED_CACHE, _RW_PAIR_MEMO, _RW_REL_MEMO
    _RW_PROB = prob_info
    _RW_SCAFFOLD = scaffold
    _RW_W1 = w1
    _RW_CUR_OBJ = cur_obj
    _RW_QUALITY_LIMIT = float(quality_limit) if quality_limit is not None else float(cur_obj)
    _RW_NONSTRICT = bool(nonstrict)
    _RW_GEO_CACHE = {}
    _RW_PLACED_CACHE = {}
    _RW_PAIR_MEMO = {}
    _RW_REL_MEMO = {}
    try:
        import numba
        numba.set_num_threads(1)
    except Exception:
        pass

def _reswap_worker(task):
    seq, members, path = task
    move_of = dict(members)
    fixed = {b: dict(a) for b, a in _RW_SCAFFOLD.items() if b not in move_of}
    plan = {bid: {'bay_id': bay, 'start': int(_RW_SCAFFOLD[bid]['entry_time'])} for bid, bay in members}
    order = []
    for move in reversed(path):
        if move[0] not in order:
            order.append(move[0])
    try:
        cand, _ = construct(_RW_PROB, timelimit=3.0, fixed_assignments=fixed, insert_ids=order, insert_order='given', plan=plan, plan_bays=True, plan_target_w=0.05 * _RW_W1, screen='exact', max_xs=24, max_ys=16, top_b=8, geo_cache=_RW_GEO_CACHE, placed_cache=_RW_PLACED_CACHE, pair_memo=_RW_PAIR_MEMO, rel_memo=_RW_REL_MEMO, verbose=False)
        _, info = retime(_RW_PROB, cand, verbose=False)
        realized = {b: dict(a) for b, a in info['assignments'].items()}
        obj = _objective_of(_RW_PROB, realized)[0]
        admissible = obj <= _RW_QUALITY_LIMIT + 1e-09 if _RW_NONSTRICT else obj < _RW_CUR_OBJ - 1e-09
        return (seq, True, obj, realized if admissible else None)
    except Exception:
        return (seq, False, 1e309, None)

def _fits(block: dict, bay: dict) -> bool:
    w, h = (int(bay['width']), int(bay['height']))
    for orient in block['shape']:
        pts = [p for layer in orient['layers'] for p in layer]
        if not pts:
            continue
        xs = [float(p[0]) for p in pts]
        ys = [float(p[1]) for p in pts]
        if max(xs) - min(xs) <= w + 1e-09 and max(ys) - min(ys) <= h + 1e-09:
            return True
    return False

def _bay_move_scorer(prob_info: dict, assignments: dict):
    blocks = prob_info['blocks']
    bays = prob_info['bays']
    m = len(bays)
    weights = prob_info.get('weights', {})
    w1 = weights.get('w1', 1.0)
    w2 = weights.get('w2', 1.0)
    w3 = weights.get('w3', 1.0)
    obj1 = 0.0
    obj3 = 0.0
    loads = [0.0] * m
    old_bay = {}
    workloads = {}
    pref_pen = {}
    for bid, a in assignments.items():
        block = blocks[bid]
        bay = int(a['bay_id'])
        old_bay[bid] = bay
        workloads[bid] = block['workload']
        penalties = [max(block['bay_preferences']) - p for p in block['bay_preferences']]
        pref_pen[bid] = penalties
        obj1 += max(0.0, a['exit_time'] - block['due_date'])
        loads[bay] += block['workload']
        obj3 += penalties[bay]
    if m > 1:
        areas = [bay['width'] * bay['height'] for bay in bays]
        avg = sum(areas) / m
        scale = [avg / area for area in areas]
    else:
        scale = [1.0]
    fixed = w1 * obj1

    def score(moves):
        moved_loads = loads.copy()
        moved_obj3 = obj3
        for bid, bay in moves:
            source = old_bay[bid]
            bay = int(bay)
            workload = workloads[bid]
            moved_loads[source] -= workload
            moved_loads[bay] += workload
            moved_obj3 += pref_pen[bid][bay] - pref_pen[bid][source]
        if m > 1:
            obj2 = float(math.floor(max((abs(scale[i] * moved_loads[i] - scale[j] * moved_loads[j]) for i in range(m) for j in range(m) if i != j))))
        else:
            obj2 = 0.0
        return fixed + w2 * obj2 + w3 * moved_obj3
    return score

def _cohorts(seed, support_fn):
    queue = deque()
    seen_paths = {(seed,)}
    queue.append((frozenset((seed,)), (seed,), 0, seed))
    yield ((seed,), (seed,), 0)
    while queue:
        members, path, depth, focus = queue.popleft()
        if depth >= 2 or len(members) >= 3:
            continue
        eligible, seen = ([], set())
        for sup in support_fn(focus, members):
            if sup in members or sup in seen:
                continue
            seen.add(sup)
            eligible.append(sup)
            if len(eligible) >= 6:
                break
        for sup in eligible:
            cpath = path + (sup,)
            if cpath in seen_paths:
                continue
            seen_paths.add(cpath)
            cmembers = members | {sup}
            queue.append((cmembers, cpath, depth + 1, sup))
            yield (tuple(sorted(cmembers)), cpath, depth + 1)

def reswap(prob_info: dict, assignments: dict, timelimit: float=60.0, geo_cache: dict | None=None, pair_memo: dict | None=None, rel_memo: dict | None=None, verbose: bool=False, parallel: bool=False, quality_gap: float=0.0, return_terminal: bool=False, improvement_cb=None):
    t0 = time.time()
    if geo_cache is None:
        geo_cache = {}
    if pair_memo is None:
        pair_memo = {}
    if rel_memo is None:
        rel_memo = {}
    placed_cache: dict = {}
    blocks = prob_info['blocks']
    bays = prob_info['bays']
    w1 = prob_info.get('weights', {}).get('w1', 1.0)
    par_on = parallel and sys.platform.startswith('linux') and True
    fit_bays = {bid: [j for j, bay in enumerate(bays) if _fits(blocks[bid], bay)] for bid in assignments}
    cur = {b: dict(a) for b, a in assignments.items()}
    cur_obj = _objective_of(prob_info, cur)[0]
    protected = {b: dict(a) for b, a in cur.items()}
    protected_obj = cur_obj
    gap = max(0.0, float(quality_gap))
    nonstrict = gap > 0.0

    def _fingerprint(assign):
        return tuple(((int(bid), int(a['bay_id']), int(a['x']), int(a['y']), int(a.get('orient_idx', 0)), int(a['entry_time']), int(a['exit_time'])) for bid, a in sorted(assign.items())))
    visited = {_fingerprint(cur)}
    stats = {'rounds': 0, 'proposals': 0, 'realize_fail': 0, 'improved_rounds': 0, 'parallel_used': False, 'parallel_failed': False, 'serial_realized': False, 'quality_gap': gap, 'nonstrict': nonstrict, 'accepted_rounds': 0, 'worsening_rounds': 0, 'lateral_rounds': 0}

    def _targets(bid):
        here = int(cur[bid]['bay_id'])
        return [j for j in fit_bays[bid] if j != here]
    for _ in range(4):
        if time.time() - t0 > timelimit - 3.0:
            break
        stats['rounds'] += 1
        _plan_score = _bay_move_scorer(prob_info, cur)
        scaffold = cur
        if _objective_of(prob_info, cur)[1][0] <= 1e-09:
            try:
                _, _ri = retime(prob_info, cur, verbose=False, timing='latest')
                scaffold = {b: dict(a) for b, a in _ri['assignments'].items()}
            except (PositiveCycleError, ValueError, RuntimeError):
                scaffold = cur
            finally:
                pass
        seeds = []
        for bid in sorted(cur):
            pref = blocks[bid]['bay_preferences']
            for bay in sorted(_targets(bid), key=lambda j: (-pref[j], j))[:2]:
                seeds.append((_plan_score(((bid, bay),)), bid, bay))
        seeds.sort()
        anchors = [(bid, bay) for _, bid, bay in seeds[:8]]
        single_cache: dict = {}

        def _single(move):
            v = single_cache.get(move)
            if v is None:
                v = _plan_score((move,))
                single_cache[move] = v
            return v

        def _support_fn(focus, members):
            planned = {m[0] for m in members}
            current = tuple(sorted(members))
            current_score = _plan_score(current)
            ranked = []
            lo = int(scaffold[focus[0]]['entry_time'])
            hi = int(scaffold[focus[0]]['exit_time'])
            for res in sorted(scaffold):
                if res in planned:
                    continue
                if int(scaffold[res]['bay_id']) != focus[1]:
                    continue
                ov = min(hi, int(scaffold[res]['exit_time'])) - max(lo, int(scaffold[res]['entry_time']))
                if ov <= 0:
                    continue
                best = None
                for bay in _targets(res):
                    if bay == focus[1]:
                        continue
                    mv = (res, bay)
                    sc = _plan_score(current + (mv,))
                    if best is None or (sc, mv) < best:
                        best = (sc, mv)
                if best is None:
                    continue
                ranked.append(((0, best[0], -ov, best[1]), best[1]))
            base = cur_obj
            for bid in sorted(cur):
                if bid in planned:
                    continue
                for bay in _targets(bid):
                    mv = (bid, bay)
                    combined = _plan_score(current + (mv,))
                    synergy = combined - current_score - (_single(mv) - base)
                    ranked.append(((1, synergy, combined, mv), mv))
            ranked.sort(key=lambda r: r[0])
            return [mv for _, mv in ranked]
        quality_limit = protected_obj * (1.0 + gap)
        best_prop = None
        best_obj = 1e309 if nonstrict else cur_obj
        best_seq = None
        n_prop = 0
        seen_plans: set = set()
        pool = None
        if par_on:
            try:
                import multiprocessing as mp
                ctx = mp.get_context('fork')
                pool = ctx.Pool(4, initializer=_reswap_worker_init, initargs=(prob_info, scaffold, w1, cur_obj, quality_limit, nonstrict))
                stats['parallel_used'] = True
            except Exception:
                pool = None
            finally:
                pass
        seq = 0
        try:
            for anchor in anchors:
                emitted = 0
                if n_prop >= 64 or time.time() - t0 > timelimit - 3.0:
                    break
                proposals = iter(_cohorts(anchor, _support_fn))
                exhausted = False
                while emitted < 8 and n_prop < 64 and (time.time() - t0 <= timelimit - 3.0) and (not exhausted):
                    batch = []
                    batch_cap = min(8 - emitted, 64 - n_prop)
                    if pool is None:
                        batch_cap = 1
                    while len(batch) < batch_cap:
                        try:
                            members, path, _depth = next(proposals)
                        except StopIteration:
                            exhausted = True
                            break
                        finally:
                            pass
                        if members in seen_plans:
                            continue
                        seen_plans.add(members)
                        batch.append((seq, members, path))
                        seq += 1
                    if not batch:
                        continue
                    results = None
                    if pool is not None:
                        try:
                            wait_s = max(4.0, min(15.0, timelimit - (time.time() - t0)))
                            results = pool.map_async(_reswap_worker, batch).get(timeout=wait_s)
                        except Exception:
                            pool.terminate()
                            pool.join()
                            pool = None
                            stats['parallel_failed'] = True
                        finally:
                            pass
                    if results is None:
                        stats['serial_realized'] = True
                        results = []
                        for sq, members, path in batch:
                            move_of = dict(members)
                            fixed = {b: dict(a) for b, a in scaffold.items() if b not in move_of}
                            plan = {bid: {'bay_id': bay, 'start': int(scaffold[bid]['entry_time'])} for bid, bay in members}
                            order = []
                            for move in reversed(path):
                                if move[0] not in order:
                                    order.append(move[0])
                            try:
                                cand, _ = construct(prob_info, timelimit=3.0, fixed_assignments=fixed, insert_ids=order, insert_order='given', plan=plan, plan_bays=True, plan_target_w=0.05 * w1, screen='exact', max_xs=24, max_ys=16, top_b=8, geo_cache=geo_cache, placed_cache=placed_cache, pair_memo=pair_memo, rel_memo=rel_memo, verbose=False)
                                _, info = retime(prob_info, cand, verbose=False)
                                realized = {b: dict(a) for b, a in info['assignments'].items()}
                                obj = _objective_of(prob_info, realized)[0]
                                admissible = obj <= quality_limit + 1e-09 if nonstrict else obj < cur_obj - 1e-09
                                results.append((sq, True, obj, realized if admissible else None))
                            except Exception:
                                results.append((sq, False, 1e309, None))
                            finally:
                                pass
                    for result_seq, ok, obj, realized in results:
                        if not ok:
                            stats['realize_fail'] += 1
                            continue
                        n_prop += 1
                        emitted += 1
                        if realized is None:
                            continue
                        if nonstrict and _fingerprint(realized) in visited:
                            continue
                        if obj < best_obj - 1e-09 or (abs(obj - best_obj) <= 1e-09 and (best_seq is None or result_seq < best_seq)):
                            best_prop, best_obj = (realized, obj)
                            best_seq = result_seq
        finally:
            if pool is not None:
                pool.close()
                pool.join()
        stats['proposals'] += n_prop
        if best_prop is None:
            break
        previous_obj = cur_obj
        cur, cur_obj = (best_prop, best_obj)
        visited.add(_fingerprint(cur))
        stats['accepted_rounds'] += 1
        stats['worsening_rounds'] += int(cur_obj > previous_obj + 1e-09)
        stats['lateral_rounds'] += int(abs(cur_obj - previous_obj) <= 1e-09)
        if cur_obj < protected_obj - 1e-09:
            protected = {b: dict(a) for b, a in cur.items()}
            protected_obj = cur_obj
            stats['improved_rounds'] += 1
            if improvement_cb is not None:
                try:
                    improvement_cb(protected, float(protected_obj), 'reswap')
                except Exception:
                    pass
                finally:
                    pass
        if verbose:
            print(f'[Reswap] round {stats['rounds']}: obj={cur_obj:,.0f} props={n_prop}  {time.time() - t0:.0f}s', flush=True)
    stats['obj'] = protected_obj
    stats['terminal_obj'] = cur_obj
    stats['wall_s'] = round(time.time() - t0, 1)
    if return_terminal:
        return (protected, stats, cur)
    return (protected, stats)
