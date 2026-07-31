from __future__ import annotations
import math
import random
import time
from constructor import construct, _objective_of, _Geo
from gpr_core import retime

def _pick_tardy(prob_info, cur, rng):
    blocks_data = prob_info['blocks']
    tardy = [(max(0, a['exit_time'] - blocks_data[b]['due_date']), b) for b, a in cur.items()]
    tardy = [(t, b) for t, b in tardy if t > 0]
    if not tardy:
        return None
    tot = sum((t for t, _ in tardy))
    r = rng.random() * tot
    acc = 0.0
    for t, b in tardy:
        acc += t
        if acc >= r:
            return b
    return tardy[-1][1]

def _op_tardy(prob_info, cur, rng, k):
    pick = _pick_tardy(prob_info, cur, rng)
    if pick is None:
        return _op_random(prob_info, cur, rng, k)
    a = cur[pick]
    S_p, F_p = (a['entry_time'], a['exit_time'])
    same = [(abs(cur[b]['x'] - a['x']), b) for b in cur if b != pick and cur[b]['bay_id'] == a['bay_id'] and (cur[b]['entry_time'] < F_p) and (S_p < cur[b]['exit_time'])]
    same.sort()
    return [pick] + [b for _, b in same[:k - 1]]

def _op_slab(prob_info, cur, rng, cap=45):
    pick = _pick_tardy(prob_info, cur, rng)
    if pick is None:
        return _op_random(prob_info, cur, rng, 12)
    a = cur[pick]
    t_star = (a['entry_time'] + a['exit_time']) // 2
    bay = a['bay_id']
    sel = [b for b in cur if cur[b]['bay_id'] == bay and cur[b]['entry_time'] <= t_star < cur[b]['exit_time']]
    if len(sel) > cap:
        rng.shuffle(sel)
        sel = sel[:cap]
    return sel if sel else [pick]

def _op_random(prob_info, cur, rng, k):
    bids = list(cur.keys())
    return rng.sample(bids, min(k, len(bids)))

def _op_blocker(prob_info, cur, rng, geo_cache, k):
    pick = _pick_tardy(prob_info, cur, rng)
    if pick is None:
        return _op_random(prob_info, cur, rng, k)
    blocks_data = prob_info['blocks']
    a = cur[pick]
    g = geo_cache.get((pick, a['orient_idx']))
    if g is None:
        g = geo_cache[pick, a['orient_idx']] = _Geo(blocks_data[pick], a['orient_idx'])
    lx0, ly0, lx1, ly1 = g.bbox
    bb = (a['x'] + lx0, a['y'] + ly0, a['x'] + lx1, a['y'] + ly1)
    R_b = blocks_data[pick]['release_time']
    W_end = R_b + blocks_data[pick]['processing_time']
    E = []
    for b2, a2 in cur.items():
        if b2 == pick or a2['bay_id'] != a['bay_id']:
            continue
        if not (a2['entry_time'] < W_end and R_b < a2['exit_time']):
            continue
        g2 = geo_cache.get((b2, a2['orient_idx']))
        if g2 is None:
            g2 = geo_cache[b2, a2['orient_idx']] = _Geo(blocks_data[b2], a2['orient_idx'])
        kx0, ky0, kx1, ky1 = g2.bbox
        if a2['x'] + kx0 < bb[2] and bb[0] < a2['x'] + kx1 and (a2['y'] + ky0 < bb[3]) and (bb[1] < a2['y'] + ky1):
            E.append((a2['entry_time'], b2))
    if not E:
        return _op_tardy(prob_info, cur, rng, k)
    E.sort()
    return [pick] + [b2 for _, b2 in E[:k - 1]]

def lns_sa(prob_info: dict, assignments: dict, timelimit: float, geo_cache: dict | None=None, seed: int=0, k: int=8, milp_stall: int=25, milp_k: int=20, milp_time: float=0.0, compact_every: int=20, workers: int=8, pair_memo: dict | None=None, rel_memo: dict | None=None, blocker_op: bool=False, max_iters: int | None=None, bay_prior: list | None=None, bay_prior_strength: float=0.0, improvement_cb=None, verbose: bool=True):
    if milp_time > 0:
        from repair_milp import milp_repair
    t0 = time.time()
    rng = random.Random(seed)
    prior_strength = max(0.0, float(bay_prior_strength))
    prior_active = bool(bay_prior is not None and prior_strength > 0.0)
    if prior_active:
        n = len(prob_info.get('blocks', []))
        m = len(prob_info.get('bays', []))
        if len(bay_prior) != n or any((len(row) != m for row in bay_prior)):
            raise ValueError('bay_prior must have one value per block and bay')
    if geo_cache is None:
        geo_cache = {}
    placed_cache: dict = {}
    if pair_memo is None:
        pair_memo = {}
    if rel_memo is None:
        rel_memo = {}

    def compact(assign):
        try:
            _, info = retime(prob_info, assign, verbose=False)
            return info['assignments']
        except Exception:
            return assign
    cur = compact({b: dict(a) for b, a in assignments.items()})
    cur_obj, _ = _objective_of(prob_info, cur)
    best, best_obj = (cur, cur_obj)
    OPS = {'tardy_small': lambda: _op_tardy(prob_info, cur, rng, rng.randint(4, 10)), 'tardy_big': lambda: _op_tardy(prob_info, cur, rng, rng.randint(14, 28)), 'slab': lambda: _op_slab(prob_info, cur, rng), 'random': lambda: _op_random(prob_info, cur, rng, rng.randint(5, 14))}
    wts = {'tardy_small': 3.0, 'tardy_big': 1.0, 'slab': 1.0, 'random': 1.0}
    if blocker_op:
        OPS['blocker'] = lambda: _op_blocker(prob_info, cur, rng, geo_cache, rng.randint(3, 8))
        wts['blocker'] = 1.0
    op_stats = {op: [0, 0] for op in OPS}
    w1 = prob_info.get('weights', {}).get('w1', 1.0)
    T0 = max(1.0, 1.0 * w1)
    it = n_acc = n_imp = n_milp = n_milp_imp = n_compact = 0
    aco_reinserted = aco_bay_changes = 0
    aco_elite_bay_selections = aco_changes_to_elite_bay = 0
    stall = 0

    def publish_improvement(event):
        if improvement_cb is None:
            return
        try:
            improvement_cb(best, float(best_obj), str(event))
        except Exception:
            pass
    while True:
        elapsed = time.time() - t0
        if max_iters is not None and it >= max_iters:
            break
        if elapsed > timelimit - 1.0:
            break
        it += 1
        names = list(OPS)
        op = rng.choices(names, weights=[wts[o] for o in names])[0]
        removed = OPS[op]()
        op_stats[op][0] += 1
        fixed = {b: a for b, a in cur.items() if b not in set(removed)}
        prior_kw = {}
        if prior_active:
            prior_kw = {'bay_prior': bay_prior, 'bay_prior_strength': prior_strength}
        new_assign, _ = construct(prob_info, timelimit=10.0 if max_iters is not None else max(2.0, timelimit - elapsed), fixed_assignments=fixed, insert_ids=removed, insert_order='given' if op == 'blocker' else 'edd', geo_cache=geo_cache, placed_cache=placed_cache, pair_memo=pair_memo, rel_memo=rel_memo, max_xs=12, max_ys=8, top_b=4, screen='exact', verbose=False, **prior_kw)
        if prior_active:
            for bid in removed:
                before = int(cur[bid]['bay_id'])
                after = int(new_assign[bid]['bay_id'])
                row = bay_prior[bid]
                low = min(row)
                elite = {bay for bay, value in enumerate(row) if value <= low + 1e-12}
                aco_reinserted += 1
                aco_bay_changes += int(after != before)
                aco_elite_bay_selections += int(after in elite)
                aco_changes_to_elite_bay += int(after != before and after in elite and (before not in elite))
        new_assign = compact(new_assign)
        new_obj, _ = _objective_of(prob_info, new_assign)
        delta = new_obj - cur_obj
        frac = it / max_iters if max_iters is not None else elapsed / timelimit
        T = T0 * (1.0 / 100.0) ** frac
        accept = delta <= 0 or rng.random() < math.exp(-delta / T)
        reward = 0.0
        if accept:
            cur, cur_obj = (new_assign, new_obj)
            n_acc += 1
            reward = 0.3
            if delta < 0:
                reward = 1.0
            if cur_obj < best_obj - 1e-09:
                best, best_obj = (cur, cur_obj)
                n_imp += 1
                op_stats[op][1] += 1
                reward = 3.0
                stall = -1
                publish_improvement(op)
                if verbose:
                    print(f'[LNS] it={it:4d} t={elapsed:5.1f}s obj={best_obj:>14,.0f}  [{op}]')
        wts[op] = 0.9 * wts[op] + 0.1 * (1.0 + reward)
        stall += 1
        if it % compact_every == 0:
            cur2 = compact(cur)
            obj2, _ = _objective_of(prob_info, cur2)
            if obj2 < cur_obj - 1e-09:
                n_compact += 1
                cur, cur_obj = (cur2, obj2)
                if cur_obj < best_obj - 1e-09:
                    best, best_obj = (cur, cur_obj)
                    n_imp += 1
                    publish_improvement('compact')
                    if verbose:
                        print(f'[LNS] it={it:4d} t={elapsed:5.1f}s obj={best_obj:>14,.0f}  [compact]')
        remaining = timelimit - (time.time() - t0)
        if milp_time > 0 and stall >= milp_stall and (remaining > milp_time + 8.0):
            removed = _op_tardy(prob_info, cur, rng, milp_k)
            new_assign, minfo = milp_repair(prob_info, cur, removed, geo_cache, timelimit=min(milp_time, remaining - 5.0), workers=workers, tag=f'lns_s{seed}')
            n_milp += 1
            if verbose:
                print(f'[LNS]      milp#{n_milp} {minfo.get('status', '?'):9s} k={len(removed)} prep={minfo.get('prep_s')}s solve={minfo.get('solver_wall')}s')
            if new_assign is not None:
                new_assign = compact(new_assign)
                new_obj, _ = _objective_of(prob_info, new_assign)
                if new_obj < cur_obj - 1e-09:
                    cur, cur_obj = (new_assign, new_obj)
                    if cur_obj < best_obj - 1e-09:
                        best, best_obj = (cur, cur_obj)
                        n_milp_imp += 1
                        publish_improvement('milp')
            stall = 0
    stats = {'iters': it, 'accepted': n_acc, 'improved': n_imp, 'compactions': n_compact, 'milp_calls': n_milp, 'milp_improved': n_milp_imp, 'obj': best_obj, 'aco_active': bool(prior_active), 'aco_strength': float(prior_strength), 'aco_reinserted': int(aco_reinserted), 'aco_bay_changes': int(aco_bay_changes), 'aco_elite_bay_selections': int(aco_elite_bay_selections), 'aco_changes_to_elite_bay': int(aco_changes_to_elite_bay), 'op_stats': {o: tuple(v) for o, v in op_stats.items()}, 'wall_s': round(time.time() - t0, 1)}
    if verbose:
        ops_str = ' '.join((f'{o}:{v[0]}/{v[1]}' for o, v in op_stats.items()))
        print(f'[LNS] done: iters={it} acc={n_acc} imp={n_imp} compact={n_compact} best={best_obj:,.0f}  [{ops_str}]')
    return (best, stats)
