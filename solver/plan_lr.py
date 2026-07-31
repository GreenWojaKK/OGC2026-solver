import math
import time
import numpy as np
from plan_core import _layer0_area, _fits_bay
_MASK64 = (1 << 64) - 1
_ZOBRIST_SEEDS = (2611923443488327891, 1376283091369227076)

def _splitmix64(value):
    value = int(value) + 11400714819323198485 & _MASK64
    value = (value ^ value >> 30) * 13787848793156543929 & _MASK64
    value = (value ^ value >> 27) * 10723151780598845931 & _MASK64
    return (value ^ value >> 31) & _MASK64

def _schedule_zobrist_token(block_id, bay_id, start, lane=0):
    value = _splitmix64(int(block_id) ^ _ZOBRIST_SEEDS[int(lane)])
    value = _splitmix64(value ^ int(bay_id) & _MASK64)
    return _splitmix64(value ^ int(start) & _MASK64)

def _schedule_zobrist(plan):
    hashes = [0, 0]
    for block_id, row in plan.items():
        for lane in range(2):
            hashes[lane] ^= _schedule_zobrist_token(block_id, row['bay_id'], row['start'], lane)
    return tuple(hashes)

def solve_plan_lr(prob, alpha=0.72, timelimit: float=2.0, iters: int=300, price=None, prox=None, prox_w: float=0.0, gamma0: float=1.0, stall_iters: int=15, lam0=None, stats=None, rc_state=None, relative_dispatch: bool=False, blocked_weight: float=0.0, zobrist_dedup: bool=False):
    t0 = time.time()
    blocks, bays = (prob['blocks'], prob['bays'])
    w = prob.get('weights', {})
    w1, w3 = (float(w.get('w1', 1.0)), float(w.get('w3', 1.0)))
    n, nb = (len(blocks), len(bays))
    a = np.array([_layer0_area(b) for b in blocks])
    A = np.array([b['width'] * b['height'] for b in bays], dtype=float)
    cap = alpha * A
    R = np.array([b['release_time'] for b in blocks])
    D = np.array([b['due_date'] for b in blocks])
    P = np.array([b['processing_time'] for b in blocks])
    work = float(np.sum(a * P))
    H = int(R.max() + P.max() + math.ceil(work / cap.sum()) * 2 + 40)
    ok = [[j for j in range(nb) if _fits_bay(b, bays[j])] for b in blocks]
    prefpen = np.array([[max(b['bay_preferences']) - b['bay_preferences'][j] for j in range(nb)] for b in blocks], dtype=float)
    s_arrs, base_vecs = ([], [])
    for i in range(n):
        s_arr = np.arange(R[i], H - P[i] + 1)
        base = w1 * np.maximum(0, s_arr + P[i] - D[i]).astype(float)
        if price:
            ui = price.get(i, 0.0)
            if ui:
                base = base + ui * s_arr
        if prox and prox_w > 0 and (i in prox):
            base = base + prox_w * np.abs(s_arr - prox[i])
        s_arrs.append(s_arr)
        base_vecs.append(base)
    lam = np.array(lam0, dtype=float) if lam0 is not None else np.zeros((nb, H))
    if lam.shape != (nb, H):
        lam = np.zeros((nb, H))
    blocked_weight = max(0.0, float(blocked_weight))
    collect_blocked = blocked_weight > 0.0

    def repair(order, target_bays=None, target_starts=None):
        used = np.zeros((nb, H))
        bad_prefix = np.empty(H + 1, dtype=np.int32)
        blocked_pressure = np.zeros((nb, H)) if collect_blocked else None
        plan = {}
        priced = true = 0.0
        for i in order:
            best = None
            for j in ok[i]:
                if a[i] > cap[j]:
                    continue
                r = int(R[i])
                p = int(P[i])
                nday = H - r
                blocked = used[j, r:] + a[i] > cap[j] + 1e-09
                if collect_blocked and j == int(target_bays[i]):
                    target = int(target_starts[i])
                    offset = target - r
                    target_blocked = blocked[offset:offset + p]
                    if target_blocked.size == p and np.any(target_blocked):
                        pressure = np.maximum(0.0, used[j, target:target + p] + a[i] - cap[j])
                        blocked_pressure[j, target:target + p] += pressure
                bad_prefix[0] = 0
                np.cumsum(blocked, dtype=np.int32, out=bad_prefix[1:nday + 1])
                free = np.nonzero(bad_prefix[p:nday + 1] == bad_prefix[:nday + 1 - p])[0]
                if free.size == 0:
                    continue
                vals = base_vecs[i][free] + w3 * prefpen[i, j]
                q = int(np.argmin(vals))
                t = int(s_arrs[i][free[q]])
                key = (float(vals[q]), t, float(prefpen[i, j]), j)
                if best is None or key < best[0]:
                    best = (key, j, t)
            if best is None:
                j = int(np.argmax(cap))
                t = int(R[i])
            else:
                _, j, t = best
            used[j, t:t + P[i]] += a[i]
            plan[i] = {'bay_id': int(j), 'start': int(t)}
            tard = max(0, t + P[i] - D[i])
            c = w1 * tard + w3 * prefpen[i, j]
            true += c
            priced += c
            if price:
                priced += price.get(i, 0.0) * t
            if prox and prox_w > 0 and (i in prox):
                priced += prox_w * abs(t - prox[i])
        return (plan, priced, true, blocked_pressure)
    best_plan, best_priced, best_true = (None, 1e309, 1e309)
    best_lam = None
    dual_lam = None
    LB_best = -1e309
    gamma = gamma0
    stall = 0
    it = 0
    seen_schedule_hashes = set()
    blocked_unique_states = 0
    blocked_revisit_states = 0
    blocked_signal_days = 0
    blocked_signal_area = 0.0
    for it in range(1, iters + 1):
        cl = np.concatenate([np.zeros((nb, 1)), np.cumsum(lam, axis=1)], axis=1)
        choice_j = np.empty(n, dtype=np.int64)
        choice_s = np.empty(n, dtype=np.int64)
        sub = 0.0
        for i in range(n):
            s_arr, base = (s_arrs[i], base_vecs[i])
            bc, bj, bs = (1e309, ok[i][0], int(R[i]))
            for j in ok[i]:
                cvec = base + prefpen[i, j] * w3 + a[i] * (cl[j, s_arr + P[i]] - cl[j, s_arr])
                k = int(np.argmin(cvec))
                if cvec[k] < bc:
                    bc, bj, bs = (float(cvec[k]), j, int(s_arr[k]))
            choice_j[i], choice_s[i] = (bj, bs)
            sub += bc
        LB = sub - float(np.dot(lam.sum(axis=1), cap))
        occ = np.zeros((nb, H))
        for i in range(n):
            occ[choice_j[i], choice_s[i]:choice_s[i] + P[i]] += a[i]
        g = occ - cap[:, None]
        if relative_dispatch:
            order = sorted(range(n), key=lambda i: (choice_s[i] - R[i], choice_s[i], -a[i], D[i]))
        else:
            order = sorted(range(n), key=lambda i: (choice_s[i], -a[i], D[i]))
        plan, priced, true, blocked_pressure = repair(order, target_bays=choice_j if collect_blocked else None, target_starts=choice_s if collect_blocked else None)
        if priced < best_priced:
            best_plan, best_priced, best_true = (plan, priced, true)
            if rc_state is not None:
                best_lam = np.array(lam, copy=True)
        if LB > LB_best + 1e-06:
            LB_best = LB
            if rc_state is not None:
                dual_lam = np.array(lam, copy=True)
            stall = 0
        else:
            stall += 1
            if stall >= stall_iters:
                gamma *= 0.5
                stall = 0
        update_signal = g
        if collect_blocked:
            use_blocked = True
            if zobrist_dedup:
                schedule_hash = _schedule_zobrist(plan)
                if schedule_hash in seen_schedule_hashes:
                    blocked_revisit_states += 1
                    use_blocked = False
                else:
                    seen_schedule_hashes.add(schedule_hash)
            if use_blocked:
                blocked_unique_states += 1
                blocked_signal_days += int(np.count_nonzero(blocked_pressure))
                blocked_signal_area += float(blocked_pressure.sum())
                update_signal = g + blocked_weight * blocked_pressure
        gn2 = float(np.sum(update_signal * update_signal))
        if gn2 < 1e-12 or gamma < 0.001 * gamma0:
            break
        step = gamma * max(best_priced - LB, 1e-06) / gn2
        lam = np.maximum(0.0, lam + step * update_signal)
        if time.time() - t0 > timelimit:
            break
    if stats is not None:
        stats.update(iters=it, LB=round(LB_best, 1), UB_priced=round(best_priced, 1), obj_true=round(best_true, 1), gap=round((best_priced - LB_best) / max(abs(best_priced), 1e-09), 4), wall=round(time.time() - t0, 2), dispatch='wait_from_release' if relative_dispatch else 'absolute_start', blocked_weight=float(blocked_weight), blocked_unique_states=int(blocked_unique_states), blocked_revisit_states=int(blocked_revisit_states), blocked_signal_days=int(blocked_signal_days), blocked_signal_area=float(blocked_signal_area), zobrist_dedup=bool(zobrist_dedup))
    if rc_state is not None:
        rc_state.clear()
        rc_state.update(lam=best_lam if best_lam is not None else np.array(lam, copy=True), dual_lam=dual_lam if dual_lam is not None else np.array(lam, copy=True), dual_lb=float(LB_best), cap=np.array(cap, copy=True), best_priced=float(best_priced), best_true=float(best_true), horizon=int(H), iters=int(it))
    return (best_plan, lam)
