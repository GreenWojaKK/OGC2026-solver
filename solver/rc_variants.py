from __future__ import annotations
import math
import numpy as np
from plan_core import _fits_bay, _layer0_area

def _diverse_minima(costs: np.ndarray, starts: np.ndarray, count: int, separation: int) -> list[tuple[float, int]]:
    if costs.size == 0 or count <= 0:
        return []
    work = np.array(costs, dtype=float, copy=True)
    out: list[tuple[float, int]] = []
    for _ in range(count):
        k = int(np.argmin(work))
        value = float(work[k])
        if not math.isfinite(value):
            break
        out.append((value, int(starts[k])))
        lo = max(0, k - separation + 1)
        hi = min(len(work), k + separation)
        work[lo:hi] = np.inf
    return out

def reduced_cost_options(prob: dict, plan: dict, lam, price: dict | None=None, prox: dict | None=None, prox_w: float=0.0, top_k: int=4) -> dict[int, dict]:
    blocks, bays = (prob['blocks'], prob['bays'])
    weights = prob.get('weights', {})
    w1 = float(weights.get('w1', 1.0))
    w3 = float(weights.get('w3', 1.0))
    lam = np.asarray(lam, dtype=float)
    if lam.ndim != 2 or lam.shape[0] != len(bays):
        raise ValueError('lam must have shape (number_of_bays, horizon)')
    horizon = int(lam.shape[1])
    cl = np.concatenate([np.zeros((len(bays), 1)), np.cumsum(lam, axis=1)], axis=1)
    bay_area = np.array([float(b['width'] * b['height']) for b in bays], dtype=float)
    result: dict[int, dict] = {}
    per_bay = max(3, int(top_k))
    for i, block in enumerate(blocks):
        release = int(block['release_time'])
        due = int(block['due_date'])
        proc = int(block['processing_time'])
        last = horizon - proc
        if last < release:
            continue
        starts = np.arange(release, last + 1, dtype=np.int64)
        base = w1 * np.maximum(0, starts + proc - due).astype(float)
        if price:
            ui = float(price.get(i, 0.0))
            if ui:
                base = base + ui * starts
        if prox and prox_w > 0.0 and (i in prox):
            base = base + float(prox_w) * np.abs(starts - int(prox[i]))
        area = float(_layer0_area(block))
        prefs = block['bay_preferences']
        pref_max = float(max(prefs))
        fitting = [j for j, bay in enumerate(bays) if _fits_bay(block, bay)]
        merged: list[dict] = []
        separation = max(2, proc // 4)
        for j in fitting:
            cvec = base + w3 * (pref_max - float(prefs[j])) + area * (cl[j, starts + proc] - cl[j, starts])
            for value, start in _diverse_minima(cvec, starts, per_bay, separation):
                merged.append({'bay_id': int(j), 'start': int(start), 'cost': float(value)})
        merged.sort(key=lambda x: (x['cost'], x['start'], x['bay_id']))
        options = merged[:max(1, int(top_k))]
        if not options:
            continue
        base_row = plan.get(i, {})
        base_bay = int(base_row.get('bay_id', -1))
        base_start = int(base_row.get('start', release))
        base_cost = 1e309
        if base_bay in fitting and release <= base_start <= last:
            k = base_start - release
            base_cost = float(base[k]) + w3 * (pref_max - float(prefs[base_bay])) + area * (cl[base_bay, base_start + proc] - cl[base_bay, base_start])
        result[i] = {'base_cost': base_cost, 'best_cost': float(options[0]['cost']), 'options': options, 'fit_count': len(fitting), 'area_ratio': area / max((bay_area[j] for j in fitting), default=1.0)}
    return result

def discrepant_plan(prob: dict, plan: dict, ranked: dict[int, dict], slip: dict[int, int] | None, round_index: int, max_changes: int=1, pool_width: int=8, gap_fraction: float=0.25) -> tuple[dict, dict]:
    blocks = prob['blocks']
    w1 = max(float(prob.get('weights', {}).get('w1', 1.0)), 1e-12)
    slips = slip or {}
    variant = {}
    for i, row in plan.items():
        copied = dict(row)
        copied['bay_id'] = -1
        variant[i] = copied
    candidates: list[dict] = []
    for i, data in ranked.items():
        if i not in plan:
            continue
        block = blocks[i]
        proc = max(1, int(block['processing_time']))
        base_bay = int(plan[i]['bay_id'])
        base_start = int(plan[i]['start'])
        best_cost = float(data['best_cost'])
        base_cost = float(data['base_cost'])
        max_gap_days = max(2.0, gap_fraction * proc)
        lateness = max(0, base_start + proc - int(block['due_date']))
        prior_slip = max(0, int(slips.get(i, 0)))
        scarcity = 1.0 / max(1, int(data['fit_count']))
        for option_rank, option in enumerate(data['options']):
            bay = int(option['bay_id'])
            start = int(option['start'])
            if bay == base_bay and start == base_start:
                continue
            gap_days = max(0.0, (float(option['cost']) - best_cost) / w1)
            if gap_days > max_gap_days + 1e-12:
                continue
            delta_days = (float(option['cost']) - base_cost) / w1 if math.isfinite(base_cost) else 0.0
            displacement = abs(start - base_start) / proc
            priority = 6.0 * min(2.0, prior_slip / proc) + 2.0 * min(2.0, lateness / proc) + 1.5 * float(data['area_ratio']) + scarcity + 2.0 * min(2.0, max(0.0, -delta_days) / proc) + 0.15 * min(2.0, displacement) - gap_days / proc - 0.03 * option_rank
            candidates.append({'block_id': int(i), 'bay_id': bay, 'start': start, 'priority': float(priority), 'gap_days': float(gap_days), 'delta_days': float(delta_days), 'prior_slip': prior_slip, 'option_rank': option_rank})
    candidates.sort(key=lambda x: (-x['priority'], x['gap_days'], x['block_id'], x['option_rank'], x['start'], x['bay_id']))
    width = min(len(candidates), max(1, int(pool_width)))
    chosen: list[dict] = []
    used: set[int] = set()
    if width:
        offset = int(round_index) % width
        scan = candidates[offset:width] + candidates[:offset]
        for item in scan:
            if item['block_id'] in used:
                continue
            chosen.append(item)
            used.add(item['block_id'])
            if len(chosen) >= max(1, int(max_changes)):
                break
    for item in chosen:
        i = item['block_id']
        variant[i]['bay_id'] = item['bay_id']
        variant[i]['start'] = item['start']
    return (variant, {'changed': bool(chosen), 'chosen': chosen, 'candidate_count': len(candidates), 'pool_width': width})
