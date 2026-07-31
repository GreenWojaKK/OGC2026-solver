from __future__ import annotations
from collections import defaultdict
import math
import time
from constructor import construct, _objective_of
from gpr_core import PositiveCycleError, assignments_from_solution, retime
from plan_core import _layer0_area
from utils import check_feasibility

def _copy_assignment(assignments):
    return {int(bid): dict(row) for bid, row in assignments.items()}

def _fingerprint(assignments):
    return tuple(((int(bid), int(row['bay_id']), int(row['x']), int(row['y']), int(row.get('orient_idx', 0)), int(row['entry_time']), int(row['exit_time'])) for bid, row in sorted(assignments.items())))

def _bay_vector(assignments):
    return tuple((int(assignments[bid]['bay_id']) for bid in sorted(assignments)))

def _normalize_plan(plan_targets):
    normalized = {}
    for raw_bid, raw in (plan_targets or {}).items():
        try:
            bid = int(raw_bid)
            if not isinstance(raw, dict) or 'start' not in raw:
                continue
            normalized[bid] = {'start': int(raw['start']), 'bay_id': int(raw.get('bay_id', -1))}
        except (TypeError, ValueError):
            continue
    return normalized

def _latest_feasible_scaffold(prob_info, assignments):
    bounds = {int(bid): max(int(prob_info['blocks'][bid]['due_date']), int(row['exit_time'])) for bid, row in assignments.items()}
    _, info = retime(prob_info, assignments, verbose=False, timing='latest', latest_exit_bounds=bounds)
    return _copy_assignment(info['assignments'])

def _safe_area(block):
    try:
        return max(0.0, float(_layer0_area(block)))
    except Exception:
        return max(0.0, float(block.get('workload', 1.0)))

def _peak_metrics(prob_info, assignments):
    blocks = prob_info['blocks']
    events = defaultdict(lambda: defaultdict(float))
    for bid, row in assignments.items():
        bay = int(row['bay_id'])
        area = _safe_area(blocks[bid])
        events[bay][int(row['entry_time'])] += area
        events[bay][int(row['exit_time'])] -= area
    segments = defaultdict(list)
    for bay, by_time in events.items():
        times = sorted(by_time)
        used = 0.0
        bay_data = prob_info['bays'][bay]
        capacity = max(1.0, float(bay_data['width']) * float(bay_data['height']))
        for index, start in enumerate(times):
            used += by_time[start]
            if index + 1 >= len(times):
                continue
            end = times[index + 1]
            if end > start:
                segments[bay].append((int(start), int(end), float(used / capacity)))
    result = {}
    for bid, row in assignments.items():
        start = int(row['entry_time'])
        end = int(row['exit_time'])
        best = (0.0, start)
        for lo, hi, utilization in segments[int(row['bay_id'])]:
            if lo < end and start < hi:
                candidate = (float(utilization), int(lo))
                if candidate[0] > best[0] + 1e-12:
                    best = candidate
        result[int(bid)] = best
    return result

def _anchor_metrics(prob_info, assignments, plan_targets=None, peak_assignments=None):
    plan = _normalize_plan(plan_targets)
    peaks = _peak_metrics(prob_info, assignments if peak_assignments is None else peak_assignments)
    metrics = {}
    for bid, row in assignments.items():
        block = prob_info['blocks'][bid]
        target = plan.get(int(bid))
        slip = max(0, int(row['entry_time']) - int(target['start'])) if target is not None else 0
        bay_slip = int(target is not None and target['bay_id'] >= 0 and (int(row['bay_id']) != target['bay_id']))
        peak, peak_time = peaks[int(bid)]
        metrics[int(bid)] = {'late': max(0, int(row['exit_time']) - int(block['due_date'])), 'slip': int(slip), 'bay_slip': int(bay_slip), 'peak': float(peak), 'peak_time': int(peak_time), 'area': float(_safe_area(block)), 'plan_start': int(target['start']) if target is not None else None}
    return metrics

def _rank_anchors(prob_info, assignments, plan_targets=None, *, peak_assignments=None, seed=0, round_index=0):
    metrics = _anchor_metrics(prob_info, assignments, plan_targets=plan_targets, peak_assignments=peak_assignments)
    count = max(1, len(assignments))

    def phase(bid, salt):
        return (int(bid) * 37 + int(seed) * 17 + int(round_index) * 13 + int(salt) * 7) % count
    late = [('late', bid) for bid in sorted(metrics, key=lambda item: (-metrics[item]['late'], -metrics[item]['slip'], -metrics[item]['peak'], phase(item, 0), item)) if metrics[bid]['late'] > 0]
    slip = [('slip', bid) for bid in sorted(metrics, key=lambda item: (-metrics[item]['slip'], -metrics[item]['bay_slip'], -metrics[item]['late'], phase(item, 1), item)) if metrics[bid]['slip'] > 0 or metrics[bid]['bay_slip'] > 0]
    peak = [('peak', bid) for bid in sorted(metrics, key=lambda item: (-metrics[item]['peak'], -metrics[item]['area'], phase(item, 2), item))]
    families = [late, slip, peak]
    rows = []
    seen = set()
    depth = 0
    while any((depth < len(family) for family in families)):
        for family in families:
            if depth >= len(family):
                continue
            kind, bid = family[depth]
            if bid not in seen:
                seen.add(bid)
                rows.append({'kind': kind, 'bid': int(bid), **metrics[bid]})
        depth += 1
    return (rows, metrics)

def _anchor_window(prob_info, assignments, anchor):
    bid = int(anchor['bid'])
    row = assignments[bid]
    block = prob_info['blocks'][bid]
    start = int(row['entry_time'])
    end = int(row['exit_time'])
    processing = max(1, int(block['processing_time']))
    if anchor['kind'] == 'slip' and anchor['plan_start'] is not None:
        lo = min(start, int(anchor['plan_start']))
        hi = max(end, start + processing)
        center = start
    elif anchor['kind'] == 'peak':
        center = int(anchor['peak_time'])
        lo = center - max(1, processing // 2)
        hi = center + max(1, (processing + 1) // 2)
    else:
        center = max(start, int(block['due_date']) - processing)
        lo = min(start, center)
        hi = max(end, center + 1)
    return (int(lo), int(hi), int(center))

def _cohort(prob_info, assignments, scaffold, anchor, metrics, size, *, seed=0, round_index=0):
    anchor_bid = int(anchor['bid'])
    target_size = min(max(1, int(size)), max(1, len(assignments)))
    lo, hi, center = _anchor_window(prob_info, assignments, anchor)
    max_late = max(1, max((metric['late'] for metric in metrics.values())))
    max_slip = max(1, max((metric['slip'] for metric in metrics.values())))
    max_area = max(1.0, max((metric['area'] for metric in metrics.values())))
    n = max(1, len(assignments))
    ranked = []
    for bid, row in assignments.items():
        bid = int(bid)
        if bid == anchor_bid:
            continue
        timed = scaffold[bid]
        start = int(timed['entry_time'])
        end = int(timed['exit_time'])
        overlap = max(0, min(hi, end) - max(lo, start))
        active = int(start <= center < end)
        distance = 0 if overlap > 0 else min(abs(end - lo), abs(start - hi))
        metric = metrics[bid]
        signal = metric['late'] / max_late + metric['slip'] / max_slip + metric['peak'] + metric['area'] / max_area
        phase = (bid * 41 + int(seed) * 19 + int(round_index) * 11) % n
        key = (-active, -int(overlap), int(distance), -float(signal), int(phase), bid)
        ranked.append((key, bid, int(row['bay_id'])))
    ranked.sort()
    members = [anchor_bid]
    selected = {anchor_bid}
    anchor_bay = int(assignments[anchor_bid]['bay_id'])
    per_bay = {}
    for key, bid, bay in ranked:
        if bay != anchor_bay and bay not in per_bay:
            per_bay[bay] = (key, bid)
    for _, bid in sorted(per_bay.values()):
        if len(members) >= target_size:
            break
        members.append(int(bid))
        selected.add(int(bid))
    for _, bid, _ in ranked:
        if len(members) >= target_size:
            break
        if bid not in selected:
            members.append(int(bid))
            selected.add(int(bid))
    others = members[1:]
    if (int(seed) + int(round_index)) % 2:
        others.sort(key=lambda bid: (-metrics[bid]['late'], -metrics[bid]['slip'], int(scaffold[bid]['entry_time']), bid))
    else:
        others.sort(key=lambda bid: (int(scaffold[bid]['entry_time']), -metrics[bid]['peak'], bid))
    order = [anchor_bid] + others
    return {'members': tuple(sorted(members)), 'order': tuple(order), 'bays': tuple(sorted({int(assignments[bid]['bay_id']) for bid in members})), 'window': (int(lo), int(hi)), 'center': int(center)}

def heavy_alap_walk(prob_info, assignments, timelimit=40.0, *, plan_targets=None, quality_gap=0.08, min_batch=8, max_batch=12, anchors=6, max_rounds=12, max_proposals_per_round=12, candidate_timelimit=2.0, seed=0, geo_cache=None, pair_memo=None, rel_memo=None, improvement_cb=None, verbose=False):
    started = time.time()
    deadline = started + max(0.0, float(timelimit))
    gap = max(0.0, float(quality_gap))
    min_batch = max(1, int(min_batch))
    max_batch = max(min_batch, int(max_batch))
    anchors = max(1, int(anchors))
    max_rounds = max(1, int(max_rounds))
    max_proposals_per_round = max(1, int(max_proposals_per_round))
    candidate_timelimit = max(0.2, float(candidate_timelimit))
    plan = _normalize_plan(plan_targets)
    geo_cache = {} if geo_cache is None else geo_cache
    pair_memo = {} if pair_memo is None else pair_memo
    rel_memo = {} if rel_memo is None else rel_memo
    placed_cache = {}
    current = _copy_assignment(assignments)
    current_obj, current_parts = _objective_of(prob_info, current)
    current_obj = float(current_obj)
    protected = _copy_assignment(current)
    protected_obj = current_obj
    protected_obj1 = float(current_parts[0])
    visited = {_fingerprint(current)}
    bay_visits = {_bay_vector(current): 1}
    stats = {'operator': 'heavy_alap_path', 'quality_gap': float(gap), 'batch_range': [int(min_batch), int(max_batch)], 'anchors': int(anchors), 'max_rounds': int(max_rounds), 'max_proposals_per_round': int(max_proposals_per_round), 'candidate_timelimit_s': float(candidate_timelimit), 'plan_targets': int(len(plan)), 'rounds': 0, 'alap_rounds': 0, 'alap_moved_blocks': 0, 'proposals': 0, 'realized': 0, 'checker_calls': 0, 'checker_feasible': 0, 'admissible': 0, 'accepted_rounds': 0, 'improved_rounds': 0, 'obj1_rejected_improvements': 0, 'worsening_rounds': 0, 'lateral_rounds': 0, 'multi_bay_proposals': 0, 'anchor_kinds': {'late': 0, 'slip': 0, 'peak': 0}, 'path_objectives': [float(current_obj)], 'round_trace': []}
    allowed_sizes = [size for size in range(min_batch, max_batch + 1) if size in (8, 9, 10, 11, 12)] or list(range(min_batch, max_batch + 1))
    for round_index in range(max_rounds):
        if deadline - time.time() <= 1.25:
            break
        stats['rounds'] += 1
        scaffold = current
        alap_used = False
        try:
            scaffold = _latest_feasible_scaffold(prob_info, current)
            alap_used = True
            stats['alap_rounds'] += 1
            stats['alap_moved_blocks'] += sum((int(scaffold[bid]['entry_time'] != current[bid]['entry_time']) for bid in current))
        except (PositiveCycleError, ValueError, RuntimeError):
            scaffold = current
        anchor_rows, metrics = _rank_anchors(prob_info, current, plan_targets=plan, peak_assignments=scaffold, seed=seed, round_index=round_index)
        anchor_rows = anchor_rows[:anchors]
        quality_limit = protected_obj + gap * max(1.0, abs(protected_obj))
        best_candidate = None
        best_candidate_obj = 1e309
        best_candidate_obj1 = 1e309
        best_candidate_key = None
        best_candidate_bay_vector = None
        best_desc = None
        proposed = realized_count = feasible_count = admissible_count = 0
        for anchor_index, anchor in enumerate(anchor_rows):
            if proposed >= max_proposals_per_round:
                break
            left = deadline - time.time()
            if left <= 1.0:
                break
            size = allowed_sizes[(round_index + anchor_index + int(seed)) % len(allowed_sizes)]
            size = min(max_batch, max(min_batch, size))
            proposal = _cohort(prob_info, current, scaffold, anchor, metrics, size, seed=seed, round_index=round_index)
            proposed += 1
            stats['proposals'] += 1
            stats['anchor_kinds'][anchor['kind']] += 1
            stats['multi_bay_proposals'] += int(len(proposal['bays']) > 1)
            members = set(proposal['members'])
            fixed = {bid: dict(row) for bid, row in scaffold.items() if bid not in members}
            rebuild_plan = {bid: {'bay_id': -1, 'start': int(scaffold[bid]['entry_time'])} for bid in proposal['members']}
            try:
                candidate, _ = construct(prob_info, timelimit=min(candidate_timelimit, max(0.2, left - 0.8)), fixed_assignments=fixed, insert_ids=list(proposal['order']), insert_order='given', plan=rebuild_plan, plan_bays=False, plan_target_w=0.02 * float(prob_info.get('weights', {}).get('w1', 1.0)), screen='exact', max_xs=24, max_ys=16, top_b=8, geo_cache=geo_cache, placed_cache=placed_cache, pair_memo=pair_memo, rel_memo=rel_memo, verbose=False)
                solution, info = retime(prob_info, candidate, verbose=False)
                realized = _copy_assignment(info['assignments'])
                realized_count += 1
                stats['realized'] += 1
            except Exception:
                continue
            try:
                stats['checker_calls'] += 1
                verdict = check_feasibility(prob_info, solution)
            except Exception:
                continue
            if not verdict.get('feasible'):
                continue
            feasible_count += 1
            stats['checker_feasible'] += 1
            realized = assignments_from_solution(solution)
            objective = float(verdict['objective'])
            _, parts = _objective_of(prob_info, realized)
            objective1 = float(parts[0])
            fingerprint = _fingerprint(realized)
            if objective > quality_limit + 1e-09 or fingerprint in visited:
                continue
            admissible_count += 1
            stats['admissible'] += 1
            bay_vector = _bay_vector(realized)
            coarse_visits = bay_visits.get(bay_vector, 0)
            protected_eligible = objective < protected_obj - 1e-09 and objective1 <= protected_obj1 + 1e-09
            if objective < protected_obj - 1e-09 and (not protected_eligible):
                stats['obj1_rejected_improvements'] += 1
            candidate_key = (0 if protected_eligible else 1, objective if protected_eligible else float(coarse_visits), float(coarse_visits) if protected_eligible else objective, int(anchor_index))
            if best_candidate_key is None or candidate_key < best_candidate_key:
                best_candidate = realized
                best_candidate_obj = objective
                best_candidate_obj1 = objective1
                best_candidate_key = candidate_key
                best_candidate_bay_vector = bay_vector
                best_desc = {'kind': anchor['kind'], 'anchor': int(anchor['bid']), 'members': int(len(proposal['members'])), 'bays': [int(bay) for bay in proposal['bays']], 'window': [int(proposal['window'][0]), int(proposal['window'][1])], 'coarse_visits_before': int(coarse_visits)}
        trace = {'round': int(round_index + 1), 'alap': bool(alap_used), 'anchors_considered': int(len(anchor_rows)), 'proposals': int(proposed), 'realized': int(realized_count), 'checker_feasible': int(feasible_count), 'admissible': int(admissible_count), 'input_objective': float(current_obj), 'input_obj1': float(current_parts[0]), 'protected_before': float(protected_obj), 'protected_obj1_before': float(protected_obj1), 'quality_limit': float(quality_limit), 'accepted': best_candidate is not None, 'accepted_move': best_desc}
        if best_candidate is None:
            stats['round_trace'].append(trace)
            break
        previous_obj = current_obj
        current = best_candidate
        current_obj = best_candidate_obj
        _, current_parts = _objective_of(prob_info, current)
        visited.add(_fingerprint(current))
        previous_coarse_visits = bay_visits.get(best_candidate_bay_vector, 0)
        bay_visits[best_candidate_bay_vector] = previous_coarse_visits + 1
        stats['accepted_rounds'] += 1
        stats['worsening_rounds'] += int(current_obj > previous_obj + 1e-09)
        stats['lateral_rounds'] += int(math.isclose(current_obj, previous_obj, rel_tol=0.0, abs_tol=1e-09))
        if current_obj < protected_obj - 1e-09 and best_candidate_obj1 <= protected_obj1 + 1e-09:
            protected = _copy_assignment(current)
            protected_obj = current_obj
            protected_obj1 = best_candidate_obj1
            stats['improved_rounds'] += 1
            if improvement_cb is not None:
                try:
                    improvement_cb(protected, float(protected_obj), 'heavy_alap')
                except Exception:
                    pass
        stats['path_objectives'].append(float(current_obj))
        trace.update({'accepted_objective': float(current_obj), 'accepted_obj1': float(best_candidate_obj1), 'protected_after': float(protected_obj), 'protected_obj1_after': float(protected_obj1), 'delta_from_previous': float(current_obj - previous_obj)})
        stats['round_trace'].append(trace)
        if verbose:
            print(f'[HEAVY-ALAP] round={round_index + 1} current={current_obj:,.0f} protected={protected_obj:,.0f} obj1={protected_obj1:,.0f} proposals={proposed}', flush=True)
    stats.update({'protected_objective': float(protected_obj), 'protected_obj1': float(protected_obj1), 'terminal_objective': float(current_obj), 'terminal_obj1': float(current_parts[0]), 'unique_bay_vectors': int(len(bay_visits)), 'wall_s': round(time.time() - started, 3)})
    return (protected, stats, current)
