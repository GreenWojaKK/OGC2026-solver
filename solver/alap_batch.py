from __future__ import annotations
import math
import time
from constructor import construct, _objective_of
from gpr_core import PositiveCycleError, retime

def _copy_assignment(assignments):
    return {int(bid): dict(row) for bid, row in assignments.items()}

def _fingerprint(assignments):
    return tuple(((int(bid), int(row['bay_id']), int(row['x']), int(row['y']), int(row.get('orient_idx', 0)), int(row['entry_time']), int(row['exit_time'])) for bid, row in sorted(assignments.items())))

def _bay_vector(assignments):
    return tuple((int(assignments[bid]['bay_id']) for bid in sorted(assignments)))

def _fits(block, bay):
    width = int(bay['width'])
    height = int(bay['height'])
    for orient in block['shape']:
        points = [point for layer in orient['layers'] for point in layer]
        if not points:
            continue
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
        if max(xs) - min(xs) <= width + 1e-09 and max(ys) - min(ys) <= height + 1e-09:
            return True
    return False

def _bbox_area(block):
    best = 0.0
    for orient in block['shape']:
        points = [point for layer in orient['layers'] for point in layer]
        if not points:
            continue
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
        best = max(best, max(1.0, max(xs) - min(xs)) * max(1.0, max(ys) - min(ys)))
    return best

def _bay_only_score(prob_info, assignments, moves):
    changed = dict(assignments)
    for bid, bay in moves:
        row = dict(assignments[bid])
        row['bay_id'] = int(bay)
        changed[bid] = row
    return float(_objective_of(prob_info, changed)[0])

def _rank_moves(prob_info, assignments, fit_bays, seed, round_index):
    rows = []
    n_blocks = max(1, len(assignments))
    for bid in sorted(assignments):
        current_bay = int(assignments[bid]['bay_id'])
        prefs = prob_info['blocks'][bid]['bay_preferences']
        for bay in fit_bays[bid]:
            if bay == current_bay:
                continue
            score = _bay_only_score(prob_info, assignments, ((bid, bay),))
            phase = (int(bid) * 37 + int(bay) * 17 + int(seed) * 13 + int(round_index) * 7) % n_blocks
            preference_gain = float(prefs[bay]) - float(prefs[current_bay])
            rows.append((score, -preference_gain, phase, int(bid), int(bay)))
    rows.sort()
    return [(bid, bay) for _, _, _, bid, bay in rows]

def _rank_companions(prob_info, assignments, anchor, fit_bays):
    ranked = []
    for bid in sorted(assignments):
        if bid == anchor[0]:
            continue
        current_bay = int(assignments[bid]['bay_id'])
        best = None
        for bay in fit_bays[bid]:
            if bay == current_bay:
                continue
            move = (int(bid), int(bay))
            score = _bay_only_score(prob_info, assignments, (anchor, move))
            row = (score, move)
            if best is None or row < best:
                best = row
        if best is not None:
            ranked.append(best)
    ranked.sort()
    return [move for _, move in ranked]

def _overlap_rows(prob_info, scaffold, forced_moves):
    blocks = prob_info['blocks']
    forced_ids = {bid for bid, _ in forced_moves}
    physical = {}
    concurrent = {}
    for focus, target_bay in forced_moves:
        lo = int(scaffold[focus]['entry_time'])
        hi = int(scaffold[focus]['exit_time'])
        for resident in sorted(scaffold):
            if resident in forced_ids:
                continue
            row = scaffold[resident]
            overlap = min(hi, int(row['exit_time'])) - max(lo, int(row['entry_time']))
            if overlap <= 0:
                continue
            size = _bbox_area(blocks[resident])
            rank = (-float(overlap) * math.sqrt(max(1.0, size)), -float(overlap), -float(size), int(resident))
            old = concurrent.get(resident)
            if old is None or rank < old:
                concurrent[resident] = rank
            if int(row['bay_id']) == int(target_bay):
                old = physical.get(resident)
                if old is None or rank < old:
                    physical[resident] = rank
    return ([bid for bid, _ in sorted(physical.items(), key=lambda item: item[1])], [bid for bid, _ in sorted(concurrent.items(), key=lambda item: item[1])])

def _ordered_members(scaffold, members, forced, free_first):
    forced_ids = set(forced)
    free_ids = [bid for bid in members if bid not in forced_ids]
    forced_order = sorted(forced_ids, key=lambda bid: (int(scaffold[bid]['entry_time']), int(scaffold[bid]['exit_time']), int(bid)))
    free_order = sorted(free_ids, key=lambda bid: (int(scaffold[bid]['entry_time']), int(scaffold[bid]['exit_time']), int(bid)))
    return free_order + list(reversed(forced_order)) if free_first else forced_order + free_order

def _proposal_rows(prob_info, assignments, scaffold, anchor, fit_bays, batch_size):
    companions = _rank_companions(prob_info, assignments, anchor, fit_bays)
    forced_sets = [(anchor,), tuple([anchor] + companions[:1]), tuple([anchor] + companions[:2])]
    rows = []
    seen = set()

    def add(label, forced_moves, free_candidates, free_first=False):
        forced = {int(bid): int(bay) for bid, bay in forced_moves}
        member_ids = list(forced)
        for bid in free_candidates:
            if bid not in forced and bid not in member_ids:
                member_ids.append(int(bid))
            if len(member_ids) >= batch_size:
                break
        members = tuple(sorted(member_ids))
        if not members:
            return
        order = tuple(_ordered_members(scaffold, members, forced, free_first))
        key = (members, tuple(sorted(forced.items())), order)
        if key in seen:
            return
        seen.add(key)
        rows.append({'label': label, 'members': members, 'forced': forced, 'order': order})
    for forced_moves in forced_sets:
        if len(forced_moves) > batch_size:
            continue
        physical, concurrent = _overlap_rows(prob_info, scaffold, forced_moves)
        add('forced', forced_moves, ())
        add('physical', forced_moves, physical)
        add('window', forced_moves, concurrent)
        if len(concurrent) + len(forced_moves) > 2:
            add('window-free-first', forced_moves, concurrent, True)
    return rows

def alap_batch_walk(prob_info, assignments, timelimit=60.0, *, quality_gap=0.1, batch_size=6, anchors=12, max_rounds=12, max_proposals_per_round=72, candidate_timelimit=3.0, latest_scaffold=True, repair_mode='construct', cpsat_det_units=1.0, cpsat_workers=1, seed=0, geo_cache=None, pair_memo=None, rel_memo=None, verbose=False):
    started = time.time()
    deadline = started + max(0.0, float(timelimit))
    gap = max(0.0, float(quality_gap))
    batch_size = max(1, int(batch_size))
    anchors = max(1, int(anchors))
    max_rounds = max(1, int(max_rounds))
    max_proposals_per_round = max(1, int(max_proposals_per_round))
    candidate_timelimit = max(0.2, float(candidate_timelimit))
    latest_scaffold = bool(latest_scaffold)
    repair_mode = str(repair_mode).strip().lower()
    if repair_mode not in ('construct', 'cpsat', 'cpsat_novelty'):
        raise ValueError(f'unknown ALAP batch repair mode: {repair_mode}')
    cpsat_det_units = max(0.05, float(cpsat_det_units))
    cpsat_workers = max(1, int(cpsat_workers))
    geo_cache = {} if geo_cache is None else geo_cache
    pair_memo = {} if pair_memo is None else pair_memo
    rel_memo = {} if rel_memo is None else rel_memo
    placed_cache = {}
    blocks = prob_info['blocks']
    bays = prob_info['bays']
    fit_bays = {bid: [bay for bay, bay_data in enumerate(bays) if _fits(blocks[bid], bay_data)] for bid in assignments}
    current = _copy_assignment(assignments)
    current_obj = float(_objective_of(prob_info, current)[0])
    protected = _copy_assignment(current)
    protected_obj = current_obj
    visited = {_fingerprint(current)}
    bay_visits = {_bay_vector(current): 1}
    stats = {'operator': 'alap_batch_nonstrict', 'quality_gap': gap, 'batch_size': batch_size, 'anchors': anchors, 'max_rounds': max_rounds, 'max_proposals_per_round': max_proposals_per_round, 'candidate_timelimit_s': candidate_timelimit, 'latest_scaffold': latest_scaffold, 'repair_mode': repair_mode, 'cpsat_det_units': cpsat_det_units if repair_mode in ('cpsat', 'cpsat_novelty') else None, 'cpsat_workers': cpsat_workers if repair_mode in ('cpsat', 'cpsat_novelty') else None, 'cpsat_calls': 0, 'cpsat_solved': 0, 'cpsat_novelty_calls': 0, 'rounds': 0, 'alap_rounds': 0, 'alap_moved_blocks': 0, 'proposals': 0, 'realized': 0, 'realize_failures': 0, 'admissible': 0, 'accepted_rounds': 0, 'improved_rounds': 0, 'worsening_rounds': 0, 'lateral_rounds': 0, 'coarse_revisit_rounds': 0, 'path_objectives': [current_obj], 'round_trace': []}
    for round_index in range(max_rounds):
        if deadline - time.time() <= candidate_timelimit + 0.25:
            break
        stats['rounds'] += 1
        scaffold = current
        alap_used = False
        try:
            if latest_scaffold and _objective_of(prob_info, current)[1][0] <= 1e-09:
                _, retimed = retime(prob_info, current, verbose=False, timing='latest')
                scaffold = _copy_assignment(retimed['assignments'])
                alap_used = True
                stats['alap_rounds'] += 1
                stats['alap_moved_blocks'] += sum((int(scaffold[bid]['entry_time'] != current[bid]['entry_time']) for bid in current))
        except (PositiveCycleError, ValueError, RuntimeError):
            scaffold = current
        ranked_moves = _rank_moves(prob_info, current, fit_bays, seed, round_index)
        anchor_moves = ranked_moves[:anchors]
        quality_limit = protected_obj * (1.0 + gap)
        best_candidate = None
        best_candidate_obj = 1e309
        best_candidate_key = None
        best_candidate_bay_vector = None
        best_desc = None
        sequence = 0
        proposed = 0
        realized_count = 0
        admissible_count = 0
        stop_round = False
        for anchor in anchor_moves:
            for proposal in _proposal_rows(prob_info, current, scaffold, anchor, fit_bays, batch_size):
                if proposed >= max_proposals_per_round:
                    stop_round = True
                    break
                left = deadline - time.time()
                if left <= 0.45:
                    stop_round = True
                    break
                proposed += 1
                stats['proposals'] += 1
                members = set(proposal['members'])
                fixed = {bid: dict(row) for bid, row in scaffold.items() if bid not in members}
                plan = {bid: {'bay_id': proposal['forced'].get(bid, -1), 'start': int(scaffold[bid]['entry_time'])} for bid in proposal['members']}
                try:
                    if repair_mode in ('cpsat', 'cpsat_novelty'):
                        from winrep import _repair_once, _rust_lib
                        stats['cpsat_calls'] += 1
                        _novelty_solve = bool(repair_mode == 'cpsat_novelty' and round_index % 2 == 1)
                        stats['cpsat_novelty_calls'] += int(_novelty_solve)
                        lib = _rust_lib()
                        if lib is None:
                            raise RuntimeError('rustcore exact-window library unavailable')
                        candidate = _repair_once(lib, prob_info, scaffold, sorted(proposal['forced'].items()), left, explicit_window_ids=proposal['members'], solver_wall_cap=min(candidate_timelimit, max(0.05, left - 0.2)), solver_det_units=cpsat_det_units, solver_workers=cpsat_workers, quality_limit=quality_limit, novelty=_novelty_solve)
                        if candidate is None:
                            raise RuntimeError('CP-SAT exact-window repair found no solution')
                        stats['cpsat_solved'] += 1
                    else:
                        candidate, _ = construct(prob_info, timelimit=min(candidate_timelimit, max(0.2, left - 0.2)), fixed_assignments=fixed, insert_ids=list(proposal['order']), insert_order='given', plan=plan, plan_bays=True, plan_target_w=0.02 * float(prob_info.get('weights', {}).get('w1', 1.0)), screen='exact', max_xs=24, max_ys=16, top_b=8, geo_cache=geo_cache, placed_cache=placed_cache, pair_memo=pair_memo, rel_memo=rel_memo, verbose=False)
                    _, retimed = retime(prob_info, candidate, verbose=False)
                    realized = _copy_assignment(retimed['assignments'])
                    objective = float(_objective_of(prob_info, realized)[0])
                    realized_count += 1
                    stats['realized'] += 1
                except Exception:
                    stats['realize_failures'] += 1
                    sequence += 1
                    continue
                fingerprint = _fingerprint(realized)
                admissible = objective <= quality_limit + 1e-09 if gap > 0.0 else objective < protected_obj - 1e-09
                if not admissible or fingerprint in visited:
                    sequence += 1
                    continue
                admissible_count += 1
                stats['admissible'] += 1
                candidate_bay_vector = _bay_vector(realized)
                coarse_visits = bay_visits.get(candidate_bay_vector, 0)
                improves = objective < protected_obj - 1e-09
                candidate_key = (0 if improves else 1, objective if improves else float(coarse_visits), float(coarse_visits) if improves else objective, int(sequence))
                if best_candidate_key is None or candidate_key < best_candidate_key:
                    best_candidate = realized
                    best_candidate_obj = objective
                    best_candidate_key = candidate_key
                    best_candidate_bay_vector = candidate_bay_vector
                    best_desc = {'anchor': [int(anchor[0]), int(anchor[1])], 'label': proposal['label'], 'members': len(proposal['members']), 'forced': len(proposal['forced']), 'coarse_visits_before': int(coarse_visits)}
                sequence += 1
            if stop_round:
                break
        trace = {'round': int(round_index + 1), 'alap': bool(alap_used), 'anchors_considered': int(len(anchor_moves)), 'proposals': int(proposed), 'realized': int(realized_count), 'admissible': int(admissible_count), 'input_objective': float(current_obj), 'protected_before': float(protected_obj), 'quality_limit': float(quality_limit), 'accepted': best_candidate is not None, 'accepted_move': best_desc}
        if best_candidate is None:
            stats['round_trace'].append(trace)
            break
        previous_obj = current_obj
        current = best_candidate
        current_obj = best_candidate_obj
        visited.add(_fingerprint(current))
        previous_coarse_visits = bay_visits.get(best_candidate_bay_vector, 0)
        bay_visits[best_candidate_bay_vector] = previous_coarse_visits + 1
        stats['accepted_rounds'] += 1
        stats['coarse_revisit_rounds'] += int(previous_coarse_visits > 0)
        stats['worsening_rounds'] += int(current_obj > previous_obj + 1e-09)
        stats['lateral_rounds'] += int(abs(current_obj - previous_obj) <= 1e-09)
        if current_obj < protected_obj - 1e-09:
            protected = _copy_assignment(current)
            protected_obj = current_obj
            stats['improved_rounds'] += 1
        stats['path_objectives'].append(float(current_obj))
        trace.update({'accepted_objective': float(current_obj), 'protected_after': float(protected_obj), 'delta_from_previous': float(current_obj - previous_obj)})
        stats['round_trace'].append(trace)
        if verbose:
            print(f'[ALAP-NS] round={round_index + 1} current={current_obj:,.0f} protected={protected_obj:,.0f} proposals={proposed}', flush=True)
    stats['protected_objective'] = float(protected_obj)
    stats['terminal_objective'] = float(current_obj)
    stats['unique_bay_vectors'] = int(len(bay_visits))
    stats['wall_s'] = round(time.time() - started, 3)
    return (protected, stats, current)
