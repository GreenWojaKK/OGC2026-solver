from __future__ import annotations
import math
import random
import time
from constructor import construct, _objective_of
from gpr_core import retime
from lns import _op_random, _op_slab, _op_tardy
_PERTURBATION_PROFILES = {'mixed': {'tardy_small': 3.0, 'tardy_big': 1.0, 'slab': 1.0, 'random': 1.0}, 'tardy_small': {'tardy_small': 6.0, 'tardy_big': 1.0, 'slab': 1.0, 'random': 1.0}, 'tardy_big': {'tardy_small': 1.0, 'tardy_big': 6.0, 'slab': 1.0, 'random': 1.0}, 'slab': {'tardy_small': 1.0, 'tardy_big': 1.0, 'slab': 6.0, 'random': 1.0}, 'random': {'tardy_small': 1.0, 'tardy_big': 1.0, 'slab': 1.0, 'random': 6.0}}

def perturbation_weights(profile='mixed'):
    resolved = str(profile or 'mixed').strip().lower()
    if resolved not in _PERTURBATION_PROFILES:
        raise ValueError(f'unknown BC perturbation profile: {profile}')
    return dict(_PERTURBATION_PROFILES[resolved])

def _row_behavior(prob_info, bid, assignment):
    block = prob_info['blocks'][bid]
    bay_id = int(assignment['bay_id'])
    bay = prob_info['bays'][bay_id]
    x_bin = min(3, max(0, int(4.0 * float(assignment['x']) / max(1.0, float(bay['width'])))))
    y_bin = min(3, max(0, int(4.0 * float(assignment['y']) / max(1.0, float(bay['height'])))))
    tard = max(0.0, float(assignment['exit_time']) - float(block['due_date']))
    tard_bin = min(4, int(math.ceil(4.0 * tard / max(1.0, float(block['processing_time'])))))
    latest_start = float(block['due_date']) - float(block['processing_time'])
    slack = max(1.0, latest_start - float(block['release_time']))
    queue = max(0.0, float(assignment['entry_time']) - float(block['release_time']))
    queue_bin = min(4, int(4.0 * queue / slack))
    return (bay_id, x_bin, y_bin, int(assignment.get('orient_idx', 0)), tard_bin, queue_bin)

def _count_bin(value, population, bins):
    value = max(0.0, min(float(population), float(value)))
    if value <= 0.0 or population <= 0:
        return 0
    scaled = math.log1p(value) / math.log1p(float(population))
    return min(bins - 1, max(1, int(scaled * bins)))

def _signed_count_bin(value, population, bins):
    if value == 0:
        return 0
    magnitude = _count_bin(abs(value), population, bins)
    return magnitude if value > 0 else -magnitude

def _secondary_less(left, right):
    scale = max(1.0, abs(float(left)), abs(float(right)))
    return float(left) < float(right) - 1e-09 * scale

def behavior_cell(prob_info, reference, candidate, bins=8):
    bids = sorted(set(reference) & set(candidate))
    if not bids:
        return (0, 0, 0, 0, 0, 0, -1, -1)
    bay_changed = spatial_changed = 0
    tardy_support_changed = tardy_severity_changed = queue_changed = 0
    ref_tardy = candidate_tardy = 0
    tardiness_by_bay = [0.0] * len(prob_info['bays'])
    worst_bid = None
    worst_tardiness = 0.0
    for bid in bids:
        left = _row_behavior(prob_info, bid, reference[bid])
        right = _row_behavior(prob_info, bid, candidate[bid])
        bay_changed += int(left[0] != right[0])
        spatial_changed += int(left[1:4] != right[1:4])
        tardy_support_changed += int((left[4] > 0) != (right[4] > 0))
        tardy_severity_changed += int(left[4] != right[4])
        queue_changed += int(left[5] != right[5])
        ref_tardy += int(left[4] > 0)
        candidate_tardy += int(right[4] > 0)
        block = prob_info['blocks'][bid]
        tardiness = max(0.0, float(candidate[bid]['exit_time']) - float(block['due_date']))
        tardiness_by_bay[right[0]] += tardiness
        if tardiness > worst_tardiness:
            worst_tardiness = tardiness
            worst_bid = bid
    n = len(bids)
    if worst_bid is None:
        hot_bay = -1
        worst_due_quartile = -1
    else:
        hot_bay = max(range(len(tardiness_by_bay)), key=lambda bay_id: (tardiness_by_bay[bay_id], -bay_id))
        due_order = sorted(bids, key=lambda bid: (float(prob_info['blocks'][bid]['due_date']), int(bid)))
        due_rank = {bid: rank for rank, bid in enumerate(due_order)}
        worst_due_quartile = min(3, int(4 * due_rank[worst_bid] / max(1, n)))
    return (_count_bin(bay_changed, n, bins), _count_bin(tardy_support_changed, n, bins), _count_bin(tardy_severity_changed, n, bins), _count_bin(queue_changed, n, bins), _count_bin(spatial_changed, n, bins), _signed_count_bin(candidate_tardy - ref_tardy, n, bins), hot_bay, worst_due_quartile)

def qd_lns(prob_info, assignments, timelimit, reference=None, archive_seeds=None, seed=1704, quality_gap=0.12, quality_days=None, archive_restart_rate=0.15, perturbation_profile='mixed', max_iters=None, elite_sink=None, elite_limit=32, secondary_score=None, verbose=False):
    t0 = time.time()
    rng = random.Random(seed)
    geo_cache = {}
    placed_cache = {}
    pair_memo = {}
    rel_memo = {}

    def compact(value):
        try:
            _, info = retime(prob_info, value, verbose=False)
            return info['assignments']
        except Exception:
            return value
    cur = compact({bid: dict(a) for bid, a in assignments.items()})
    ref = compact({bid: dict(a) for bid, a in (reference if reference is not None else assignments).items()})
    cur_obj = _objective_of(prob_info, cur)[0]
    ref_obj = _objective_of(prob_info, ref)[0]
    use_secondary = secondary_score is not None
    cur_secondary = float(secondary_score(cur)) if use_secondary else None
    initial_secondary = cur_secondary
    best = {bid: dict(a) for bid, a in cur.items()}
    best_obj = cur_obj
    best_secondary = cur_secondary
    first_cell = behavior_cell(prob_info, ref, cur)
    archive = {first_cell: {'objective': cur_obj, **({'secondary': cur_secondary} if use_secondary else {}), 'assignments': {bid: dict(a) for bid, a in cur.items()}, 'visits': 0}}
    quality_anchor = min(ref_obj, cur_obj)
    w1 = float(prob_info.get('weights', {}).get('w1', 1.0))
    if quality_days is None:
        quality_limit = quality_anchor * (1.0 + max(0.0, quality_gap))
    else:
        quality_limit = quality_anchor + max(0.0, float(quality_days)) * max(1.0, w1)
    n_seed_cells = 1
    for seed_assignments in archive_seeds or ():
        seeded = compact({bid: dict(a) for bid, a in seed_assignments.items()})
        seeded_obj = _objective_of(prob_info, seeded)[0]
        seeded_secondary = float(secondary_score(seeded)) if use_secondary else None
        seeded_cell = behavior_cell(prob_info, ref, seeded)
        previous = archive.get(seeded_cell)
        seeded_tie_better = use_secondary and previous is not None and (abs(seeded_obj - previous['objective']) <= 1e-09) and _secondary_less(seeded_secondary, previous['secondary'])
        if seeded_obj <= quality_limit + 1e-09 and (previous is None or seeded_obj < previous['objective'] - 1e-09 or seeded_tie_better):
            archive[seeded_cell] = {'objective': seeded_obj, **({'secondary': seeded_secondary} if use_secondary else {}), 'assignments': {bid: dict(a) for bid, a in seeded.items()}, 'visits': 0}
            n_seed_cells = len(archive)
        if seeded_obj < best_obj - 1e-09:
            best = {bid: dict(a) for bid, a in seeded.items()}
            best_obj = seeded_obj
            best_secondary = seeded_secondary
        elif use_secondary and abs(seeded_obj - best_obj) <= 1e-09 and _secondary_less(seeded_secondary, best_secondary):
            best = {bid: dict(a) for bid, a in seeded.items()}
            best_secondary = seeded_secondary
    stats = {'iterations': 0, 'accepted': 0, 'novel_accepts': 0, 'cell_improve_accepts': 0, 'improvements': 0, 'cells': len(archive), 'seed_cells': n_seed_cells, 'archive_restarts': 0, 'band_rejects': 0, 'secondary_tie_candidates': 0, 'secondary_tie_accepts': 0, 'secondary_tie_rejects': 0, 'secondary_tie_descents': 0, 'secondary_tie_neutral': 0, 'secondary_tie_ascents': 0, 'secondary_tie_delta_sum': 0.0, 'secondary_tie_delta_min': None, 'secondary_tie_delta_max': None, 'secondary_cell_replacements': 0, 'secondary_best_updates': 0}
    resolved_perturbation = str(perturbation_profile or 'mixed').strip().lower()
    op_bias = perturbation_weights(resolved_perturbation)
    op_weights = dict(op_bias)
    op_stats = {name: [0, 0, 0] for name in op_weights}
    while time.time() - t0 < timelimit - 1.0:
        if max_iters is not None and stats['iterations'] >= max_iters:
            break
        elapsed = time.time() - t0
        stats['iterations'] += 1
        parent = cur
        parent_obj = cur_obj
        parent_secondary = cur_secondary
        if len(archive) > 1 and rng.random() < max(0.0, min(1.0, archive_restart_rate)):
            sample = rng.sample(list(archive.values()), min(4, len(archive)))
            elite = min(sample, key=lambda row: (row['visits'], row['objective'], row.get('secondary', 0.0)))
            elite['visits'] += 1
            parent = elite['assignments']
            parent_obj = elite['objective']
            parent_secondary = elite.get('secondary')
            stats['archive_restarts'] += 1
        op_name = rng.choices(list(op_weights), weights=[op_weights[name] for name in op_weights])[0]
        op_stats[op_name][0] += 1
        if op_name == 'tardy_small':
            removed = _op_tardy(prob_info, parent, rng, rng.randint(4, 10))
        elif op_name == 'tardy_big':
            removed = _op_tardy(prob_info, parent, rng, rng.randint(14, 28))
        elif op_name == 'slab':
            removed = _op_slab(prob_info, parent, rng)
        else:
            removed = _op_random(prob_info, parent, rng, rng.randint(5, 14))
        removed_set = set(removed)
        fixed = {bid: a for bid, a in parent.items() if bid not in removed_set}
        try:
            proposal, _ = construct(prob_info, timelimit=10.0 if max_iters is not None else max(2.0, timelimit - elapsed), fixed_assignments=fixed, insert_ids=removed, insert_order='edd', screen='exact', geo_cache=geo_cache, placed_cache=placed_cache, pair_memo=pair_memo, rel_memo=rel_memo, max_xs=12, max_ys=8, top_b=4, verbose=False)
            proposal = compact(proposal)
            proposal_obj = _objective_of(prob_info, proposal)[0]
            proposal_secondary = float(secondary_score(proposal)) if use_secondary else None
        except Exception:
            continue
        cell = behavior_cell(prob_info, ref, proposal)
        previous = archive.get(cell)
        novel = previous is None
        objective_cell_improvement = previous is not None and proposal_obj < previous['objective'] - 1e-09
        secondary_cell_improvement = use_secondary and previous is not None and (abs(proposal_obj - previous['objective']) <= 1e-09) and _secondary_less(proposal_secondary, previous['secondary'])
        cell_improvement = objective_cell_improvement or secondary_cell_improvement
        within_band = proposal_obj <= quality_limit + 1e-09
        delta = proposal_obj - parent_obj
        archive_insert = within_band and (novel or cell_improvement)
        if archive_insert:
            archive[cell] = {'objective': proposal_obj, **({'secondary': proposal_secondary} if use_secondary else {}), 'assignments': {bid: dict(a) for bid, a in proposal.items()}, 'visits': 0}
            stats['cells'] = len(archive)
            stats['secondary_cell_replacements'] += int(secondary_cell_improvement)
        elif not within_band:
            stats['band_rejects'] += 1
        objective_tie = abs(delta) <= 1e-09
        secondary_descent = use_secondary and objective_tie and _secondary_less(proposal_secondary, parent_secondary)
        secondary_ascent = use_secondary and objective_tie and _secondary_less(parent_secondary, proposal_secondary)
        secondary_neutral = use_secondary and objective_tie and (not secondary_descent) and (not secondary_ascent)
        if use_secondary and objective_tie:
            stats['secondary_tie_candidates'] += 1
            stats['secondary_tie_descents'] += int(secondary_descent)
            stats['secondary_tie_neutral'] += int(secondary_neutral)
            stats['secondary_tie_ascents'] += int(secondary_ascent)
            secondary_delta = float(proposal_secondary - parent_secondary)
            stats['secondary_tie_delta_sum'] += secondary_delta
            low = stats['secondary_tie_delta_min']
            high = stats['secondary_tie_delta_max']
            stats['secondary_tie_delta_min'] = secondary_delta if low is None else min(low, secondary_delta)
            stats['secondary_tie_delta_max'] = secondary_delta if high is None else max(high, secondary_delta)
        fraction = stats['iterations'] / max_iters if max_iters is not None else elapsed / max(1.0, timelimit)
        fraction = min(1.0, fraction)
        temperature = max(1.0, w1) * (1.0 / 100.0) ** fraction
        if use_secondary and objective_tie:
            secondary_admissible = not secondary_ascent
            sa_accept = within_band and secondary_admissible
            qd_accept = archive_insert and secondary_admissible
        else:
            sa_accept = within_band and (delta <= 0.0 or rng.random() < math.exp(-max(0.0, delta) / temperature))
            qd_accept = archive_insert
        if not (sa_accept or qd_accept):
            if use_secondary and objective_tie:
                stats['secondary_tie_rejects'] += 1
            op_weights[op_name] = 0.9 * op_weights[op_name] + 0.1 * (1.0 if resolved_perturbation == 'mixed' else op_bias[op_name])
            continue
        cur, cur_obj = (proposal, proposal_obj)
        cur_secondary = proposal_secondary
        stats['accepted'] += 1
        stats['secondary_tie_accepts'] += int(use_secondary and objective_tie)
        stats['novel_accepts'] += int(qd_accept and novel)
        stats['cell_improve_accepts'] += int(qd_accept and cell_improvement)
        reward = 1.0 if archive_insert else 0.3
        op_stats[op_name][1] += int(archive_insert)
        if proposal_obj < best_obj:
            best = {bid: dict(a) for bid, a in proposal.items()}
            best_obj = proposal_obj
            best_secondary = proposal_secondary
            stats['improvements'] += 1
            op_stats[op_name][2] += 1
            reward = 3.0
        elif use_secondary and abs(proposal_obj - best_obj) <= 1e-09 and _secondary_less(proposal_secondary, best_secondary):
            best = {bid: dict(a) for bid, a in proposal.items()}
            best_secondary = proposal_secondary
            stats['secondary_best_updates'] += 1
            reward = max(reward, 1.0)
        op_weights[op_name] = 0.9 * op_weights[op_name] + 0.1 * (1.0 + reward if resolved_perturbation == 'mixed' else op_bias[op_name] * (1.0 + reward))
    stats['op_stats'] = {name: tuple(values) for name, values in op_stats.items()}
    stats['quality_anchor'] = quality_anchor
    stats['quality_limit'] = quality_limit
    stats['quality_days'] = quality_days
    stats['archive_restart_rate'] = archive_restart_rate
    stats['perturbation_profile'] = resolved_perturbation
    stats['initial_operator_weights'] = perturbation_weights(resolved_perturbation)
    if use_secondary:
        stats['secondary_initial'] = float(initial_secondary)
        stats['secondary_best'] = float(best_secondary)
    if elite_sink is not None:
        elite_sink.clear()
        ranked = sorted(archive.items(), key=(lambda item: (item[1]['objective'], item[1]['secondary'], item[0])) if use_secondary else lambda item: (item[1]['objective'], item[0]))
        for cell, row in ranked[:max(1, int(elite_limit))]:
            elite_sink.append({'cell': tuple(cell), 'objective': float(row['objective']), **({'secondary': float(row['secondary'])} if use_secondary else {}), 'assignments': {bid: dict(a) for bid, a in row['assignments'].items()}})
        stats['exported_elites'] = len(elite_sink)
    if verbose:
        print(f'[BC-QD] it={stats['iterations']} acc={stats['accepted']} novel={stats['novel_accepts']} cellimp={stats['cell_improve_accepts']} cells={stats['cells']} restarts={stats['archive_restarts']} bandrej={stats['band_rejects']} best={best_obj:,.0f}', flush=True)
    return (best, stats)
