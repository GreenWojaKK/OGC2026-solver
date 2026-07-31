import os
import sys
import time
PROGRESS = {}
RUN_PROVENANCE = {}

def _rust_enabled():
    return sys.platform.startswith('linux') and True


def _pup_gate_active(use_aeff, dr, pressure):
    return float(dr) < 0.6 and (bool(use_aeff) and float(pressure) >= 0.93)

def _winrep_enabled(pup_gate=False):
    if pup_gate and True:
        return False
    return True

def _portfolio_pup_state(arm_meta):
    return (any((bool(meta.get('pup_gate', False)) for meta in arm_meta.values())), any((bool(meta.get('pup_promoted', False)) for meta in arm_meta.values())))

def _resolve_arm_route(dr, pup, pup_heavy=False, forced_route='', pup_force_light=False):
    if forced_route:
        return (forced_route in ('heavy', 'both'), forced_route == 'both')
    if pup and pup_force_light:
        return (False, False)
    both = (0.6 <= dr < 0.7 or pup) and (not pup_heavy)
    slr = dr >= 0.7 or both or pup_heavy
    return (slr, both)

def _parent_tail_reserve_configured_s(plan_region, arm_dr):
    if plan_region and arm_dr < 0.7:
        return 0.0
    if plan_region:
        return 45.0
    return 45.0


def _parent_tail_reserve_s(plan_region, arm_dr):
    configured = _parent_tail_reserve_configured_s(plan_region, arm_dr)
    return configured

def _light_seed_descriptor(prob_info, assignments):
    blocks = prob_info['blocks']
    bays = prob_info['bays']
    m = len(bays)
    loads = [0.0] * m
    areas = [max(1.0, float(bay['width']) * float(bay['height'])) for bay in bays]
    avg_area = sum(areas) / max(1, m)
    rows = {}
    for bid, block in enumerate(blocks):
        row = assignments[bid]
        bay = int(row['bay_id'])
        loads[bay] += float(block.get('workload', 1.0))
        prefs = block.get('bay_preferences', [0.0] * m)
        pref_hi = float(max(prefs)) if prefs else 0.0
        pref_lo = float(min(prefs)) if prefs else 0.0
        pref_regret = (pref_hi - float(prefs[bay])) / (pref_hi - pref_lo) if pref_hi > pref_lo + 1e-12 else 0.0
        release = float(block.get('release_time', 0.0))
        latest = float(block.get('due_date', release)) - float(block.get('processing_time', 0.0))
        phase = (float(row['entry_time']) - release) / max(1.0, latest - release)
        bay_data = bays[bay]
        rows[bid] = (bay, max(0.0, min(1.0, pref_regret)), max(0.0, min(1.0, phase)), max(0.0, min(1.0, float(row['x']) / max(1.0, float(bay_data['width'])))), max(0.0, min(1.0, float(row['y']) / max(1.0, float(bay_data['height'])))), int(row.get('orient_idx', 0)))
    adjusted = [avg_area * loads[bay] / areas[bay] for bay in range(m)]
    load_scale = max(1.0, sum((abs(value) for value in adjusted)))
    return {'rows': rows, 'load_share': tuple((value / load_scale for value in adjusted))}

def _light_seed_distance(left, right):
    left_rows = left['rows']
    right_rows = right['rows']
    bids = sorted(set(left_rows) | set(right_rows))
    if not bids:
        return 0.0
    bay_delta = pref_delta = phase_delta = spatial_delta = 0.0
    for bid in bids:
        a = left_rows.get(bid)
        b = right_rows.get(bid)
        if a is None or b is None:
            bay_delta += 1.0
            pref_delta += 1.0
            phase_delta += 1.0
            spatial_delta += 1.0
            continue
        bay_delta += float(a[0] != b[0])
        pref_delta += abs(a[1] - b[1])
        phase_delta += abs(a[2] - b[2])
        spatial_delta += (abs(a[3] - b[3]) + abs(a[4] - b[4]) + float(a[5] != b[5])) / 3.0
    n = float(len(bids))
    left_load = left['load_share']
    right_load = right['load_share']
    load_delta = 0.5 * sum((abs(a - b) for a, b in zip(left_load, right_load)))
    return min(1.0, max(0.0, 0.42 * bay_delta / n + 0.18 * pref_delta / n + 0.15 * phase_delta / n + 0.1 * spatial_delta / n + 0.15 * load_delta))


def _append_strict_incumbent_event(trace, objective, elapsed_s, event, **fields):
    if not isinstance(trace, dict) or not trace.get('enabled', False):
        return False
    try:
        objective = float(objective)
        elapsed_s = float(elapsed_s)
    except (TypeError, ValueError):
        return False
    best = trace.get('best_objective')
    if isinstance(best, (int, float)) and objective >= float(best) - 1e-09:
        return False
    row = {'elapsed_s': round(elapsed_s, 3), 'event': str(event), 'objective': objective}
    row.update({key: value for key, value in fields.items() if value is not None})
    trace.setdefault('events', []).append(row)
    trace['best_objective'] = objective
    return True





def _select_mseed_starts(prob_info, pool, count, mode=None, pool_limit=None, quality_gap=None):
    count = max(0, int(count))
    ranked = sorted(pool, key=lambda row: (float(row[0]), int(row[1]), int(row[2])))
    unique = []
    seen = set()
    for row in ranked:
        fingerprint = row[3]
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        unique.append(row)
    mode = str(mode if mode is not None else 'objective').lower()
    if mode not in ('objective', 'qd31', 'qd22'):
        mode = 'objective'
    if count == 0 or not unique:
        return ([], {'mode': mode, 'unique': len(unique), 'eligible': 0, 'selected': []})
    preserve = {'objective': count, 'qd31': min(3, count), 'qd22': min(2, count)}[mode]
    selected = list(unique[:preserve])
    eligible = unique
    if mode != 'objective':
        limit = int(pool_limit if pool_limit is not None else '12')
        gap = float(quality_gap if quality_gap is not None else '0.06')
        ceiling = float(unique[0][0]) + max(0.0, gap) * max(1.0, abs(float(unique[0][0])))
        eligible = [row for row in unique[:max(count, limit)] if float(row[0]) <= ceiling + 1e-09]
        descriptor_rows = list(eligible)
        descriptor_hashes = {row[3] for row in descriptor_rows}
        descriptor_rows.extend((row for row in selected if row[3] not in descriptor_hashes))
        descriptors = {row[3]: _light_seed_descriptor(prob_info, row[4]) for row in descriptor_rows}
        while len(selected) < count:
            candidates = [row for row in eligible if row not in selected]
            if not candidates:
                break
            row = min(candidates, key=lambda candidate: (-min((_light_seed_distance(descriptors[candidate[3]], descriptors[chosen[3]]) for chosen in selected)), float(candidate[0]), int(candidate[1]), int(candidate[2])))
            selected.append(row)
    for row in unique:
        if len(selected) >= count:
            break
        if row not in selected:
            selected.append(row)
    descriptor_cache = {}
    diagnostics = []
    for rank, row in enumerate(selected):
        min_distance = 0.0
        if rank:
            descriptor = descriptor_cache.setdefault(row[3], _light_seed_descriptor(prob_info, row[4]))
            min_distance = min((_light_seed_distance(descriptor, descriptor_cache.setdefault(previous[3], _light_seed_descriptor(prob_info, previous[4]))) for previous in selected[:rank]))
        diagnostics.append({'objective': float(row[0]), 'worker': int(row[1]), 'index': int(row[2]), 'rank': int(unique.index(row)), 'min_distance': float(min_distance)})
    return ([row[4] for row in selected[:count]], {'mode': mode, 'unique': len(unique), 'eligible': len(eligible), 'selected': diagnostics})

def _seatgap_log(**rec):
    return


def _scout_worker(q, w, prob_info, n_star, stride, *, stream=False, stop_event=None):
    try:
        try:
            import numba
            numba.set_num_threads(1)
        except Exception:
            pass
        os.environ['OGC_ARM_THREADS'] = '1'
        from constructor import construct_multi
        col = {'rows': [], 'assigns': [], 'k': 0 if stream else 8}
        if stream:
            col.update({'stream_queue': q, 'stream_worker': int(w), 'stop_event': stop_event})
        n_w = n_star // stride + (1 if w < n_star % stride else 0)
        construct_multi(prob_info, timelimit=1000000000.0, verbose=False, use_legacy=w == 0, max_passes=n_w, var_slice=(w, stride), collect_top=col)
        if stream:
            q.put(('done', int(w)))
        else:
            q.put((w, col['rows'], col['assigns']))
    except Exception as exc:
        if stream:
            q.put(('error', int(w), repr(exc)))
        else:
            q.put((w, [], []))

def _mseed_work_plan(prob_info, timelimit, selection_mode=None, default_seconds=None):
    from plan_core import demand_ratio
    n_blocks = len(prob_info.get('blocks', []))
    n_bays = len(prob_info['bays'])
    dr = demand_ratio(prob_info)
    if dr < 0.45:
        pass_s = 0.21 if n_blocks <= 120 else 0.28 if n_blocks <= 170 else 0.55 if n_blocks <= 240 else (1.25 if n_bays >= 5 else 0.7) if n_blocks <= 350 else 1.9 if n_bays >= 5 else 1.05
    else:
        pass_s = 0.45 if n_blocks <= 120 else 1.5 if n_blocks <= 240 else 2.6 if n_blocks <= 350 else 3.9
    seconds_raw = None
    if seconds_raw is None and default_seconds is not None:
        seconds_raw = str(float(default_seconds))
    if seconds_raw is None and dr >= 0.4:
        seconds_raw = '20'
    if seconds_raw is not None:
        target_s = max(6.0, float(seconds_raw))
        draws = max(8, min(256, int(max(1.0, target_s - 4.0) * 2.0 / pass_s)))
        scheduler = 'seconds'
    else:
        draws = max(32, min(256, int(0.2 * float(timelimit) * 4.0 / (2.0 * pass_s))))
        target_s = 0.2 * float(timelimit) + 4.0
        scheduler = 'fraction'
    return {'dr': float(dr), 'pass_seconds': float(pass_s), 'draws': int(draws), 'planned_seconds': float(draws * pass_s / 2.0 + 4.0), 'target_seconds': float(target_s), 'scheduler': scheduler}

def _arm_refill_plan(timelimit, prearm_wall, base_arm_tl, *, enabled=False, cap=30.0, quantum=5.0, parent_margin=8.0):
    total = max(0.0, float(timelimit))
    elapsed = max(0.0, float(prearm_wall))
    floor = max(15.0, float(base_arm_tl))
    margin = max(5.0, float(parent_margin))
    refill_cap = max(0.0, float(cap))
    step = max(1.0, float(quantum))
    available_tl = max(15.0, total - elapsed - margin)
    raw = max(0.0, available_tl - floor)
    bounded = min(refill_cap, raw) if enabled else 0.0
    refill = step * int(bounded // step)
    return {'enabled': bool(enabled), 'base_arm_timelimit': float(floor), 'prearm_wall_s': float(elapsed), 'parent_margin_s': float(margin), 'available_arm_timelimit': float(available_tl), 'raw_surplus_s': float(raw), 'cap_s': float(refill_cap), 'quantum_s': float(step), 'refill_s': float(refill), 'arm_timelimit': float(floor + refill)}

def _arm_refill_enabled(menu):
    return menu == 'alap_mixed4'

def _arm_worker(q, idx, prob_info, timelimit, cfg, start_assign=None, portfolio_t0=None, factory_handoff_queues=None):
    worker_started = time.time()
    os.environ.update(cfg)
    try:
        try:
            import numba
            numba.set_num_threads(int(cfg.get('OGC_ARM_THREADS', '2')))
        except Exception:
            pass
        finally:
            pass
        from constructor import _objective_of as _obj_of
        from gpr_core import retime as _retime
        _sent = [None, 0.0]
        _curve_sent = [None]

        def _ckpt(assign):
            o = _obj_of(prob_info, assign)[0]
            if _sent[0] is not None and (o >= _sent[0] - 1e-09 or time.time() - _sent[1] < 6.0):
                return
            try:
                snap, _ = _retime(prob_info, assign, verbose=False)
                q.put(('ckpt', idx, snap, float(o)))
                _sent[0], _sent[1] = (o, time.time())
            except Exception:
                pass

        def _curve_ckpt(assign, stage):
            return
        run_meta = {'arm': int(idx)}

        def _route_meta(meta):
            try:
                q.put(('meta', idx, None, None, dict(meta)))
            except Exception:
                pass

        def _factory_handoff(payload):
            delivered = 0
            for feed_q in factory_handoff_queues or ():
                try:
                    feed_q.put(payload)
                    delivered += 1
                except Exception:
                    pass
            try:
                q.put(('factory_handoff', idx, None, float(payload['objective']), {'source_arm': int(idx), 'rounds_completed': int(payload['rounds_completed']), 'plan_bucket': int(payload.get('plan_bucket', 0)), 'objective': float(payload['objective']), 'consumers': int(delivered)}))
            except Exception:
                pass
        sol = _algorithm_single(prob_info, timelimit, ckpt_cb=_ckpt, start_assign=start_assign, run_meta=run_meta, route_meta_cb=_route_meta, trace_cb=_curve_ckpt, trace_origin=portfolio_t0, factory_handoff_cb=_factory_handoff if factory_handoff_queues else None)
        from utils import check_feasibility
        r = check_feasibility(prob_info, sol)
        run_meta['objective'] = float(r['objective']) if r['feasible'] else None
        run_meta['feasible'] = bool(r['feasible'])
        run_meta['worker_wall_s'] = float(time.time() - worker_started)
        final_message = ('final', idx, sol if r['feasible'] else None, float(r['objective']) if r['feasible'] else None, run_meta)
        q.put(final_message)
    except Exception as e:
        q.put(('final', idx, None, f'{type(e).__name__}: {e}', {'arm': int(idx), 'error': f'{type(e).__name__}: {e}', 'worker_wall_s': float(time.time() - worker_started)}))
    finally:
        pass

def _factory_consumer_worker(q, feed_q, idx, prob_info, timelimit, cfg, portfolio_t0=None):
    worker_started = time.time()
    t_end = worker_started + float(timelimit)
    os.environ.update(cfg)
    role = cfg.get('OGC_FACTORY_ROLE', 'consumer')
    try:
        try:
            import numba
            numba.set_num_threads(int(cfg.get('OGC_ARM_THREADS', '1')))
        except Exception:
            pass
        finally:
            pass
        from constructor import _objective_of as _obj_of
        from gpr_core import retime as _retime
        from utils import check_feasibility
        wait_budget = min(max(0.0, float(timelimit) - 20.0), max(15.0, float(timelimit) * 0.45))
        wait_deadline = worker_started + wait_budget
        handoff = None
        while handoff is None and time.time() < wait_deadline:
            try:
                item = feed_q.get(timeout=max(0.05, min(1.0, wait_deadline - time.time())))
            except Exception:
                continue
            finally:
                pass
            if isinstance(item, dict) and isinstance(item.get('plan'), dict) and isinstance(item.get('assignment'), dict):
                handoff = item
        _sent = [None, 0.0]

        def _ckpt(assign):
            objective = _obj_of(prob_info, assign)[0]
            if _sent[0] is not None and (objective >= _sent[0] - 1e-09 or time.time() - _sent[1] < 6.0):
                return
            try:
                snap, _ = _retime(prob_info, assign, verbose=False)
                q.put(('ckpt', idx, snap, float(objective)))
                _sent[0], _sent[1] = (objective, time.time())
            except Exception:
                pass
        run_meta = {'arm': int(idx), 'factory_role': str(role), 'factory_consumer': True, 'factory_handoff_received': bool(handoff is not None), 'factory_handoff_wait_s': float(time.time() - worker_started), 'factory_handoff_objective': float(handoff['objective']) if handoff is not None else None, 'factory_handoff_rounds': int(handoff['rounds_completed']) if handoff is not None else None, 'factory_handoff_plan_bucket': int(handoff.get('plan_bucket', 0)) if handoff is not None else None}

        def _route_meta(meta):
            try:
                q.put(('meta', idx, None, None, dict(meta)))
            except Exception:
                pass
        remaining_tl = max(15.0, t_end - time.time())
        start_assign = handoff['assignment'] if handoff is not None else None
        sol = _algorithm_single(prob_info, remaining_tl, ckpt_cb=_ckpt, start_assign=start_assign, run_meta=run_meta, route_meta_cb=_route_meta, trace_origin=portfolio_t0, factory_handoff=handoff)
        result = check_feasibility(prob_info, sol)
        run_meta['objective'] = float(result['objective']) if result['feasible'] else None
        run_meta['feasible'] = bool(result['feasible'])
        run_meta['worker_wall_s'] = float(time.time() - worker_started)
        q.put(('final', idx, sol if result['feasible'] else None, float(result['objective']) if result['feasible'] else None, run_meta))
    except Exception as exc:
        q.put(('final', idx, None, f'{type(exc).__name__}: {exc}', {'arm': int(idx), 'factory_role': str(role), 'factory_consumer': True, 'error': f'{type(exc).__name__}: {exc}', 'worker_wall_s': float(time.time() - worker_started)}))
    finally:
        pass


def _relay_tail_worker(q, feed_q, idx, prob_info, timelimit, cfg, start_assign=None):
    try:
        os.environ.update(cfg)
        try:
            import numba
            numba.set_num_threads(1)
        except Exception:
            pass
        os.environ['OGC_ARM_THREADS'] = '1'
        from constructor import _objective_of as _obj_of, _RUST_LIB
        from gpr_core import assignments_from_solution, retime
        from lns import lns_sa
        from utils import check_feasibility
        t_end = time.time() + float(timelimit)
        window = max(5.0, float(os.environ.get('OGC_RELAY_WINDOW', '15')))
        seed = int(cfg.get('OGC_ARM_SEED', 700 + int(idx)))
        run_meta = {'arm': int(idx), 'route': 'relay', 'relay_seat': True, 'relay_source': cfg.get('OGC_RELAY_SOURCE', 'global'), 'rust_enabled': bool(_rust_enabled()), 'rust_native_loaded': bool(_RUST_LIB is not None), 'relay_window_s': float(window), 'windows': 0, 'adoptions': [], 'winning_stage': 'relay'}
        try:
            q.put(('meta', idx, None, None, dict(run_meta)))
        except Exception:
            pass
        _late = os.environ.get('OGC_RELAY_LATE', '1') == '1'
        run_meta['relay_late'] = bool(_late)
        if _late:
            cur, cur_obj = (None, 1e309)
            run_meta['seed_objective'] = None
        else:
            cur = {int(b): dict(row) for b, row in start_assign.items()}
            cur_obj = float(_obj_of(prob_info, cur)[0])
            run_meta['seed_objective'] = cur_obj
        best_sol, best_obj = (None, 1e309)
        _sent = [None, 0.0]

        def _publish(assign):
            o = _obj_of(prob_info, assign)[0]
            if _sent[0] is not None and (o >= _sent[0] - 1e-09 or time.time() - _sent[1] < 6.0):
                return
            try:
                snap, _ = retime(prob_info, assign, verbose=False)
                q.put(('ckpt', idx, snap, float(o)))
                _sent[0], _sent[1] = (o, time.time())
            except Exception:
                pass
        if _late:
            _start_frac = min(0.9, max(0.0, float(os.environ.get('OGC_RELAY_START_FRAC', '0.15'))))
            _t_go = t_end - float(timelimit) * (1.0 - _start_frac)
            run_meta['relay_start_frac'] = float(_start_frac)
            _prelude_mode = os.environ.get('OGC_RELAY_PRELUDE', '').strip().lower()
            run_meta['relay_prelude'] = {'requested': _prelude_mode or None, 'attempted': False, 'feasible': False, 'budget_s': 0.0, 'wall_s': 0.0, 'objective': None}
            if _prelude_mode:
                _prelude_frac = min(_start_frac, max(0.0, float(os.environ.get('OGC_RELAY_PRELUDE_FRAC', str(max(0.0, _start_frac - 0.05))))))
                _prelude_tl = min(float(timelimit) * _prelude_frac, max(0.0, _t_go - time.time() - 3.0))
                run_meta['relay_prelude']['budget_s'] = float(_prelude_tl)
                if _prelude_tl >= 15.0:
                    _prelude_started = time.time()
                    run_meta['relay_prelude']['attempted'] = True
                    _prelude_keys = ('OGC_FORCE_ROUTE', 'OGC_PLAN_BUCKET')
                    _missing = object()
                    _prelude_saved = {key: os.environ.get(key, _missing) for key in _prelude_keys}
                    try:
                        if _prelude_mode in ('heavy', 'heavy_bucket', 'heavy_bucket5'):
                            os.environ['OGC_FORCE_ROUTE'] = 'heavy'
                            os.environ['OGC_PLAN_BUCKET'] = os.environ.get('OGC_RELAY_PRELUDE_BUCKET', '5')
                        else:
                            raise ValueError('OGC_RELAY_PRELUDE must be heavy_bucket5')
                        _prelude_meta = {}
                        _prelude_sol = _algorithm_single(prob_info, timelimit=_prelude_tl, run_meta=_prelude_meta)
                        _prelude_check = check_feasibility(prob_info, _prelude_sol)
                        run_meta['relay_prelude'].update({'feasible': bool(_prelude_check['feasible']), 'objective': float(_prelude_check['objective']) if _prelude_check['feasible'] else None, 'route': _prelude_meta.get('route'), 'winning_stage': _prelude_meta.get('winning_stage')})
                        if _prelude_check['feasible']:
                            cur = assignments_from_solution(_prelude_sol)
                            cur_obj = float(_obj_of(prob_info, cur)[0])
                            best_sol = _prelude_sol
                            best_obj = float(_prelude_check['objective'])
                            q.put(('ckpt', idx, _prelude_sol, float(_prelude_check['objective'])))
                    except Exception as _prelude_exc:
                        run_meta['relay_prelude']['error'] = f'{type(_prelude_exc).__name__}: {_prelude_exc}'
                    finally:
                        for key, value in _prelude_saved.items():
                            if value is _missing:
                                os.environ.pop(key, None)
                            else:
                                os.environ[key] = value
                        run_meta['relay_prelude']['wall_s'] = float(time.time() - _prelude_started)
            while time.time() < min(_t_go, t_end - 8.0):
                time.sleep(0.5)
            _acc = []
            while True:
                try:
                    _acc.append(feed_q.get_nowait())
                except Exception:
                    break
            _acc = [a for a in _acc if isinstance(a[0], float) and a[1] is not None]
            if _acc:
                f_obj, f_sol = min(_acc, key=lambda a: a[0])
                if cur is None or f_obj < cur_obj - 1e-09:
                    cur = assignments_from_solution(f_sol)
                    cur_obj = float(f_obj)
                    run_meta['late_start_s'] = round(float(timelimit) - (t_end - time.time()), 1)
                    run_meta['adoptions'].append({'window': 0, 'objective': float(f_obj), 'elapsed_s': run_meta['late_start_s']})
            if cur is None:
                while cur is None and time.time() < t_end - 8.0:
                    try:
                        f_obj, f_sol = feed_q.get(timeout=2.0)
                    except Exception:
                        continue
                    if isinstance(f_obj, float) and f_sol is not None:
                        cur = assignments_from_solution(f_sol)
                        cur_obj = float(f_obj)
                        run_meta['late_start_s'] = round(float(timelimit) - (t_end - time.time()), 1)
                        run_meta['adoptions'].append({'window': 0, 'objective': float(f_obj), 'elapsed_s': run_meta['late_start_s']})
        while cur is not None and time.time() < t_end - 2.0:
            while True:
                try:
                    item = feed_q.get_nowait()
                except Exception:
                    break
                try:
                    f_obj, f_sol = item
                    if not isinstance(f_obj, float) or f_sol is None:
                        continue
                    if f_obj < cur_obj - 1e-09:
                        cur = assignments_from_solution(f_sol)
                        cur_obj = float(f_obj)
                        run_meta['adoptions'].append({'window': int(run_meta['windows']), 'objective': float(f_obj), 'elapsed_s': round(float(timelimit) - (t_end - time.time()), 1)})
                except Exception:
                    continue
            budget = min(window, t_end - time.time() - 2.0)
            if budget < 3.0:
                break
            run_meta['windows'] += 1
            try:
                walked, _ = lns_sa(prob_info, cur, timelimit=budget, seed=seed + run_meta['windows'], verbose=False)
            except Exception:
                break
            try:
                w_obj = float(_obj_of(prob_info, walked)[0])
            except Exception:
                continue
            if w_obj < cur_obj - 1e-09:
                cur, cur_obj = (walked, w_obj)
                _publish(cur)
            if w_obj < best_obj - 1e-09:
                try:
                    sol, _ = retime(prob_info, walked, verbose=False)
                    r = check_feasibility(prob_info, sol)
                    if r['feasible']:
                        best_sol = sol
                        best_obj = float(r['objective'])
                except Exception:
                    pass
        _fin_best = None
        while True:
            try:
                f_obj, f_sol = feed_q.get_nowait()
            except Exception:
                break
            if isinstance(f_obj, float) and f_sol is not None and (_fin_best is None or f_obj < _fin_best[0]):
                _fin_best = (float(f_obj), f_sol)
        _rem = t_end - time.time()
        if _fin_best is not None and _rem > 6.0 and (_fin_best[0] < min(cur_obj, best_obj) - 1e-09):
            try:
                cur = assignments_from_solution(_fin_best[1])
                cur_obj = float(_fin_best[0])
                run_meta['adoptions'].append({'window': int(run_meta['windows']) + 1, 'objective': float(_fin_best[0]), 'elapsed_s': round(float(timelimit) - _rem, 1), 'final_drain': True})
                walked, _ = lns_sa(prob_info, cur, timelimit=_rem - 4.0, seed=seed + 97, verbose=False)
                w_obj = float(_obj_of(prob_info, walked)[0])
                if w_obj < best_obj - 1e-09:
                    sol, _ = retime(prob_info, walked, verbose=False)
                    r = check_feasibility(prob_info, sol)
                    if r['feasible']:
                        best_sol = sol
                        best_obj = float(r['objective'])
            except Exception:
                pass
        run_meta['objective'] = float(best_obj) if best_sol is not None else None
        run_meta['feasible'] = bool(best_sol is not None)
        run_meta['winning_internal_objective'] = float(best_obj) if best_sol is not None else None
        run_meta['emit_checker_verified'] = bool(best_sol is not None)
        q.put(('final', idx, best_sol, float(best_obj) if best_sol is not None else None, run_meta))
    except Exception as e:
        q.put(('final', idx, None, f'{type(e).__name__}: {e}', {'arm': int(idx), 'route': 'relay', 'relay_seat': True, 'error': f'{type(e).__name__}: {e}'}))

def _seed_lns_worker(q, idx, prob_info, timelimit, cfg, seed_starts, portfolio_t0=None):
    worker_started = time.time()
    t_end = worker_started + float(timelimit)
    try:
        os.environ.update(cfg)
        try:
            import numba
            numba.set_num_threads(1)
        except Exception:
            pass
        os.environ['OGC_ARM_THREADS'] = '1'
        from constructor import _objective_of as _obj_of, construct
        from gpr_core import retime
        from lns import lns_sa
        from utils import check_feasibility
        arm_seed = int(cfg.get('OGC_ARM_SEED', 800 + int(idx)))
        run_meta = {'arm': int(idx), 'route': 'seed_lns', 'factory_role': 'seed_lns', 'winning_stage': 'seed_lns', 'seed_dives': []}
        try:
            q.put(('meta', idx, None, None, dict(run_meta)))
        except Exception:
            pass
        _sent = [None]

        def _publish(sol, objective):
            if _sent[0] is not None and objective >= _sent[0] - 1e-09:
                return
            try:
                q.put(('ckpt', idx, sol, float(objective)))
                _sent[0] = float(objective)
            except Exception:
                pass
        dives = [(rank, start) for rank, start in seed_starts or [] if isinstance(start, dict) and start]
        if not dives:
            probe, _ = construct(prob_info, timelimit=max(10.0, min(60.0, t_end - time.time() - 20.0)), verbose=False)
            dives = [(None, probe)]
            run_meta['seed_fallback'] = 'own_probe'
        best_sol, best_obj = (None, 1e309)
        for index, (rank, start) in enumerate(dives):
            budget = (t_end - 2.0 - time.time()) / (len(dives) - index)
            if budget < 10.0:
                break
            dive_started = time.time()
            cur = {int(b): dict(row) for b, row in start.items()}
            cur_obj = float(_obj_of(prob_info, cur)[0])
            try:
                walked, _ = lns_sa(prob_info, cur, timelimit=budget, seed=arm_seed + 131 * index, verbose=False)
                walked_obj = float(_obj_of(prob_info, walked)[0])
            except Exception:
                walked, walked_obj = (cur, cur_obj)
            run_meta['seed_dives'].append({'rank': int(rank) if rank is not None else None, 'seed_objective': cur_obj, 'walk_objective': walked_obj, 'wall_s': round(time.time() - dive_started, 1)})
            if walked_obj < best_obj - 1e-09:
                try:
                    sol, _ = retime(prob_info, walked, verbose=False)
                    checked = check_feasibility(prob_info, sol)
                    if checked['feasible']:
                        best_sol = sol
                        best_obj = float(checked['objective'])
                        _publish(best_sol, best_obj)
                except Exception:
                    pass
        run_meta['objective'] = float(best_obj) if best_sol is not None else None
        run_meta['feasible'] = bool(best_sol is not None)
        run_meta['emit_checker_verified'] = bool(best_sol is not None)
        run_meta['worker_wall_s'] = float(time.time() - worker_started)
        q.put(('final', idx, best_sol, float(best_obj) if best_sol is not None else None, run_meta))
    except Exception as exc:
        q.put(('final', idx, None, f'{type(exc).__name__}: {exc}', {'arm': int(idx), 'route': 'seed_lns', 'factory_role': 'seed_lns', 'error': f'{type(exc).__name__}: {exc}', 'worker_wall_s': float(time.time() - worker_started)}))

def _surplus_worker(q, idx, prob_info, start_solution, timelimit, seed):
    try:
        try:
            import numba
            numba.set_num_threads(1)
        except Exception:
            pass
        os.environ['OGC_ARM_THREADS'] = '1'
        from gpr_core import assignments_from_solution, retime
        from lns import lns_sa
        from utils import check_feasibility
        start = assignments_from_solution(start_solution)
        walked, _ = lns_sa(prob_info, start, timelimit=timelimit, seed=seed, verbose=False)
        sol, _ = retime(prob_info, walked, verbose=False)
        result = check_feasibility(prob_info, sol)
        q.put((idx, sol if result['feasible'] else None, float(result['objective']) if result['feasible'] else None))
    except Exception as exc:
        q.put((idx, None, f'{type(exc).__name__}: {exc}'))

def _surplus_heavy_alap_worker(q, idx, prob_info, start_solution, timelimit, seed, plan_targets=None, profile='heavy'):
    try:
        worker_started = time.time()
        try:
            import numba
            numba.set_num_threads(1)
        except Exception:
            pass
        os.environ['OGC_ARM_THREADS'] = '1'
        from gpr_core import assignments_from_solution, retime
        from heavy_alap import heavy_alap_walk
        from utils import check_feasibility
        profile = 'factory' if str(profile).strip().lower() == 'factory' else 'heavy'
        if profile == 'factory':
            from constructor import _objective_of
            from lns import lns_sa

        def _cfg(name, default, cast):
            return cast(str(default))
        start = assignments_from_solution(start_solution)
        if profile == 'factory':
            start_objective, start_parts = _objective_of(prob_info, start)
            factory_max_obj1 = _cfg('MAX_OBJ1', 2.0, float)
            factory_alap_eligible = bool(float(start_parts[0]) <= factory_max_obj1 + 1e-09)
        else:
            start_objective = start_parts = factory_max_obj1 = None
            factory_alap_eligible = False
        if profile == 'factory' and (not factory_alap_eligible):
            walked, lns_stats = lns_sa(prob_info, start, timelimit=timelimit, seed=int(seed), verbose=False)
            sol, _ = retime(prob_info, walked, verbose=False)
            result = check_feasibility(prob_info, sol)
            stats = {'operator': 'factory_alap_tail', 'tail_profile': 'factory', 'alap_gate': {'eligible': False, 'start_obj1': float(start_parts[0]), 'max_obj1': float(factory_max_obj1), 'fallback': 'seeded_lns'}, 'start_objective': float(start_objective), 'alap_budget_s': 0.0, 'harvest_attempted': True, 'harvest_start_path': 'incumbent', 'harvest_start_objective': float(start_objective), 'harvest_budget_s': float(timelimit), 'harvest_stats': lns_stats, 'harvest_objective': float(_objective_of(prob_info, walked)[0]), 'candidate_objectives': {'incumbent_lns': float(result['objective']) if result['feasible'] else None}, 'selected_path': 'incumbent_lns', 'worker_wall_s': round(time.time() - worker_started, 3), 'emit_checker_verified': bool(result['feasible'])}
            q.put((idx, sol if result['feasible'] else None, float(result['objective']) if result['feasible'] else None, stats))
            return
        geo_cache = {} if profile == 'factory' else None
        pair_memo = {} if profile == 'factory' else None
        rel_memo = {} if profile == 'factory' else None
        if profile == 'factory':
            usable = max(0.0, float(timelimit) - 3.0)
            alap_fraction = min(0.8, max(0.2, _cfg('FRAC', 0.55, float)))
            walk_budget = min(max(0.0, usable - 6.0), max(8.0, usable * alap_fraction))
            default_rounds = 48
        else:
            walk_budget = max(0.0, float(timelimit) - 2.5)
            default_rounds = 12
        protected, stats, terminal = heavy_alap_walk(prob_info, start, timelimit=walk_budget, plan_targets=plan_targets, quality_gap=_cfg('QGAP', 0.08, float), min_batch=_cfg('MIN_BATCH', 8, int), max_batch=_cfg('MAX_BATCH', 12, int), anchors=_cfg('ANCHORS', 6, int), max_rounds=_cfg('ROUNDS', default_rounds, int), max_proposals_per_round=_cfg('PROPOSALS', 12, int), candidate_timelimit=_cfg('CAND_TL', 2.0, float), seed=seed, geo_cache=geo_cache, pair_memo=pair_memo, rel_memo=rel_memo, verbose=False)
        stats['operator'] = 'factory_alap_tail' if profile == 'factory' else 'heavy_alap_path'
        stats['tail_profile'] = profile
        if profile == 'factory':
            stats['alap_gate'] = {'eligible': True, 'start_obj1': float(start_parts[0]), 'max_obj1': float(factory_max_obj1), 'fallback': None}
        if profile == 'factory':
            stats['alap_budget_s'] = round(float(walk_budget), 3)
            protected_objective = float(_objective_of(prob_info, protected)[0])
            terminal_objective = float(_objective_of(prob_info, terminal)[0])
            if int(stats.get('improved_rounds', 0)) > 0 or terminal_objective > protected_objective * 1.02 + 1e-09:
                harvest_start_path = 'alap_protected'
                harvest_start = protected
            else:
                harvest_start_path = 'alap_terminal'
                harvest_start = terminal
            stats['harvest_start_path'] = harvest_start_path
            stats['harvest_start_objective'] = float(_objective_of(prob_info, harvest_start)[0])
            harvest_budget = max(0.0, float(timelimit) - (time.time() - worker_started) - 3.0)
            stats['harvest_budget_s'] = round(harvest_budget, 3)
            stats['harvest_attempted'] = bool(harvest_budget >= 4.0)
            harvested = None
            if stats['harvest_attempted']:
                harvested, harvest_stats = lns_sa(prob_info, harvest_start, timelimit=harvest_budget, geo_cache=geo_cache, pair_memo=pair_memo, rel_memo=rel_memo, seed=int(seed), verbose=False)
                stats['harvest_stats'] = harvest_stats
                stats['harvest_objective'] = float(_objective_of(prob_info, harvested)[0])
            candidates = [('incumbent', start), ('alap_protected', protected)]
            if harvested is not None:
                candidates.append(('terminal_lns', harvested))
            verified = []
            candidate_objectives = {}
            for name, candidate in candidates:
                try:
                    candidate_sol, _ = retime(prob_info, candidate, verbose=False)
                    verdict = check_feasibility(prob_info, candidate_sol)
                except Exception as exc:
                    candidate_objectives[name] = f'{type(exc).__name__}: {exc}'
                    continue
                if not verdict['feasible']:
                    candidate_objectives[name] = None
                    continue
                objective = float(verdict['objective'])
                candidate_objectives[name] = objective
                verified.append((objective, name, candidate_sol))
            if not verified:
                raise RuntimeError('factory ALAP tail produced no verified candidate')
            _, selected_path, sol = min(verified, key=lambda row: (row[0], row[1]))
            result = check_feasibility(prob_info, sol)
            stats['candidate_objectives'] = candidate_objectives
            stats['selected_path'] = selected_path
            stats['worker_wall_s'] = round(time.time() - worker_started, 3)
        else:
            sol, _ = retime(prob_info, protected, verbose=False)
            result = check_feasibility(prob_info, sol)
        stats['emit_checker_verified'] = bool(result['feasible'])
        q.put((idx, sol if result['feasible'] else None, float(result['objective']) if result['feasible'] else None, stats))
    except Exception as exc:
        q.put((idx, None, f'{type(exc).__name__}: {exc}', {'operator': 'factory_alap_tail' if str(profile).strip().lower() == 'factory' else 'heavy_alap_path', 'tail_profile': str(profile), 'error': f'{type(exc).__name__}: {exc}'}))



def _pure_bc_worker(q, idx, prob_info, incumbent_solution, timelimit, seed, perturbation_profile):
    worker_started = time.time()
    try:
        try:
            import numba
            numba.set_num_threads(1)
        except Exception:
            pass
        finally:
            pass
        os.environ['OGC_ARM_THREADS'] = '1'
        from bc_search import qd_lns
        from gpr_core import assignments_from_solution, retime
        from utils import check_feasibility
        protected = check_feasibility(prob_info, incumbent_solution)
        if not protected['feasible']:
            raise ValueError('BC incumbent is not checker-feasible')
        incumbent = assignments_from_solution(incumbent_solution)
        incumbent_objective = float(protected['objective'])
        search_budget = max(3.0, float(timelimit) - 2.0)
        walked, stats = qd_lns(prob_info, incumbent, timelimit=search_budget, reference=incumbent, archive_seeds=[], seed=int(seed), quality_gap=0.12, quality_days=4.0, archive_restart_rate=0.15, perturbation_profile=str(perturbation_profile), verbose=False)
        candidate_solution, _ = retime(prob_info, walked, verbose=False)
        candidate = check_feasibility(prob_info, candidate_solution)
        candidate_objective = float(candidate['objective']) if candidate['feasible'] else None
        if candidate_objective is not None and candidate_objective < incumbent_objective - 1e-09:
            best_solution = candidate_solution
            best_objective = candidate_objective
            selected = 'qd'
        else:
            best_solution = incumbent_solution
            best_objective = incumbent_objective
            selected = 'incumbent'
        q.put((int(idx), best_solution, float(best_objective), {'operator': 'quality_diversity', 'profile': str(perturbation_profile), 'seed': int(seed), 'timelimit_s': float(timelimit), 'search_budget_s': float(search_budget), 'incumbent_objective': float(incumbent_objective), 'candidate_objective': candidate_objective, 'selected': selected, 'stats': stats, 'worker_wall_s': round(time.time() - worker_started, 3), 'emit_checker_verified': True}))
    except Exception as exc:
        q.put((int(idx), None, f'{type(exc).__name__}: {exc}', {'operator': 'quality_diversity', 'profile': str(perturbation_profile), 'seed': int(seed), 'error': f'{type(exc).__name__}: {exc}', 'worker_wall_s': round(time.time() - worker_started, 3), 'emit_checker_verified': False}))
    finally:
        pass

def _pure_bc_round(prob_info, incumbent_solution, incumbent_objective, deadline, profiles=None, seed_base=1701):
    import multiprocessing as mp
    profiles = tuple(profiles or _HEAVY_QD_PERTURBATIONS)
    started = time.time()
    remaining = float(deadline) - started
    meta = {'activated': False, 'policy': 'four quality-diversity lanes; one shared checker-verified incumbent; min complete results only', 'input_objective': float(incumbent_objective), 'output_objective': float(incumbent_objective), 'budget_s': max(0.0, remaining - 3.0), 'wall_s': 0.0, 'completed': 0, 'winner': None, 'lanes': []}
    worker_tl = max(0.0, remaining - 3.0)
    if worker_tl < 8.0 or not profiles:
        return (incumbent_solution, float(incumbent_objective), meta)
    ctx = mp.get_context('fork')
    queue = ctx.Queue()
    processes = []
    seeds = []
    for lane, profile in enumerate(profiles):
        seed = int(seed_base + lane)
        process = ctx.Process(target=_pure_bc_worker, args=(queue, lane, prob_info, incumbent_solution, worker_tl, seed, profile), daemon=True)
        process.start()
        processes.append(process)
        seeds.append(seed)
    outputs = {}
    while len(outputs) < len(processes) and time.time() < float(deadline):
        try:
            item = queue.get(timeout=max(0.1, min(1.0, float(deadline) - time.time())))
            lane, solution, objective = item[:3]
            worker_meta = item[3] if len(item) > 3 and isinstance(item[3], dict) else None
            outputs[int(lane)] = (solution, objective, worker_meta)
        except Exception:
            pass
    for process in processes:
        if process.is_alive():
            process.terminate()
        process.join(timeout=0.5)
    best_solution = incumbent_solution
    best_objective = float(incumbent_objective)
    winner = None
    for lane in sorted(outputs):
        solution, objective, _ = outputs[lane]
        if solution is not None and isinstance(objective, float) and (objective < best_objective - 1e-09):
            best_solution = solution
            best_objective = float(objective)
            winner = int(lane)
    meta.update({'activated': True, 'output_objective': float(best_objective), 'wall_s': float(time.time() - started), 'completed': int(len(outputs)), 'winner': None if winner is None else {'lane': winner, 'profile': str(profiles[winner]), 'seed': int(seeds[winner])}, 'lanes': [{'lane': int(lane), 'profile': str(profile), 'seed': int(seeds[lane]), 'completed': bool(lane in outputs), 'objective': float(outputs[lane][1]) if lane in outputs and isinstance(outputs[lane][1], float) else None, 'error': str(outputs[lane][1]) if lane in outputs and isinstance(outputs[lane][1], str) else None, 'worker_meta': outputs[lane][2] if lane in outputs else None} for lane, profile in enumerate(profiles)]})
    return (best_solution, best_objective, meta)



def _production_qd_lns_tail(prob_info, incumbent_solution, incumbent_objective, deadline, qd_fraction=0.25, qd_seed_base=1701, lns_seed_base=701):
    started = time.time()
    final_deadline = float(deadline)
    initial_objective = float(incumbent_objective)
    best_solution = incumbent_solution
    best_objective = initial_objective
    fraction = min(0.5, max(0.1, float(qd_fraction)))
    stages = []
    remaining = max(0.0, final_deadline - time.time())
    qd_window = min(max(0.0, remaining - 12.0), max(0.0, remaining * fraction))
    qd_before = float(best_objective)
    if qd_window >= 11.0:
        qd_deadline = min(final_deadline - 12.0, time.time() + qd_window)
        best_solution, best_objective, qd_meta = _pure_bc_round(prob_info, best_solution, best_objective, deadline=qd_deadline, seed_base=int(qd_seed_base))
    else:
        qd_meta = {'activated': False, 'policy': 'four non-strict quality-diversity lanes', 'input_objective': qd_before, 'output_objective': qd_before, 'budget_s': float(qd_window), 'wall_s': 0.0, 'completed': 0, 'winner': None, 'lanes': [], 'skip_reason': 'protect_lns_reserve'}
    stages.append({'stage': 1, 'operator': 'quality_diversity', 'activated': bool(qd_meta.get('activated')), 'input_objective': qd_before, 'output_objective': float(best_objective), 'improved': bool(best_objective < qd_before - 1e-09), 'detail': qd_meta})
    lns_before = float(best_objective)
    best_solution, best_objective, lns_meta = _heavy_closer_round(prob_info, best_solution, best_objective, deadline=final_deadline, seed_base=int(lns_seed_base), kinds=('deep_lns',) * 4)
    stages.append({'stage': 2, 'operator': 'lns', 'activated': bool(lns_meta.get('activated')), 'input_objective': lns_before, 'output_objective': float(best_objective), 'improved': bool(best_objective < lns_before - 1e-09), 'detail': lns_meta})
    return (best_solution, best_objective, {'enabled': True, 'activated': any((bool(stage['activated']) for stage in stages)), 'policy': 'ordinary heavy LR rank-0 -> strict parent reswap -> short non-strict four-profile quality-diversity min -> four-way deep-LNS min; checker-verified incumbent protected at every boundary', 'qd_fraction': float(fraction), 'input_objective': initial_objective, 'output_objective': float(best_objective), 'wall_s': float(time.time() - started), 'stages': stages})

def _heavy_closer_worker(q, idx, prob_info, incumbent_solution, timelimit, seed, kind, trace_origin):
    worker_started = time.time()
    try:
        try:
            import numba
            numba.set_num_threads(1)
        except Exception:
            pass
        os.environ['OGC_ARM_THREADS'] = '1'
        from constructor import _objective_of
        from gpr_core import assignments_from_solution, retime
        from lns import lns_sa
        from utils import check_feasibility
        kind = str(kind)
        deadline = worker_started + max(0.0, float(timelimit))
        trace = []

        def _record(event, objective, objective_kind, source=None, selected=None):
            trace.append({'elapsed_s': round(max(0.0, time.time() - float(trace_origin)), 3), 'event': str(event), 'source': str(source) if source is not None else None, 'objective': float(objective), 'objective_kind': str(objective_kind), 'selected': str(selected) if selected is not None else None})
        protected = check_feasibility(prob_info, incumbent_solution)
        if not protected['feasible']:
            raise RuntimeError('closer incumbent is infeasible')
        protected_objective = float(protected['objective'])
        _record('start', protected_objective, 'checker_verified', source='incumbent')
        start = assignments_from_solution(incumbent_solution)
        lns_start = start
        candidates = [('incumbent', start)]
        geo_cache = {}
        pair_memo = {}
        rel_memo = {}
        prefix_budget = 0.0
        prefix_meta = None

        def _prefix_improvement(_assignments, objective, event):
            _record(event, objective, 'assignment_objective', source=kind)
        usable = max(0.0, float(timelimit) - 3.0)
        if kind == 'reswap_lns' and usable >= 16.0:
            from reswap import reswap
            prefix_budget = min(24.0, max(8.0, 0.3 * usable), max(0.0, usable - 8.0))
            reswap_best, prefix_meta, reswap_terminal = reswap(prob_info, start, timelimit=prefix_budget, geo_cache=geo_cache, pair_memo=pair_memo, rel_memo=rel_memo, parallel=False, quality_gap=0.0, return_terminal=True, improvement_cb=_prefix_improvement, verbose=False)
            candidates.extend((('reswap_protected', reswap_best), ('reswap_terminal', reswap_terminal)))
            lns_start = reswap_best
            _record('reswap_complete', float(_objective_of(prob_info, reswap_best)[0]), 'assignment_objective', source='reswap')
        elif kind == 'heavy_alap_lns' and usable >= 16.0:
            from heavy_alap import heavy_alap_walk
            prefix_budget = min(24.0, max(8.0, 0.3 * usable), max(0.0, usable - 8.0))
            alap_best, prefix_meta, alap_terminal = heavy_alap_walk(prob_info, start, timelimit=prefix_budget, quality_gap=0.08, min_batch=8, max_batch=12, anchors=6, max_rounds=12, max_proposals_per_round=12, candidate_timelimit=2.0, seed=int(seed), geo_cache=geo_cache, pair_memo=pair_memo, rel_memo=rel_memo, improvement_cb=_prefix_improvement, verbose=False)
            candidates.extend((('heavy_alap_protected', alap_best), ('heavy_alap_terminal', alap_terminal)))
            lns_start = alap_terminal
            _record('heavy_alap_complete', float(_objective_of(prob_info, alap_terminal)[0]), 'assignment_objective', source='heavy_alap')

        def _lns_improvement(_assignments, objective, event):
            _record(f'lns:{event}', objective, 'assignment_objective', source=kind)
        lns_budget = max(0.0, deadline - time.time() - 2.5)
        lns_stats = None
        if lns_budget >= 4.0:
            walked, lns_stats = lns_sa(prob_info, lns_start, timelimit=lns_budget, geo_cache=geo_cache, pair_memo=pair_memo, rel_memo=rel_memo, seed=int(seed), improvement_cb=_lns_improvement, verbose=False)
            candidates.append(('lns', walked))
        best_solution = incumbent_solution
        best_objective = protected_objective
        selected = 'incumbent'
        checked = [{'label': 'incumbent', 'objective': protected_objective}]
        for label, candidate in candidates[1:]:
            try:
                solution, _ = retime(prob_info, candidate, verbose=False)
                verdict = check_feasibility(prob_info, solution)
                objective = float(verdict['objective']) if verdict['feasible'] else None
                checked.append({'label': str(label), 'objective': objective})
                if objective is not None and objective < best_objective - 1e-09:
                    best_solution = solution
                    best_objective = objective
                    selected = str(label)
            except Exception as exc:
                checked.append({'label': str(label), 'objective': None, 'error': f'{type(exc).__name__}: {exc}'})
        _record('final_checked', best_objective, 'checker_verified', source=kind, selected=selected)
        q.put((int(idx), best_solution, float(best_objective), {'operator': 'heavy_closer', 'kind': kind, 'seed': int(seed), 'timelimit_s': float(timelimit), 'prefix_budget_s': float(prefix_budget), 'lns_budget_s': float(lns_budget), 'selected': selected, 'checked': checked, 'prefix': prefix_meta, 'lns': lns_stats, 'trace': trace, 'worker_wall_s': round(time.time() - worker_started, 3), 'emit_checker_verified': True}))
    except Exception as exc:
        q.put((int(idx), None, f'{type(exc).__name__}: {exc}', {'operator': 'heavy_closer', 'kind': str(kind), 'seed': int(seed), 'error': f'{type(exc).__name__}: {exc}', 'worker_wall_s': round(time.time() - worker_started, 3), 'emit_checker_verified': False}))

def _heavy_closer_round(prob_info, incumbent_solution, incumbent_objective, deadline, seed_base=701, kinds=None):
    import multiprocessing as mp
    kinds = tuple(kinds or ('deep_lns', 'deep_lns', 'reswap_lns', 'heavy_alap_lns'))
    if not kinds:
        raise ValueError('closer requires at least one lane')
    started = time.time()
    remaining = float(deadline) - started
    meta = {'activated': False, 'policy': f'{len(kinds)} protected lanes {list(kinds)}; checker-verified min over complete workers only', 'input_objective': float(incumbent_objective), 'budget_s': max(0.0, remaining - 3.0), 'wall_s': 0.0, 'completed': 0, 'winner': None, 'output_objective': float(incumbent_objective), 'lanes': [], 'candidate_events': [], 'candidate_envelope': [], 'verified_envelope': []}
    worker_tl = max(0.0, remaining - 3.0)
    if worker_tl < 12.0:
        return (incumbent_solution, float(incumbent_objective), meta)
    ctx = mp.get_context('fork')
    queue = ctx.Queue()
    processes = []
    seeds = []
    for lane, kind in enumerate(kinds):
        seed = int(seed_base + lane)
        process = ctx.Process(target=_heavy_closer_worker, args=(queue, lane, prob_info, incumbent_solution, worker_tl, seed, kind, started), daemon=True)
        process.start()
        processes.append(process)
        seeds.append(seed)
    outputs = {}
    while len(outputs) < len(processes) and time.time() < float(deadline):
        try:
            item = queue.get(timeout=max(0.1, min(1.0, float(deadline) - time.time())))
            lane, solution, objective = item[:3]
            worker_meta = item[3] if len(item) > 3 and isinstance(item[3], dict) else None
            outputs[int(lane)] = (solution, objective, worker_meta)
        except Exception:
            pass
    for process in processes:
        if process.is_alive():
            process.terminate()
        process.join(timeout=0.5)
    best_solution = incumbent_solution
    best_objective = float(incumbent_objective)
    winner = None
    for lane in sorted(outputs):
        solution, objective, _ = outputs[lane]
        if solution is not None and isinstance(objective, float) and (objective < best_objective - 1e-09):
            best_solution = solution
            best_objective = float(objective)
            winner = int(lane)
    candidate_events = []
    verified_events = [{'elapsed_s': 0.0, 'lane': None, 'kind': 'incumbent', 'event': 'start', 'objective': float(incumbent_objective)}]
    for lane in sorted(outputs):
        worker_meta = outputs[lane][2]
        if not isinstance(worker_meta, dict):
            continue
        for event in worker_meta.get('trace', []):
            row = dict(event)
            row.update({'lane': int(lane), 'kind': str(kinds[lane])})
            candidate_events.append(row)
            if row.get('objective_kind') == 'checker_verified':
                verified_events.append({'elapsed_s': float(row['elapsed_s']), 'lane': int(lane), 'kind': str(kinds[lane]), 'event': str(row['event']), 'objective': float(row['objective'])})
    candidate_events.sort(key=lambda row: (float(row['elapsed_s']), int(row['lane'])))
    verified_events.sort(key=lambda row: (float(row['elapsed_s']), -1 if row['lane'] is None else int(row['lane'])))

    def _envelope(events):
        rows = []
        value = 1e309
        for event in events:
            objective = event.get('objective')
            if isinstance(objective, (int, float)) and float(objective) < value - 1e-09:
                value = float(objective)
                rows.append(dict(event))
        return rows
    meta.update({'activated': True, 'wall_s': float(time.time() - started), 'completed': int(len(outputs)), 'winner': None if winner is None else {'lane': winner, 'kind': str(kinds[winner]), 'seed': int(seeds[winner])}, 'output_objective': float(best_objective), 'lanes': [{'lane': int(lane), 'kind': str(kinds[lane]), 'seed': int(seeds[lane]), 'completed': bool(lane in outputs), 'objective': float(outputs[lane][1]) if lane in outputs and isinstance(outputs[lane][1], float) else None, 'error': str(outputs[lane][1]) if lane in outputs and isinstance(outputs[lane][1], str) else None, 'worker_meta': outputs[lane][2] if lane in outputs else None} for lane in range(len(kinds))], 'candidate_events': candidate_events, 'candidate_envelope': _envelope(candidate_events), 'verified_envelope': _envelope(verified_events)})
    return (best_solution, best_objective, meta)



def _solution_fingerprint(solution):
    import hashlib
    import json
    payload = json.dumps(solution, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.blake2b(payload, digest_size=8, person=b'OGC-parent-seed').hexdigest()





def _parent_plan_targets_for_arm(arm_meta, arm_idx):
    try:
        meta = arm_meta.get(int(arm_idx), {})
    except Exception:
        return None
    targets = meta.get('_parent_plan_targets')
    if isinstance(targets, dict) and targets:
        return targets
    source = meta.get('relay_source')
    try:
        source_idx = int(source)
    except (TypeError, ValueError):
        return None
    source_meta = arm_meta.get(source_idx, {})
    targets = source_meta.get('_parent_plan_targets')
    return targets if isinstance(targets, dict) and targets else None


def _factory_alap_tail_sources(best, best_obj, arm_meta, slots=4):
    slots = max(1, int(slots))
    sources = [('tail', best, float(best_obj)) for _ in range(slots)]
    plans = [None] * slots
    if slots >= 2:
        sources[-1] = ('factory-alap-B', best, float(best_obj))
        plans[-1] = _parent_plan_targets_for_arm(arm_meta, 0)
    return (sources, plans)

_LIGHT_V606_ROBUST_ARM_SPECS = ((2, 0), (2, 1), (8, 0), (8, 1))
_LIGHT_GAMBLE_ARM_SPECS = ((2, 0), (8, 3), (8, 6), (8, 9))

def _apply_light_alap_arm_specs(arms, specs):
    if len(arms) != 4 or len(specs) != 4:
        raise ValueError('light ALAP arm menu must contain four seats')
    for arm, (top_b, rank) in zip(arms, specs):
        arm['OGC_ARM_SEED'] = str(int(rank))
        arm['OGC_MSEED_RANK'] = str(int(rank))
        arm['OGC_TOP_B'] = str(int(top_b))
    return arms

def _nonheavy_portfolio_arm_configs(common, default_menu='auto4', n_blocks=None, timelimit=None):
    menu = default_menu

    def _auto4():
        return [dict(common, OGC_ARM='0', OGC_FORCE_ROUTE='light'), dict(common, OGC_ARM='1', OGC_ARM_SEED='1', OGC_ARM_SAMPLER='1', OGC_NO_GHAT='1', OGC_BOTH_SPLIT='0.80'), dict(common, OGC_ARM='2', OGC_ARM_SEED='2', OGC_PUP_FORCE_LIGHT='1', OGC_BOTH_SPLIT='0.35'), dict(common, OGC_ARM='3', OGC_ARM_SEED='3', OGC_BOTH_SPLIT='0.15')]
    if menu == 'auto4':
        return _auto4()
    if menu == 'pbbb':
        arms = _auto4()
        primal = _heavy_factory_arm(common, arm_index=0, aeff_scale=1.0, profile='primal-random')
        primal.update({'OGC_BOTH_SPLIT': '0.80', 'OGC_PRIMAL_DIRECT_BOTH': '1', 'OGC_PRIMAL_JITTER_DAYS': '4', 'OGC_PRIMAL_SEED': '223', 'OGC_PRIMAL_WINDOW': '9', 'OGC_PRIMAL_BAY_WEIGHT': '0.3'})
        arms[0] = primal
        arms[1] = dict(common, OGC_ARM='1', OGC_ARM_SEED='1', OGC_PORTFOLIO_ROLE='incumbent_lns', OGC_RELAY_LATE='1', OGC_RELAY_START_FRAC='0.45', OGC_RELAY_WINDOW='15')
        return arms
    if menu == 'factory4':
        _factory_a3 = dict(common, OGC_ARM='3', OGC_ARM_SEED='3', OGC_FACTORY_PRODUCER='0', OGC_FACTORY_ROLE='bucket_lns', OGC_FORCE_ROUTE='light', OGC_FACTORY_PLAN_BUCKET='5')
        _factory_a2_share = '0.70' if float(240.0 if timelimit is None else timelimit) >= 255.0 else '0.25'
        return [dict(common, OGC_ARM='0', OGC_ARM_SEED='0', OGC_ARM_SAMPLER='1', OGC_NO_GHAT='1', OGC_FORCE_ROUTE='both', OGC_WHO_WAITS='1', OGC_FACTORY_PRODUCER='1', OGC_FACTORY_ROLE=''), dict(common, OGC_ARM='1', OGC_ARM_SEED='1', OGC_FACTORY_PRODUCER='0', OGC_FACTORY_ROLE='incumbent_lns', OGC_RELAY_START_FRAC='0.45'), dict(common, OGC_ARM='2', OGC_ARM_SEED='2', OGC_FACTORY_PRODUCER='0', OGC_FACTORY_ROLE='', OGC_FORCE_ROUTE='both', OGC_BOTH_ROUNDS='2', OGC_MULTI_SHARE=_factory_a2_share, OGC_FACTORY_A2_SHARE_POLICY='timelimit_gate', OGC_FACTORY_A2_SHARE_GATE_TL='255.0'), _factory_a3]
    if menu == 'alap_mixed4':
        _arm_refill = _arm_refill_enabled(menu)
        _r2_rounds = '256' if _arm_refill else '84'
        _r3_rounds = '256' if _arm_refill else '64'
        _arms = []
        for _idx, (_top_b, _rank) in enumerate(_LIGHT_GAMBLE_ARM_SPECS):
            _arms.append(dict(common, OGC_ARM=str(_idx), OGC_ARM_SEED=str(_rank), OGC_MSEED_RANK=str(_rank), OGC_FORCE_ROUTE='light', OGC_LIGHT_PROFILE='alap_batch', OGC_TOP_B=str(_top_b), OGC_ALAP_NS_BATCH='10', OGC_ALAP_NS_QGAP='0.20', OGC_ALAP_NS_ROUNDS=_r3_rounds, OGC_ALAP_NS_PROPOSALS='96', OGC_ALAP_NS_CAND_TL='3', OGC_ALAP_NS_LATEST='1', OGC_ALAP_NS_REPAIR='construct', OGC_ALAP_NS_CPSAT_DET='1', OGC_ALAP_NS_CPSAT_WORKERS='1'))
        return _arms
    return _auto4()



def _structured_primal_priority_bias(prob_info, jitter_days, seed, window=9, bay_weight=0.3):
    import hashlib
    import numpy as np
    blocks = prob_info.get('blocks', [])
    bays = prob_info.get('bays', [])
    n_blocks = len(blocks)
    n_bays = max(1, len(bays))
    strength = float(jitter_days)
    if strength < 0.0:
        raise ValueError('primal priority jitter must be nonnegative')
    if not 0.0 <= float(bay_weight) <= 1.0:
        raise ValueError('primal priority bay_weight must be in [0, 1]')
    latest = np.array([int(block.get('due_date', 0)) - int(block.get('processing_time', 0)) for block in blocks], dtype=np.int64)
    horizon = max(1, int(latest.max()) + 1) if latest.size else 1
    width = max(1, min(int(window), horizon))
    if width > 1 and width % 2 == 0:
        width -= 1
    radius = width // 2
    kernel = np.arange(1, radius + 2, dtype=float)
    kernel = np.concatenate((kernel, kernel[-2::-1]))
    kernel /= float(kernel.sum())
    rng = np.random.RandomState(int(seed))
    time_field = np.convolve(rng.standard_normal(horizon), kernel, mode='same')
    bay_field = rng.standard_normal(n_bays)
    local_field = rng.standard_normal(n_blocks)
    preferred = []
    for block in blocks:
        prefs = block.get('bay_preferences', [])
        preferred.append(int(np.argmax(prefs)) if len(prefs) else 0)
    preferred = np.asarray(preferred, dtype=np.int64)
    anchors = np.clip(latest, 0, horizon - 1)
    raw = (1.0 - float(bay_weight)) * time_field[anchors] + float(bay_weight) * bay_field[np.clip(preferred, 0, n_bays - 1)] + 0.2 * local_field
    if raw.size:
        raw -= float(raw.mean())
        rms = float(np.sqrt(np.mean(raw * raw)))
        if rms > 1e-12:
            raw /= rms
    bias = strength * raw
    diag = {'enabled': bool(strength > 0.0 and n_blocks > 0), 'policy': 'fixed-seed smooth latest-feasible-time plus preferred-bay priority field; LR plan and lambda updates unchanged; same field reused by core and extension realizations', 'jitter_days': strength, 'seed': int(seed), 'window': int(width), 'bay_weight': float(bay_weight), 'blocks': int(n_blocks), 'bias_rms_days': float(np.sqrt(np.mean(bias * bias))) if bias.size else 0.0, 'bias_min_days': float(bias.min()) if bias.size else 0.0, 'bias_max_days': float(bias.max()) if bias.size else 0.0, 'bias_hash': hashlib.blake2b(np.asarray(bias, dtype=np.float64).tobytes(), digest_size=8, person=b'OGC-primal-bias').hexdigest()}
    return (bias, diag)

def _primal_realization_order(prob_info, plan, bias, plan_bucket=0):
    import numpy as np
    blocks = prob_info.get('blocks', [])
    n_blocks = len(blocks)
    offset = np.asarray(bias, dtype=float)
    if offset.shape != (n_blocks,):
        raise ValueError('primal priority bias must have one value per block')
    bucket = max(0, int(plan_bucket))

    def _bbox_area(block_id):
        vertices = [vertex for layer in blocks[block_id]['shape'][0]['layers'] for vertex in layer]
        if not vertices:
            return 0.0
        xs = [vertex[0] for vertex in vertices]
        ys = [vertex[1] for vertex in vertices]
        return (max(xs) - min(xs)) * (max(ys) - min(ys))

    def _key(block_id, use_bias):
        row = plan.get(block_id, {})
        start = float(row.get('start', 10 ** 9))
        if use_bias:
            start += float(offset[block_id])
        if bucket > 0:
            return (int(np.floor(start / bucket)), -_bbox_area(block_id), blocks[block_id]['due_date'], blocks[block_id]['release_time'])
        return (start, blocks[block_id]['due_date'], blocks[block_id]['release_time'])
    baseline = sorted(range(n_blocks), key=lambda block_id: _key(block_id, False))
    order = sorted(range(n_blocks), key=lambda block_id: _key(block_id, True))
    baseline_rank = {block_id: rank for rank, block_id in enumerate(baseline)}
    shifts = np.asarray([abs(rank - baseline_rank[block_id]) for rank, block_id in enumerate(order)], dtype=float)
    diag = {'bucket': int(bucket), 'moved_blocks': int(np.count_nonzero(shifts)), 'moved_fraction': float(np.count_nonzero(shifts)) / max(1, n_blocks), 'mean_abs_rank_shift': float(shifts.mean()) if shifts.size else 0.0, 'max_abs_rank_shift': float(shifts.max()) if shifts.size else 0.0}
    return (order, diag)

def _who_waits_locked_problem(prob_info, plan):
    blocks = []
    for block_id, block in enumerate(prob_info.get('blocks', [])):
        target = plan.get(block_id)
        if target is not None and int(target['start']) > int(block['release_time']):
            block = dict(block)
            block['release_time'] = int(target['start'])
        blocks.append(block)
    locked = dict(prob_info)
    locked['blocks'] = blocks
    return locked
_HEAVY_FACTORY_RESET_KEYS = ('OGC_PLAN_BUCKET', 'OGC_AEFF_SCALE', 'OGC_NO_KARM', 'OGC_NO_KARM_HI', 'OGC_KARM', 'OGC_NO_HEAVY_MS', 'OGC_RC_SHADOW', 'OGC_PRIMAL_REALIZATION', 'OGC_PRIMAL_JITTER_DAYS', 'OGC_PRIMAL_SEED', 'OGC_PRIMAL_WINDOW', 'OGC_PRIMAL_BAY_WEIGHT', 'OGC_PRIMAL_DIRECT_BOTH', 'OGC_ARM_SEED', 'OGC_HEAVY_RELAY_CONSUMER', 'OGC_ARM_SAMPLER', 'OGC_NO_GHAT', 'OGC_HEAVY_FACTORY_SAMPLER', 'OGC_HEAVY_FACTORY_PROFILE')

def _heavy_factory_arm(base, arm_index, aeff_scale, profile):
    arm = dict(base)
    for key in _HEAVY_FACTORY_RESET_KEYS:
        arm.pop(key, None)
    arm.update({'OGC_ARM': str(int(arm_index)), 'OGC_ARM_SAMPLER': '1', 'OGC_NO_GHAT': '1', 'OGC_HEAVY_FACTORY_SAMPLER': '1', 'OGC_AEFF_SCALE': f'{float(aeff_scale):.6g}', 'OGC_HEAVY_FACTORY_PROFILE': str(profile)})
    if profile == 'cold':
        arm['OGC_NO_KARM'] = '1'
    elif profile == 'primal-random':
        arm['OGC_PRIMAL_REALIZATION'] = '1'
    elif profile != 'kappa':
        raise ValueError(f'unknown heavy factory profile: {profile}')
    return arm

def _heavy_portfolio_arm_configs(arms, menu=None):
    resolved = str(menu) if menu is not None else 'current'
    configured = [dict(arm) for arm in arms]
    if resolved == 'lr_qd_lns4':
        for arm in configured:
            arm['OGC_PRIMAL_REALIZATION'] = '0'
            arm.pop('OGC_PRIMAL_JITTER_DAYS', None)
            arm.pop('OGC_PRIMAL_SEED', None)
            arm.pop('OGC_PRIMAL_WINDOW', None)
            arm.pop('OGC_PRIMAL_BAY_WEIGHT', None)
        return configured
    return configured
_HEAVY_QD_PERTURBATIONS = ('tardy_small', 'tardy_big', 'slab', 'random')

def _default_nonheavy_menu(plan_region, arm_dr):
    if not plan_region and float(arm_dr) < 0.4:
        return 'alap_mixed4'
    if not plan_region:
        return 'factory4'
    if float(arm_dr) < 0.7:
        return 'pbbb'
    return 'auto4'

def _parent_stripped_alap_menu(plan_region, menu):
    return bool(not plan_region and menu == 'alap_mixed4')

def _parent_factory_band(menu):
    return str(menu) == 'factory4'

def _factory_seat_plan(arms):
    lock_consumers, cross_seats, seed_seats = ({}, {}, {})
    for seat in range(1, len(arms)):
        role = arms[seat].get('OGC_PORTFOLIO_ROLE') or arms[seat].get('OGC_FACTORY_ROLE', '')
        if role in ('deep_lns', 'bucket_lns'):
            lock_consumers[seat] = role
        elif role == 'incumbent_lns':
            cross_seats[seat] = None
        elif role == 'seed_lns':
            seed_seats[seat] = tuple((int(token) for token in arms[seat].get('OGC_SEED_LNS_RANKS', '').split(',') if token.strip()))
    return (lock_consumers, cross_seats, seed_seats)

def _parallel_portfolio(prob_info, timelimit, plan_region=False):
    import multiprocessing as mp
    t0 = time.time()
    _forecast_raw_events = []
    _arm_curve_raw_events = []

    def _forecast_offer(event, arm, solution, objective, objective_kind):
        pass

    def _arm_curve_offer(event, arm, solution, objective, objective_kind, representation, meta=None):
        return
    _common = {'OGC_ARM_THREADS': '1', 'OGC_WORK_SCALE': '1.15'}
    from plan_core import demand_ratio as _arm_dratio
    _arm_dr = _arm_dratio(prob_info)
    _heavy_production_qd_tail = bool(_arm_dr >= 0.7)
    _menu_resolved = None
    _heavy_menu_resolved = None
    _arm_refill_active = False
    if _arm_dr >= 0.7:
        _heavy_menu_resolved = 'lr_qd_lns4'
        _hcommon = dict(_common)
        if _rust_enabled():
            _hcommon['OGC_WORK_SCALE'] = '0.90'
        _base_arm = dict(_hcommon, OGC_ARM='0')
        _cold_arm = dict(_hcommon, OGC_ARM='3', OGC_ARM_SEED='1')
        if _arm_dr >= 0.7 and True:
            _cold_arm['OGC_NO_KARM'] = '1'
        _rc_arm = dict(_hcommon, OGC_ARM='2', OGC_AEFF_SCALE='0.92')
        arms = [_base_arm, dict(_hcommon, OGC_ARM='1', OGC_PLAN_BUCKET='5'), _rc_arm, _cold_arm]
        arms = _heavy_portfolio_arm_configs(arms, menu=_heavy_menu_resolved)
    else:
        _ncommon = dict(_common)
        if plan_region and _rust_enabled():
            _ncommon['OGC_WORK_SCALE'] = '0.90'
        _menu_default = _default_nonheavy_menu(plan_region, _arm_dr)
        _menu_resolved = _menu_default
        _arm_refill_active = _arm_refill_enabled(_menu_resolved)
        if _arm_refill_active:
            _ncommon['OGC_ARM_FINAL_MARGIN'] = '8'
        arms = _nonheavy_portfolio_arm_configs(_ncommon, default_menu=_menu_default, timelimit=timelimit)
    arms = arms[:4]
    _stripped_alap_menu = _parent_stripped_alap_menu(plan_region, _menu_resolved)
    _tail_res_configured = _parent_tail_reserve_configured_s(plan_region, _arm_dr)
    _tail_res_returned = _stripped_alap_menu
    _tail_res = 0.0 if _stripped_alap_menu else _parent_tail_reserve_s(plan_region, _arm_dr)
    ctx = mp.get_context('fork')
    q = ctx.Queue()
    procs = []
    _factory_roles, _relay_seats, _seed_seats = _factory_seat_plan(arms)
    _role_flow_enabled = bool(arms and arms[0].get('OGC_FACTORY_PRODUCER', '') == '1' and (_factory_roles or _relay_seats))
    _mseed_on = not plan_region and (_menu_resolved != 'factory4' or bool(_seed_seats))
    _mseed_splan = 0.0
    _mseed_wall_s = 0.0
    _mseed_work = None
    _mseed_starts = [None] * len(arms)
    _factory_seed_picks = []
    _mseed_ranks = [int(cfg.get('OGC_MSEED_RANK', -1 if _menu_resolved == 'factory4' else idx)) for idx, cfg in enumerate(arms)]
    _mseed_pick_count = max((rank for ranks in _seed_seats.values() for rank in ranks), default=-1) + 1 if _menu_resolved == 'factory4' else max(_mseed_ranks) + 1 if _mseed_ranks else 0
    _mseed_pick_diag = {'mode': 'objective', 'unique': 0, 'eligible': 0, 'selected': []}
    if _mseed_on:
        _mseed_started = time.time()
        _sprocs = []
        _scout_stop = None
        try:
            _sctx = mp.get_context('fork')
            _sq = _sctx.Queue()
            _pool = []
            if _menu_resolved in ('alap_mixed4', 'factory4'):
                _n_star = 256
                _mseed_splan = 10.0
                _mseed_work = {'scheduler': 'unique_or_wall', 'unique_target': 24, 'wall_cap_seconds': 10.0, 'draw_cap': _n_star, 'planned_seconds': 10.0, 'target_seconds': 10.0}
                _scout_stop = _sctx.Event()
                for _w in range(4):
                    _sp = _sctx.Process(target=_scout_worker, args=(_sq, _w, prob_info, _n_star, 4), kwargs={'stream': True, 'stop_event': _scout_stop}, daemon=True)
                    _sp.start()
                    _sprocs.append(_sp)
                _seen_hashes = set()
                _raw_completed = 0
                _workers_done = 0
                _sdl = _mseed_started + 10.0
                while _workers_done < 4 and len(_seen_hashes) < 24 and (time.time() < _sdl):
                    try:
                        _remaining = _sdl - time.time()
                        if _remaining <= 0.0:
                            break
                        _message = _sq.get(timeout=max(0.01, min(0.25, _remaining)))
                        _kind = _message[0]
                        if _kind == 'candidate':
                            _, _w, _i, _o, _h, _a = _message
                            _raw_completed += 1
                            if _h in _seen_hashes:
                                continue
                            _seen_hashes.add(_h)
                            _pool.append((_o, _w, _i, _h, _a))
                        elif _kind in ('done', 'error'):
                            _workers_done += 1
                    except Exception:
                        pass
                    finally:
                        pass
                _scout_stop.set()
                _mseed_work.update({'raw_completed': int(_raw_completed), 'unique_completed': int(len(_seen_hashes)), 'target_reached': bool(len(_seen_hashes) >= 24), 'workers_done': int(_workers_done)})
            else:
                _mseed_work = _mseed_work_plan(prob_info, timelimit, 'objective', default_seconds=None)
                _n_star = _mseed_work['draws']
                _mseed_splan = _mseed_work['planned_seconds']
                for _w in range(4):
                    _sp = _sctx.Process(target=_scout_worker, args=(_sq, _w, prob_info, _n_star, 4), daemon=True)
                    _sp.start()
                    _sprocs.append(_sp)
                _sdl = time.time() + 3.0 * _mseed_splan + 20.0
                _got = 0
                while _got < 4 and time.time() < _sdl:
                    try:
                        _w, _rows, _tops = _sq.get(timeout=max(0.1, min(2.0, _sdl - time.time())))
                        _got += 1
                        for _o, _i, _h, _a in _tops:
                            _pool.append((_o, _w, _i, _h, _a))
                    except Exception:
                        pass
                    finally:
                        pass
            for _sp in _sprocs:
                if _sp.is_alive():
                    _sp.terminate()
            for _sp in _sprocs:
                _sp.join(timeout=0.05)
            _picks, _pick_diag = _select_mseed_starts(prob_info, _pool, _mseed_pick_count, mode='objective')
            if _menu_resolved == 'alap_mixed4' and _pick_diag['unique'] < 10:
                _gamble_unique = int(_pick_diag['unique'])
                _apply_light_alap_arm_specs(arms, _LIGHT_V606_ROBUST_ARM_SPECS)
                _mseed_ranks = [int(cfg['OGC_MSEED_RANK']) for cfg in arms]
                _mseed_pick_count = max(_mseed_ranks) + 1
                _picks, _pick_diag = _select_mseed_starts(prob_info, _pool, _mseed_pick_count, mode='objective')
                _pick_diag.update({'fallback': 'v6.0.6_robust', 'gamble_unique_observed': _gamble_unique, 'gamble_unique_required': 10})
            elif _menu_resolved == 'alap_mixed4':
                _pick_diag.update({'fallback': None, 'gamble_unique_observed': int(_pick_diag['unique']), 'gamble_unique_required': 10})
            _mseed_pick_diag = _pick_diag
            if _menu_resolved == 'factory4':
                _factory_seed_picks = list(_picks)
            for _arm_idx, _rank in enumerate(_mseed_ranks):
                if 0 <= _rank < len(_picks):
                    _mseed_starts[_arm_idx] = _picks[_rank]
        except Exception as _me:
            if _scout_stop is not None:
                _scout_stop.set()
            for _sp in _sprocs:
                if _sp.is_alive():
                    _sp.terminate()
            for _sp in _sprocs:
                _sp.join(timeout=0.05)
            _mseed_splan = 0.0
            _mseed_starts = [None] * len(arms)
        finally:
            _mseed_wall_s = float(time.time() - _mseed_started)
    _base_arm_tl = max(15.0, timelimit - 12.0 - _tail_res - _mseed_splan)
    _refill_plan = _arm_refill_plan(timelimit, time.time() - t0, _base_arm_tl, enabled=_arm_refill_active, cap=30.0, quantum=5.0, parent_margin=5.0)
    arm_tl = float(_refill_plan['arm_timelimit'])
    _factory_feed_q = {seat: ctx.Queue() for seat in _factory_roles}
    _factory_handoff_meta = None
    _relay_feed_q = {s: ctx.Queue() for s in _relay_seats}
    _relay_feed_best = {s: 1e309 for s in _relay_seats}
    _relay_feed_count = 0
    for i, cfg in enumerate(arms):
        if i in _factory_roles:
            p = ctx.Process(target=_factory_consumer_worker, args=(q, _factory_feed_q[i], i, prob_info, arm_tl, cfg, t0), daemon=True)
        elif i in _relay_seats:
            _rcfg = dict(cfg)
            if _relay_seats[i] is not None:
                _rcfg['OGC_RELAY_SOURCE'] = str(_relay_seats[i])
            p = ctx.Process(target=_relay_tail_worker, args=(q, _relay_feed_q[i], i, prob_info, arm_tl, _rcfg, _mseed_starts[i]), daemon=True)
        elif i in _seed_seats:
            _seed_starts = [(rank, _factory_seed_picks[rank]) for rank in _seed_seats[i] if 0 <= rank < len(_factory_seed_picks)]
            p = ctx.Process(target=_seed_lns_worker, args=(q, i, prob_info, arm_tl, cfg, _seed_starts, t0), daemon=True)
        else:
            p = ctx.Process(target=_arm_worker, args=(q, i, prob_info, arm_tl, cfg, _mseed_starts[i], t0, list(_factory_feed_q.values()) if i == 0 else None), daemon=True)
        p.start()
        procs.append(p)
    deadline = t0 + timelimit - 5.0
    finals, ckpts, arm_meta, final_kinds = ({}, {}, {}, {})
    ckpt_audit = {}
    while len(finals) < len(arms) and time.time() < deadline:
        try:
            message = q.get(timeout=max(0.1, min(2.0, deadline - time.time())))
            kind, idx, sol, obj = message[:4]
            meta = message[4] if len(message) > 4 else None
            if kind == 'trace':
                _arm_curve_offer('arm_improvement', idx, sol, obj, meta.get('objective_kind') if isinstance(meta, dict) else 'assignment_recomputed_in_worker', 'assignments', meta)
            if kind == 'final':
                if isinstance(meta, dict):
                    meta = dict(meta)
                    meta['parent_completion_elapsed_s'] = float(time.time() - t0)
                _final_trace_meta = {'stage': meta.get('winning_stage') if isinstance(meta, dict) else None, 'worker_elapsed_s': meta.get('worker_wall_s') if isinstance(meta, dict) else None}
                _arm_curve_offer('arm_final', idx, sol, obj, 'checker_verified', 'solution', _final_trace_meta)
                finals[idx] = (sol, obj)
                final_kinds[idx] = 'final'
                if isinstance(meta, dict):
                    arm_meta[idx] = dict(meta)
                if sol is not None and isinstance(obj, float):
                    for _rs, _rsrc in _relay_seats.items():
                        if idx != _rs and (_rsrc is None or idx == _rsrc) and (obj < _relay_feed_best[_rs] - 1e-09):
                            _relay_feed_best[_rs] = float(obj)
                            _relay_feed_count += 1
                            try:
                                _relay_feed_q[_rs].put((float(obj), sol))
                            except Exception:
                                pass
                            finally:
                                pass
            elif kind == 'meta':
                if isinstance(meta, dict):
                    arm_meta[idx] = dict(meta)
            elif kind == 'factory_handoff':
                if isinstance(meta, dict):
                    _factory_handoff_meta = dict(meta)
            elif kind == 'ckpt':
                ckpts[idx] = (sol, obj)
                _ca = ckpt_audit.setdefault(idx, {'count': 0, 'best_streamed': None})
                _ca['count'] += 1
                if isinstance(obj, float) and (_ca['best_streamed'] is None or obj < _ca['best_streamed']):
                    _ca['best_streamed'] = float(obj)
                if sol is not None and isinstance(obj, float):
                    for _rs, _rsrc in _relay_seats.items():
                        if idx != _rs and (_rsrc is None or idx == _rsrc) and (obj < _relay_feed_best[_rs] - 1e-09):
                            _relay_feed_best[_rs] = float(obj)
                            _relay_feed_count += 1
                            try:
                                _relay_feed_q[_rs].put((float(obj), sol))
                            except Exception:
                                pass
                            finally:
                                pass
        except Exception:
            pass
        finally:
            pass
    for p in procs:
        if p.is_alive():
            p.terminate()
    from utils import check_feasibility
    for idx, (sol, obj) in sorted(ckpts.items()):
        if idx in finals and finals[idx][0] is not None:
            continue
        try:
            r = check_feasibility(prob_info, sol)
            if r['feasible']:
                finals[idx] = (sol, float(r['objective']))
                final_kinds[idx] = 'checkpoint'
        except Exception:
            pass
        finally:
            pass
    best, best_obj, best_idx = (None, 1e309, -1)
    for idx in sorted(finals):
        sol, obj = finals[idx]
        if sol is not None and isinstance(obj, float) and (obj < best_obj):
            best, best_obj, best_idx = (sol, obj, idx)
    portfolio_best_idx = int(best_idx)
    portfolio_best_obj = float(best_obj) if best is not None else None
    final_origin_arm = portfolio_best_idx
    parent_improvements = []
    _parent_lr_handoff = {'requested_candidates': 1, 'eligible_route': bool(_arm_dr >= 0.7), 'selection_stage': 'arm_final_before_parent_window_reswap', 'objective_gap': 0.08, 'minimum_behavior_distance': 0.0, 'selected': [] if best is None else [{'rank': 1, 'arm': int(best_idx), 'objective': float(best_obj), 'solution_hash': _solution_fingerprint(best), 'behavior_distance_from_primary': 0.0}], 'alternate_available': False, 'tail_attempted': False, 'tail_activated': False, 'tail_sources': [], 'tail_results': [], 'fallback_reason': None}
    _parent_lr_handoff['fallback_reason'] = 'one_candidate_requested'
    _parent_pup_gate, _parent_pup_promoted = _portfolio_pup_state(arm_meta)
    _factory_band = _parent_factory_band(_menu_resolved)
    _parent_winrep_enabled = _winrep_enabled(_parent_pup_gate) and (not _stripped_alap_menu) and (not _factory_band) and (not _heavy_production_qd_tail)
    _parent_winrep_disable_reason = None
    if _stripped_alap_menu:
        _parent_winrep_disable_reason = 'stripped_alap_menu'
    elif _heavy_production_qd_tail:
        _parent_winrep_disable_reason = 'heavy_production_qd_tail_policy'
    elif _factory_band:
        _parent_winrep_disable_reason = 'factory_band_policy'
    elif _parent_pup_gate and (not _parent_winrep_enabled):
        _parent_winrep_disable_reason = 'pup_policy'
    _winrep_meta = {'enabled': bool(_parent_winrep_enabled), 'global_enabled': bool(_winrep_enabled(False)), 'pup_gate': bool(_parent_pup_gate), 'pup_promoted': bool(_parent_pup_promoted), 'disable_reason': _parent_winrep_disable_reason, 'configured_reserve_s': float(_tail_res_configured), 'planned_reserve_s': float(_tail_res), 'reserve_returned_to_arms': bool(_tail_res_returned), 'reserve_policy': 'returned_to_arms' if _tail_res_returned else 'protected_parent', 'heavy_alap_parent_reserve_s': 0.0, 'admission_floor_s': None, 'left_s': None, 'budget_s': 0.0, 'attempted': False, 'wall_s': 0.0, 'returned_improvement': False, 'seed_objective': None, 'candidate_objective': None, 'delta': None, 'error': None}
    if best is not None and _parent_winrep_enabled:
        _wr_left = t0 + timelimit - 6.0 - time.time()
        _winrep_meta['admission_floor_s'] = 30.0
        _winrep_meta['left_s'] = float(_wr_left)
        _wr_cap = float(max(30.0, _tail_res + 15.0))
        _wr_budget = max(0.0, _wr_left - 6.0)
        if _wr_cap is not None:
            _wr_budget = min(_wr_budget, _wr_cap)
        _winrep_meta['budget_s'] = float(_wr_budget)
        if _wr_budget > 30.0 - 6.0:
            _wr_started = time.time()
            try:
                _winrep_meta['attempted'] = True
                _winrep_meta['seed_objective'] = float(best_obj)
                from winrep import window_stage, base_lateness
                from gpr_core import assignments_from_solution
                _wr_asgn = assignments_from_solution(best)
                _wr_deadline = time.time() + _wr_budget
                _wr = window_stage(prob_info, _wr_asgn, best_obj, base_lateness(prob_info, _wr_asgn), budget_fn=lambda: _wr_deadline - time.time(), chart=True, debug=False, max_windows=4)
                if _wr is not None:
                    _sol2, _r2 = _wr
                    _winrep_meta['returned_improvement'] = True
                    _winrep_meta['candidate_objective'] = float(_r2['objective'])
                    _winrep_meta['delta'] = float(_r2['objective']) - float(best_obj)
                    parent_improvements.append({'stage': 'window', 'from': float(best_obj), 'to': float(_r2['objective'])})
                    best = _sol2
                    best_obj = float(_r2['objective'])
            except Exception as _wr_exc:
                _winrep_meta['error'] = f'{type(_wr_exc).__name__}: {_wr_exc}'
            finally:
                _winrep_meta['wall_s'] = float(time.time() - _wr_started)
    _parent_fork_safe = True
    if best is not None and (not _stripped_alap_menu):
        _rs_left = t0 + timelimit - 6.0 - time.time()
        _rs_budget = max(0.0, _rs_left - 6.0)
        if _rs_left > 12.0 and _rs_budget > 3.0:
            try:
                from reswap import reswap
                from gpr_core import assignments_from_solution
                from gpr_core import retime as _rs_retime
                _rw, _rs = reswap(prob_info, assignments_from_solution(best), timelimit=_rs_budget, parallel=True)
                _parent_fork_safe = not _rs.get('serial_realized', False)
                if _rs['improved_rounds'] > 0:
                    _sol3, _ = _rs_retime(prob_info, _rw, verbose=False)
                    _r3 = check_feasibility(prob_info, _sol3)
                    if _r3['feasible'] and float(_r3['objective']) < best_obj - 1e-09:
                        parent_improvements.append({'stage': 'reswap', 'from': float(best_obj), 'to': float(_r3['objective'])})
                        best = _sol3
                        best_obj = float(_r3['objective'])
            except Exception:
                _parent_fork_safe = False
            finally:
                pass
    _sx_left = t0 + timelimit - 6.0 - time.time()
    _production_qd_tail_meta = {'enabled': bool(_heavy_production_qd_tail), 'activated': False, 'policy': 'ordinary LR producer rank-0; winrep disabled; normal strict parent reswap; short non-strict four-profile quality-diversity min; four-way deep-LNS min; no primal or spill gate' if _heavy_production_qd_tail else None, 'qd_fraction': 0.25 if _heavy_production_qd_tail else None, 'input_objective': float(best_obj) if _heavy_production_qd_tail and best is not None else None, 'output_objective': None, 'tail': None, 'error': None}
    _heavy_alap_reference = finals.get(best_idx, (None, None))[0] if best_idx >= 0 else None
    _heavy_alap_tail_meta = {'enabled': False, 'defaulted_from_factory_menu': False, 'activated': False, 'policy': None, 'quality_gap': 0.08, 'parent_reserve_s': 0.0, 'surplus_left_s': float(_sx_left), 'surplus_budget_s': float(max(0.0, _sx_left - 5.0)), 'distinct_gap': 0.08, 'distinct_min_distance': 0.05, 'distinct': None, 'distinct_fallback': None, 'sources': [], 'workers': {}}
    _factory_alap_tail_enabled = bool(_role_flow_enabled)
    _factory_alap_tail_meta = {'enabled': bool(_factory_alap_tail_enabled), 'activated': False, 'policy': '3x protected incumbent LNS; 1x near-zero-tardy ALAP path + seeded LNS, otherwise the original fourth seeded LNS', 'source': 'post-reswap incumbent', 'plan_anchor_arm': 0, 'quality_gap': 0.08, 'max_obj1': 2.0, 'surplus_left_s': float(_sx_left), 'surplus_budget_s': float(max(0.0, _sx_left - 5.0)), 'plan_targets': 0, 'worker': None}
    if best is not None and _heavy_production_qd_tail and _parent_fork_safe:
        try:
            _prod_qd_before = float(best_obj)
            best, best_obj, _prod_qd_tail = _production_qd_lns_tail(prob_info, best, best_obj, deadline=t0 + timelimit - 6.0, qd_fraction=_production_qd_tail_meta['qd_fraction'], qd_seed_base=1701, lns_seed_base=701)
            _production_qd_tail_meta.update({'activated': bool(_prod_qd_tail.get('activated')), 'input_objective': float(_prod_qd_before), 'output_objective': float(best_obj), 'tail': _prod_qd_tail})
            _parent_lr_handoff['production_qd_tail_attempted'] = True
            _parent_lr_handoff['production_qd_tail_stages'] = [{'stage': int(row['stage']), 'operator': str(row['operator']), 'activated': bool(row['activated']), 'input_objective': float(row['input_objective']), 'output_objective': float(row['output_objective']), 'improved': bool(row.get('improved', False))} for row in _prod_qd_tail.get('stages', [])]
            for row in _prod_qd_tail.get('stages', []):
                if row.get('improved'):
                    parent_improvements.append({'stage': 'production_qd_tail', 'operator': str(row['operator']), 'from': float(row['input_objective']), 'to': float(row['output_objective'])})
        except Exception as _prod_qd_exc:
            _production_qd_tail_meta['error'] = f'{type(_prod_qd_exc).__name__}: {_prod_qd_exc}'
        finally:
            pass
    if best is not None and (not _heavy_production_qd_tail) and (not _stripped_alap_menu) and _parent_fork_safe and (_sx_left > 18.0):
        try:
            _sctx = mp.get_context('fork')
            _sq = _sctx.Queue()
            _sprocs = []
            _sx_tl = _sx_left - 5.0
            _sx_sources = [('tail', best, best_obj)] * 4
            _sx_plan_targets = [None] * 4
            _sx_factory_alap = bool(_factory_alap_tail_enabled and len(_sx_sources) == 4)
            if _sx_factory_alap:
                _sx_sources, _sx_plan_targets = _factory_alap_tail_sources(best, best_obj, arm_meta, slots=len(_sx_sources))
                _factory_alap_tail_meta['activated'] = True
                _factory_alap_tail_meta['plan_targets'] = int(len(_sx_plan_targets[-1] or {}))
            _sx_seeds = (701, 702, 703, 704)
            _parent_lr_handoff['tail_attempted'] = True
            _parent_lr_handoff['tail_sources'] = [{'worker': int(_si), 'source': str(_src), 'start_objective': float(_src_obj), 'start_solution_hash': _solution_fingerprint(_start), 'seed': int(_sx_seeds[_si])} for _si, (_src, _start, _src_obj) in enumerate(_sx_sources)]
            for _si, (_src, _start, _src_obj) in enumerate(_sx_sources):
                if _sx_factory_alap and _si == len(_sx_sources) - 1:
                    _sp = _sctx.Process(target=_surplus_heavy_alap_worker, args=(_sq, _si, prob_info, _start, _sx_tl, _sx_seeds[_si], _sx_plan_targets[_si], 'factory'), daemon=True)
                else:
                    _sp = _sctx.Process(target=_surplus_worker, args=(_sq, _si, prob_info, _start, _sx_tl, _sx_seeds[_si]), daemon=True)
                _sp.start()
                _sprocs.append(_sp)
            _sx_deadline = t0 + timelimit - 6.0
            _sx_out = {}
            while len(_sx_out) < len(_sprocs) and time.time() < _sx_deadline:
                try:
                    _sx_item = _sq.get(timeout=max(0.1, min(1.0, _sx_deadline - time.time())))
                    _i, _sol, _obj = _sx_item[:3]
                    _worker_meta = _sx_item[3] if len(_sx_item) > 3 and isinstance(_sx_item[3], dict) else None
                    _sx_out[_i] = (_sol, _obj, _worker_meta)
                    if _sx_factory_alap and _worker_meta is not None and (int(_i) == len(_sx_sources) - 1):
                        _factory_alap_tail_meta['worker'] = _worker_meta
                except Exception:
                    pass
                finally:
                    pass
            for _sp in _sprocs:
                if _sp.is_alive():
                    _sp.terminate()
                _sp.join(timeout=0.5)
            _parent_lr_handoff['tail_results'] = [{'worker': int(_si), 'source': str(_sx_sources[_si][0]), 'completed': bool(_si in _sx_out), 'objective': float(_sx_out[_si][1]) if _si in _sx_out and isinstance(_sx_out[_si][1], float) else None, 'error': str(_sx_out[_si][1]) if _si in _sx_out and isinstance(_sx_out[_si][1], str) else None} for _si in range(len(_sx_sources))]
            for _i in sorted(_sx_out):
                _sol, _obj, _worker_meta = _sx_out[_i]
                if _sol is not None and isinstance(_obj, float) and (_obj < best_obj - 1e-09):
                    _src = _sx_sources[_i][0]
                    parent_improvements.append({'stage': 'heavy_alap' if _src.startswith('heavy-alap') else 'factory_alap' if _src.startswith('factory-alap') else 'surplus', 'worker': int(_i), 'start_source': _src, 'from': float(best_obj), 'to': float(_obj)})
                    if _src.startswith('arm'):
                        try:
                            final_origin_arm = int(_src[3:].split('@', 1)[0])
                        except Exception:
                            pass
                        finally:
                            pass
                    elif _src.startswith('bc-arm'):
                        try:
                            final_origin_arm = int(_src[6:].split('@', 1)[0])
                        except Exception:
                            pass
                        finally:
                            pass
                    elif _src.startswith('heavy-alap-D-arm'):
                        try:
                            final_origin_arm = int(_src[16:].split('@', 1)[0])
                        except Exception:
                            pass
                        finally:
                            pass
                    best, best_obj = (_sol, _obj)
        except Exception as _se:
            pass
        finally:
            pass
    arm_records = []
    for _idx, _cfg in enumerate(arms):
        _meta = dict(arm_meta.get(_idx, {}))
        _parent_plan_count = len(_meta.pop('_parent_plan_targets', {}) or {})
        _forced = _cfg.get('OGC_FORCE_ROUTE', os.environ.get('OGC_FORCE_ROUTE', '')) or None
        _split = _cfg.get('OGC_BOTH_SPLIT', os.environ.get('OGC_BOTH_SPLIT', ''))
        _fixed_both_rounds_cfg = _cfg.get('OGC_BOTH_ROUNDS', os.environ.get('OGC_BOTH_ROUNDS', ''))
        _multi_share_cfg = _cfg.get('OGC_MULTI_SHARE', os.environ.get('OGC_MULTI_SHARE', ''))
        _sampler_cfg = _cfg.get('OGC_ARM_SAMPLER', os.environ.get('OGC_ARM_SAMPLER', '')) == '1'
        _heavy_factory_cfg = _cfg.get('OGC_HEAVY_FACTORY_SAMPLER', os.environ.get('OGC_HEAVY_FACTORY_SAMPLER', '')) == '1'
        _aeff_scale_cfg = _cfg.get('OGC_AEFF_SCALE', os.environ.get('OGC_AEFF_SCALE', ''))
        _heavy_factory_profile_cfg = _cfg.get('OGC_HEAVY_FACTORY_PROFILE', os.environ.get('OGC_HEAVY_FACTORY_PROFILE', '')) or None
        _no_karm_cfg = _cfg.get('OGC_NO_KARM', os.environ.get('OGC_NO_KARM', '')) == '1'
        _primal_realization_cfg = _cfg.get('OGC_PRIMAL_REALIZATION', os.environ.get('OGC_PRIMAL_REALIZATION', '')) == '1'
        _primal_direct_both_cfg = _cfg.get('OGC_PRIMAL_DIRECT_BOTH', os.environ.get('OGC_PRIMAL_DIRECT_BOTH', '')) == '1'
        _primal_jitter_cfg = _cfg.get('OGC_PRIMAL_JITTER_DAYS', os.environ.get('OGC_PRIMAL_JITTER_DAYS', ''))
        _primal_seed_cfg = _cfg.get('OGC_PRIMAL_SEED', os.environ.get('OGC_PRIMAL_SEED', ''))
        _pup_light_cfg = _cfg.get('OGC_PUP_FORCE_LIGHT', os.environ.get('OGC_PUP_FORCE_LIGHT', '')) == '1'
        _light_profile_cfg = _cfg.get('OGC_LIGHT_PROFILE', os.environ.get('OGC_LIGHT_PROFILE', '')) or None
        _factory_producer_cfg = _cfg.get('OGC_FACTORY_PRODUCER', os.environ.get('OGC_FACTORY_PRODUCER', '')) == '1'
        _factory_role_cfg = _cfg.get('OGC_FACTORY_ROLE', os.environ.get('OGC_FACTORY_ROLE', '')) or None
        _portfolio_role_cfg = _cfg.get('OGC_PORTFOLIO_ROLE', os.environ.get('OGC_PORTFOLIO_ROLE', '')) or None
        _factory_bucket_cfg = _cfg.get('OGC_FACTORY_PLAN_BUCKET', os.environ.get('OGC_FACTORY_PLAN_BUCKET', ''))
        _lr_relative_dispatch_cfg = _cfg.get('OGC_LR_RELATIVE_DISPATCH', os.environ.get('OGC_LR_RELATIVE_DISPATCH', '')) == '1'
        _lr_blocked_weight_cfg = _cfg.get('OGC_LR_BLOCKED_WEIGHT', os.environ.get('OGC_LR_BLOCKED_WEIGHT', ''))
        _lr_zobrist_dedup_cfg = _cfg.get('OGC_LR_ZOBRIST_DEDUP', os.environ.get('OGC_LR_ZOBRIST_DEDUP', '')) == '1'
        _alap_ns_qgap = _cfg.get('OGC_ALAP_NS_QGAP', os.environ.get('OGC_ALAP_NS_QGAP', ''))
        _alap_ns_batch = _cfg.get('OGC_ALAP_NS_BATCH', os.environ.get('OGC_ALAP_NS_BATCH', ''))
        _alap_ns_latest = _cfg.get('OGC_ALAP_NS_LATEST', os.environ.get('OGC_ALAP_NS_LATEST', ''))
        _nest_lane_cfg = _cfg.get('OGC_NEST_LANE', os.environ.get('OGC_NEST_LANE', ''))
        _aeff_scale_cfg = float(_cfg.get('OGC_AEFF_SCALE', os.environ.get('OGC_AEFF_SCALE', '1.0')))
        _no_karm_cfg = _cfg.get('OGC_NO_KARM', os.environ.get('OGC_NO_KARM', '')) == '1'
        _no_heavy_ms_cfg = _cfg.get('OGC_NO_HEAVY_MS', os.environ.get('OGC_NO_HEAVY_MS', '')) == '1'
        _no_ext_cfg = _cfg.get('OGC_NO_EXT', os.environ.get('OGC_NO_EXT', '')) == '1'
        _sol_obj = finals.get(_idx, (None, None))[1]
        _meta.update({'arm': int(_idx), 'mseed_rank': int(_cfg.get('OGC_MSEED_RANK', -1 if _menu_resolved == 'factory4' else _idx)) if _mseed_on else None, 'configured_route': _forced or 'auto', 'configured_both_split': None if _fixed_both_rounds_cfg not in ('', None) else float(_split) if _split not in ('', None) else None, 'configured_both_rounds': int(_fixed_both_rounds_cfg) if _fixed_both_rounds_cfg not in ('', None) else None, 'configured_multi_share': float(_multi_share_cfg) if _multi_share_cfg not in ('', None) else None, 'configured_sampler': bool(_sampler_cfg), 'configured_heavy_factory': bool(_heavy_factory_cfg), 'configured_aeff_scale': float(_aeff_scale_cfg) if _aeff_scale_cfg not in ('', None) else None, 'configured_heavy_factory_profile': _heavy_factory_profile_cfg, 'configured_no_karm': bool(_no_karm_cfg), 'configured_primal_realization': bool(_primal_realization_cfg), 'configured_primal_direct_both': bool(_primal_direct_both_cfg), 'configured_primal_jitter_days': float(_primal_jitter_cfg) if _primal_jitter_cfg not in ('', None) else None, 'configured_primal_seed': int(_primal_seed_cfg) if _primal_seed_cfg not in ('', None) else None, 'configured_pup_force_light': bool(_pup_light_cfg), 'configured_light_profile': _light_profile_cfg, 'configured_factory_producer': bool(_factory_producer_cfg), 'configured_factory_role': _factory_role_cfg, 'configured_portfolio_role': _portfolio_role_cfg, 'configured_factory_plan_bucket': int(_factory_bucket_cfg) if _factory_bucket_cfg not in ('', None) else None, 'configured_lr_relative_dispatch': bool(_lr_relative_dispatch_cfg), 'configured_lr_blocked_weight': float(_lr_blocked_weight_cfg) if _lr_blocked_weight_cfg not in ('', None) else None, 'configured_lr_zobrist_dedup': bool(_lr_zobrist_dedup_cfg), 'configured_alap_nonstrict_quality_gap': float(_alap_ns_qgap) if _alap_ns_qgap not in ('', None) else None, 'configured_alap_nonstrict_batch_size': int(_alap_ns_batch) if _alap_ns_batch not in ('', None) else None, 'configured_alap_latest_scaffold': _alap_ns_latest != '0' if _alap_ns_latest not in ('', None) else None, 'configured_nest_lane': int(_nest_lane_cfg) if str(_nest_lane_cfg).strip() not in ('', 'None') else None, 'parent_plan_targets_available': int(_parent_plan_count), 'status': final_kinds.get(_idx, 'missing'), 'objective': float(_sol_obj) if isinstance(_sol_obj, float) else None})
        if isinstance(_sol_obj, str):
            _meta['error'] = _sol_obj
        if 'route' not in _meta and _forced in ('light', 'both', 'heavy'):
            _meta['route'] = _forced
        arm_records.append(_meta)
    _heavy_producer_menu_meta = {'enabled': bool(_heavy_menu_resolved == 'lr_qd_lns4'), 'policy': 'four ordinary LR producer axes; checker rank-0 handoff; primal realization and spill classification disabled' if _heavy_menu_resolved == 'lr_qd_lns4' else None, 'all_primal_disabled': bool(_heavy_menu_resolved == 'lr_qd_lns4' and all((not row.get('configured_primal_realization', False) for row in arm_records))), 'seats': [{'arm': row.get('arm'), 'plan_bucket': int(arms[index]['OGC_PLAN_BUCKET']) if index < len(arms) and arms[index].get('OGC_PLAN_BUCKET') not in (None, '') else None, 'aeff_scale': row.get('configured_aeff_scale'), 'no_karm': row.get('configured_no_karm'), 'primal_realization': row.get('configured_primal_realization')} for index, row in enumerate(arm_records)]}
    _origin_meta = arm_records[final_origin_arm] if 0 <= final_origin_arm < len(arm_records) else {}
    _ckpt_audit_rows = []
    for _ai in range(len(arm_records)):
        _ca = ckpt_audit.get(_ai)
        _fin = arm_records[_ai].get('objective')
        _row = {'arm': int(_ai), 'streamed': int(_ca['count']) if _ca else 0, 'best_streamed_internal': _ca['best_streamed'] if _ca else None, 'arm_final_objective': _fin, 'arm_result_kind': final_kinds.get(_ai), 'regressed_from_stream': False, 'verified_checkpoint_objective': None, 'recoverable_gain_percent': None}
        if _ca is not None and _ca['best_streamed'] is not None and isinstance(_fin, float) and (_ca['best_streamed'] < _fin - 1e-09) and (final_kinds.get(_ai) == 'final'):
            _row['regressed_from_stream'] = True
            try:
                _cr = check_feasibility(prob_info, ckpts[_ai][0])
                if _cr.get('feasible'):
                    _cv = float(_cr['objective'])
                    _row['verified_checkpoint_objective'] = _cv
                    if _cv < _fin - 1e-09:
                        _row['recoverable_gain_percent'] = 100.0 * (_cv / _fin - 1.0)
            except Exception:
                pass
            finally:
                pass
        _ckpt_audit_rows.append(_row)
    _forecast_rows = []
    _incumbent_rows = []
    _arm_curve_rows = []
    RUN_PROVENANCE.clear()
    RUN_PROVENANCE.update({'schema_version': 1, 'mode': 'portfolio', 'forecast_trace': {'enabled': False, 'schema_version': 1, 'candidate_events': int(len(_forecast_raw_events)), 'incumbent_events': int(len(_forecast_rows)), 'policy': 'strict portfolio incumbent after recomputing assignment objective; physical features extracted after search', 'events': _forecast_rows}, 'incumbent_trace': {'enabled': False, 'schema_version': 1, 'clock': 'portfolio_start', 'candidate_events': int(len(_forecast_raw_events)), 'incumbent_events': int(len(_incumbent_rows)), 'policy': 'strict portfolio incumbent after recomputing assignment objective; no search decision reads this trace', 'events': _incumbent_rows}, 'arm_curve_trace': {'enabled': False, 'schema_version': 1, 'clock': 'portfolio_start', 'candidate_events': int(len(_arm_curve_raw_events)), 'incumbent_events': int(len(_arm_curve_rows)), 'policy': 'strict arm-local incumbent after recomputing assignment objective; no search decision reads this trace', 'events': _arm_curve_rows}, 'checkpoint_audit': _ckpt_audit_rows, 'relay_arm': {'enabled': bool(_relay_seats), 'seats': [{'arm': int(_rs), 'source': int(_rv) if _rv is not None else 'global', 'best_fed_objective': float(_relay_feed_best[_rs]) if _relay_feed_best[_rs] != 1e309 else None} for _rs, _rv in sorted(_relay_seats.items())], 'window_s': float(os.environ.get('OGC_RELAY_WINDOW', '15')) if _relay_seats else None, 'feeds_sent': int(_relay_feed_count)}, 'factory_flow': {'enabled': bool(_role_flow_enabled), 'band': 'factory' if _menu_resolved == 'factory4' else 'direct_both' if _role_flow_enabled else None, 'producer_arm': 0 if _role_flow_enabled else None, 'handoff_round_target': 2 if _role_flow_enabled else None, 'consumer_roles': [{'arm': int(seat), 'role': str(role)} for seat, role in sorted(_factory_roles.items())], 'cross_seats': [{'arm': int(seat), 'source': 'global' if source is None else int(source), 'start_frac': float(arms[seat].get('OGC_RELAY_START_FRAC', os.environ.get('OGC_RELAY_START_FRAC', '0.15')))} for seat, source in sorted(_relay_seats.items()) if _role_flow_enabled], 'seed_seats': [{'arm': int(seat), 'ranks': [int(rank) for rank in ranks], 'picks_delivered': sum((1 for rank in ranks if 0 <= rank < len(_factory_seed_picks)))} for seat, ranks in sorted(_seed_seats.items())], 'independent_lr_arms': [{'arm': int(seat), 'dispatch': 'wait_from_release', 'blocked_weight': float(cfg.get('OGC_LR_BLOCKED_WEIGHT', '0')), 'zobrist_dedup': cfg.get('OGC_LR_ZOBRIST_DEDUP', '') == '1'} for seat, cfg in enumerate(arms) if cfg.get('OGC_LR_RELATIVE_DISPATCH', '') == '1'], 'handoff': _factory_handoff_meta, 'mseed_disabled': bool(_role_flow_enabled and (not _mseed_on)), 'who_waits_locked': bool(_role_flow_enabled), 'a2_multi_share': float(arms[2]['OGC_MULTI_SHARE']) if _role_flow_enabled and len(arms) > 2 and (arms[2].get('OGC_MULTI_SHARE') not in (None, '')) else None, 'a2_multi_share_policy': arms[2].get('OGC_FACTORY_A2_SHARE_POLICY') if _role_flow_enabled and len(arms) > 2 else None, 'a2_multi_share_gate_tl': float(arms[2]['OGC_FACTORY_A2_SHARE_GATE_TL']) if _role_flow_enabled and len(arms) > 2 and (arms[2].get('OGC_FACTORY_A2_SHARE_GATE_TL') not in (None, '')) else None}, 'timelimit': float(timelimit), 'arm_timelimit': float(arm_tl), 'arm_allocation': dict(_refill_plan), 'nonheavy_arm_menu': _menu_resolved, 'stripped_alap_menu': bool(_stripped_alap_menu), 'heavy_arm_menu': _heavy_menu_resolved, 'heavy_production_qd_tail': _production_qd_tail_meta, 'heavy_producer_menu': _heavy_producer_menu_meta, 'mseed_enabled': bool(_mseed_on), 'mseed_planned_s': float(_mseed_splan), 'mseed_wall_s': float(_mseed_wall_s), 'mseed_work': _mseed_work, 'mseed_selection': _mseed_pick_diag, 'parent_pup_gate': bool(_parent_pup_gate), 'parent_pup_promoted': bool(_parent_pup_promoted), 'parent_lr_handoff': _parent_lr_handoff, 'winrep': _winrep_meta, 'factory_alap_tail': _factory_alap_tail_meta, 'heavy_alap_tail': _heavy_alap_tail_meta, 'arms': arm_records, 'portfolio_winner_arm': portfolio_best_idx if portfolio_best_idx >= 0 else None, 'portfolio_winner_route': arm_records[portfolio_best_idx].get('route') if 0 <= portfolio_best_idx < len(arm_records) else None, 'portfolio_winner_stage': arm_records[portfolio_best_idx].get('winning_stage') if 0 <= portfolio_best_idx < len(arm_records) else None, 'portfolio_winner_objective': portfolio_best_obj, 'final_origin_arm': final_origin_arm if final_origin_arm >= 0 else None, 'final_origin_route': _origin_meta.get('route'), 'parent_improvements': parent_improvements, 'final_stage': parent_improvements[-1]['stage'] if parent_improvements else 'arm', 'final_objective': float(best_obj) if best is not None else None})
    return best

def algorithm(prob_info, timelimit=60):
    RUN_PROVENANCE.clear()
    if os.environ.get('OGC_ARM', '') == '' and sys.platform.startswith('linux'):
        from plan_core import demand_ratio
        _dr_gate = demand_ratio(prob_info)
        _rust_arms = _rust_enabled()
        if _dr_gate < 0.6 or _rust_arms or False:
            _t0 = time.time()
            sol = _parallel_portfolio(prob_info, timelimit, plan_region=_dr_gate >= 0.6)
            if sol is not None:
                return sol
            timelimit = max(10.0, timelimit - (time.time() - _t0) - 2.0)
    _single_meta = {}
    _single = _algorithm_single(prob_info, timelimit, run_meta=_single_meta)
    RUN_PROVENANCE.clear()
    RUN_PROVENANCE.update({'schema_version': 1, 'mode': 'single', 'timelimit': float(timelimit), 'arms': [_single_meta], 'portfolio_winner_arm': None, 'portfolio_winner_route': _single_meta.get('route'), 'portfolio_winner_stage': _single_meta.get('winning_stage'), 'portfolio_winner_objective': _single_meta.get('winning_internal_objective'), 'final_origin_arm': None, 'final_origin_route': _single_meta.get('route'), 'parent_improvements': [], 'final_stage': 'single', 'final_objective': _single_meta.get('winning_internal_objective')})
    return _single

def _algorithm_single(prob_info, timelimit=60, ckpt_cb=None, start_assign=None, run_meta=None, route_meta_cb=None, trace_cb=None, trace_origin=None, factory_handoff=None, factory_handoff_cb=None):
    t_start = time.time()
    trace_origin = t_start if trace_origin is None else float(trace_origin)
    baseline_start_assign = start_assign
    _factory_role = os.environ.get('OGC_FACTORY_ROLE', '').strip()
    _factory_producer = os.environ.get('OGC_FACTORY_PRODUCER', '') == '1'
    _factory_shared_start = bool(isinstance(factory_handoff, dict) and isinstance(factory_handoff.get('assignment'), dict))
    _factory_locked_prob = _who_waits_locked_problem(prob_info, factory_handoff['plan']) if _factory_shared_start and isinstance(factory_handoff.get('plan'), dict) else None
    _factory_lns_start = None
    _lr_relative_dispatch = os.environ.get('OGC_LR_RELATIVE_DISPATCH', '') == '1'
    _lr_blocked_weight = max(0.0, float(os.environ.get('OGC_LR_BLOCKED_WEIGHT', '0')))
    _lr_zobrist_dedup = os.environ.get('OGC_LR_ZOBRIST_DEDUP', '') == '1'
    _lr_feedback_meta = {'enabled': bool(_lr_relative_dispatch or _lr_blocked_weight > 0.0), 'dispatch': 'wait_from_release' if _lr_relative_dispatch else 'absolute_start', 'blocked_weight': float(_lr_blocked_weight), 'zobrist_dedup': bool(_lr_zobrist_dedup), 'rounds': []}
    from constructor import construct, construct_multi, _objective_of, _RUST_LIB
    from gpr_core import retime
    from lns import lns_sa
    from utils import check_feasibility
    n_blocks = len(prob_info.get('blocks', []))
    _margin_override = os.environ.get('OGC_ARM_FINAL_MARGIN')
    margin = max(5.0, float(_margin_override)) if _margin_override not in (None, '') else min(45.0, max(8.0, 0.1 * timelimit, n_blocks * 0.04))
    deadline = t_start + timelimit - margin

    def remaining():
        return deadline - time.time()
    geo_cache = {}
    pair_memo = {}
    rel_memo = {}
    candidates = []
    candidate_sources = {}
    if run_meta is not None:
        run_meta.update({'rust_enabled': bool(_rust_enabled()), 'rust_native_loaded': bool(_RUST_LIB is not None), 'configured_timelimit_s': float(timelimit), 'final_margin_s': float(margin), 'search_deadline_s': float(max(0.0, timelimit - margin)), 'factory_producer': bool(_factory_producer), 'factory_role': _factory_role or None, 'factory_handoff_start': bool(_factory_shared_start), 'lr_wait_pressure': _lr_feedback_meta})

    def obj_of(assign):
        return _objective_of(prob_info, assign)[0]

    def _trajectory_best():
        return min(candidates, key=obj_of)
    _incumbent_trace = {'enabled': False, 'schema_version': 1, 'clock': 'arm_start' if trace_origin == t_start else 'portfolio_start', 'events': [], 'best_objective': None}
    if run_meta is not None:
        run_meta['incumbent_trace'] = _incumbent_trace

    def _trace_objective(objective, event, elapsed_s=None, **fields):
        return _append_strict_incumbent_event(_incumbent_trace, objective, time.time() - trace_origin if elapsed_s is None else elapsed_s, event, **fields)
    PROGRESS.clear()
    _sg_seen = [0]
    _sg_best = [None]

    def _stage(tag):
        for c in candidates[_sg_seen[0]:]:
            candidate_sources.setdefault(id(c), tag)
            o = obj_of(c)
            if _sg_best[0] is None or o < _sg_best[0]:
                _sg_best[0] = o
            _trace_objective(o, 'stage_candidate', stage=tag)
        _sg_seen[0] = len(candidates)
        PROGRESS.update(stage=tag, obj=_sg_best[0], t=round(time.time() - t_start, 1))
        if candidates and (ckpt_cb is not None or trace_cb is not None):
            _stage_incumbent = min(candidates, key=obj_of)
            if ckpt_cb is not None:
                ckpt_cb(_stage_incumbent)
            if trace_cb is not None:
                trace_cb(_stage_incumbent, tag)
    total_budget = deadline - t_start
    _arm_seed = int(os.environ.get('OGC_ARM_SEED', '0'))
    _est_pass = 0.5 + n_blocks / 40.0
    from plan_core import demand_ratio as _dratio
    _m_sched = len(prob_info['bays'])
    _dr_sched = _dratio(prob_info)
    if _dr_sched < 0.45:
        if n_blocks <= 120:
            _ps = 0.21
        elif n_blocks <= 170:
            _ps = 0.28
        elif n_blocks <= 240:
            _ps = 0.55
        elif n_blocks <= 350:
            _ps = 1.25 if _m_sched >= 5 else 0.7
        else:
            _ps = 1.9 if _m_sched >= 5 else 1.05
    else:
        _ps = 0.45 if n_blocks <= 120 else 1.5 if n_blocks <= 240 else 2.6 if n_blocks <= 350 else 3.9
    _it = 0.055 if n_blocks <= 120 else 0.075 if n_blocks <= 170 else 0.11 if n_blocks <= 240 else 0.14 if n_blocks <= 350 else 0.21
    _rd = 4.5 if n_blocks <= 120 else 7.0 if n_blocks <= 170 else 13.0 if n_blocks <= 220 else 20.0 if n_blocks <= 350 else 30.0
    _pt = 0.012 if n_blocks <= 120 else 0.013 if n_blocks <= 170 else 0.017 if n_blocks <= 240 else 0.024 if n_blocks <= 350 else 0.036
    _st = 0.12 if n_blocks <= 120 else 0.15 if n_blocks <= 170 else 0.19 if n_blocks <= 240 else 0.22 if n_blocks <= 350 else 0.33
    _wscale = float(os.environ.get('OGC_WORK_SCALE', '1.0'))
    _ps *= _wscale
    _it *= _wscale
    _rd *= _wscale
    _pt *= _wscale
    _st *= _wscale
    probe_degraded = True
    if _factory_shared_start:
        probe = {int(block_id): dict(row) for block_id, row in factory_handoff['assignment'].items()}
        candidates.append(probe)
        probe_wall = 0.0
        probe_degraded = False
    else:
        try:
            probe, pstats = construct(prob_info, timelimit=max(5.0, total_budget), geo_cache=geo_cache, pair_memo=pair_memo, rel_memo=rel_memo, verbose=False)
            candidates.append(probe)
            probe_wall = pstats['wall_s']
            probe_degraded = bool(pstats.get('degraded', True))
        except Exception:
            probe_wall = total_budget
        finally:
            pass
    if not candidates:
        probe, _ = construct(prob_info, timelimit=max(5.0, remaining()), max_xs=4, max_ys=3, top_b=1, nest_lane=0, slide=False, geo_cache=geo_cache, verbose=False)
        candidates.append(probe)
    _stage('probe')
    costly_pass = _est_pass > 0.15 * total_budget
    tail_reserve = min(90.0, max(20.0, 0.12 * total_budget)) if total_budget > 150.0 else 0.0
    workable = max(0.0, remaining() - tail_reserve)
    _multi_share = min(0.95, max(0.05, float(os.environ.get('OGC_MULTI_SHARE', '0.45'))))
    _lns_share = 1.0 - _multi_share
    lns_budget = 0.0 if costly_pass else workable * _lns_share
    multi_budget = max(0.0, workable - lns_budget)
    from plan_core import demand_ratio
    _aeff = 0.72
    _abar = 0.72
    import numpy as _np
    from plan_core import _layer0_area as _l0a, _fits_bay as _fitb
    _bl, _by = (prob_info['blocks'], prob_info['bays'])
    _nbay = len(_by)
    _bA = _np.array([b['width'] * b['height'] for b in _by], dtype=float)

    def _sense_av(_asgn):
        _T = max((a['exit_time'] for a in _asgn.values())) + 1
        _occ = _np.zeros((_nbay, _T))
        _con = _np.zeros((_nbay, _T), dtype=bool)
        for _i, _a in _asgn.items():
            _blk = _bl[_i]
            _occ[_a['bay_id'], _a['entry_time']:_a['exit_time']] += _l0a(_blk)
            _gate = max(_blk['release_time'], _blk['due_date'] - _blk['processing_time'])
            if _a['entry_time'] > _gate:
                for _j in range(_nbay):
                    if _fitb(_blk, _by[_j]):
                        _con[_j, _gate:_a['entry_time']] = True
        _av = _np.full(_nbay, 0.72)
        _seen = _np.zeros(_nbay, dtype=bool)
        for _j in range(_nbay):
            if _con[_j].any():
                _seen[_j] = True
                _cp = float(_occ[_j][_con[_j]].max()) / _bA[_j]
                _mf = max((_l0a(b) for b in _bl if _fitb(b, _by[_j])), default=0.0) / _bA[_j]
                _av[_j] = min(1.0, max(0.25, _cp + 0.02, _mf + 0.02))
        return (_av, _seen)
    _aeff, _ = _sense_av(probe)
    _ascale = float(os.environ.get('OGC_AEFF_SCALE', '1.0'))
    if _ascale != 1.0:
        _aeff = _np.clip(_aeff * _ascale, 0.25, 1.0)
    _abar = float((_aeff * _bA).sum() / _bA.sum())
    _route = os.environ.get('OGC_FORCE_ROUTE', '')
    _dr = demand_ratio(prob_info)
    _P = _dr / _abar
    _oh_full = (40.0 if _dr_sched >= 0.45 else 0.0) + (20.0 if n_blocks > 350 else 0.0)
    _sched_oh = _oh_full
    _wk = max(0.0, total_budget - tail_reserve - _est_pass - _sched_oh)
    _pup = _pup_gate_active(True, _dr, _P)
    _pup_forced = _pup and False
    _pup_force_light = os.environ.get('OGC_PUP_FORCE_LIGHT', '') == '1'
    _slr, _both = _resolve_arm_route(_dr, _pup, False, _route, _pup_force_light)
    _factory_deep = _factory_role in ('deep_lns', 'bucket_lns')
    if _factory_role:
        multi_budget = 0.0
        lns_budget = max(0.0, workable) if _factory_deep else 0.0
    _light_profile_requested = os.environ.get('OGC_LIGHT_PROFILE', '')
    _light_alap_nonstrict = bool(_light_profile_requested in ('alap_nonstrict', 'alap_batch') and (not _slr) and (not _both))
    _alap_ns_meta = {'requested': _light_profile_requested in ('alap_nonstrict', 'alap_batch'), 'enabled': bool(_light_alap_nonstrict), 'quality_gap': max(0.0, float(os.environ.get('OGC_ALAP_NS_QGAP', '0.10'))), 'batch_size': max(1, int(os.environ.get('OGC_ALAP_NS_BATCH', '6'))), 'anchors': 16, 'max_rounds': max(1, int(os.environ.get('OGC_ALAP_NS_ROUNDS', '48'))), 'max_proposals_per_round': max(1, int(os.environ.get('OGC_ALAP_NS_PROPOSALS', '96'))), 'candidate_timelimit_s': max(0.2, float(os.environ.get('OGC_ALAP_NS_CAND_TL', '3'))), 'latest_scaffold': os.environ.get('OGC_ALAP_NS_LATEST', '1') != '0', 'repair_mode': os.environ.get('OGC_ALAP_NS_REPAIR', 'construct').strip().lower(), 'cpsat_det_units': max(0.05, float(os.environ.get('OGC_ALAP_NS_CPSAT_DET', '1'))), 'cpsat_workers': max(1, int(os.environ.get('OGC_ALAP_NS_CPSAT_WORKERS', '1'))), 'acceptance_mode': 'nonstrict' if float(os.environ.get('OGC_ALAP_NS_QGAP', '0.10')) > 0.0 else 'strict', 'excluded_arm_stages': ['seat', 'lns', 'pull', 'extension', 'window_repair', 'reswap'] if _light_alap_nonstrict else [], 'attempted': False, 'budget_s': 0.0, 'result': None}
    _primal_direct_both = bool(_both and os.environ.get('OGC_PRIMAL_DIRECT_BOTH', '') == '1')
    _primal_realization_enabled = bool(_slr and (not _both or _primal_direct_both) and (os.environ.get('OGC_PRIMAL_REALIZATION', '') == '1'))
    _primal_bias = None
    _primal_meta = {'enabled': bool(_primal_realization_enabled), 'direct_both': bool(_primal_direct_both), 'policy': 'fixed-seed correlated insertion-priority bias; LR plan, lambda, bay scoring, and target timing remain unchanged', 'jitter_days': float(os.environ.get('OGC_PRIMAL_JITTER_DAYS', '4.0')), 'seed': int(os.environ.get('OGC_PRIMAL_SEED', '109')), 'window': max(1, int(os.environ.get('OGC_PRIMAL_WINDOW', '9'))), 'bay_weight': min(1.0, max(0.0, float(os.environ.get('OGC_PRIMAL_BAY_WEIGHT', '0.30')))), 'calls': 0, 'core_calls': 0, 'extension_calls': 0, 'mean_moved_fraction': 0.0, 'mean_abs_rank_shift': 0.0, 'max_abs_rank_shift': 0.0, 'first_order': None, 'last_order': None}
    if _primal_realization_enabled:
        _primal_bias, _primal_bias_diag = _structured_primal_priority_bias(prob_info, jitter_days=_primal_meta['jitter_days'], seed=_primal_meta['seed'], window=_primal_meta['window'], bay_weight=_primal_meta['bay_weight'])
        _primal_meta.update(_primal_bias_diag)

    def _primal_order(plan, plan_bucket, phase):
        if not _primal_realization_enabled or _primal_bias is None:
            return None
        order, diag = _primal_realization_order(prob_info, plan, _primal_bias, plan_bucket=plan_bucket)
        calls = int(_primal_meta['calls']) + 1
        _primal_meta['calls'] = calls
        phase_key = 'extension_calls' if str(phase).startswith('extension') else 'core_calls'
        _primal_meta[phase_key] = int(_primal_meta[phase_key]) + 1
        running_values = {'mean_moved_fraction': float(diag['moved_fraction']), 'mean_abs_rank_shift': float(diag['mean_abs_rank_shift'])}
        for key, value in running_values.items():
            previous = float(_primal_meta[key])
            _primal_meta[key] = previous + (value - previous) / calls
        _primal_meta['max_abs_rank_shift'] = max(float(_primal_meta['max_abs_rank_shift']), float(diag['max_abs_rank_shift']))
        event = {'phase': str(phase), **diag}
        if _primal_meta['first_order'] is None:
            _primal_meta['first_order'] = event
        _primal_meta['last_order'] = event
        return order
    if run_meta is not None:
        run_meta['primal_realization'] = _primal_meta

    def _lns_walk(*args, **kwargs):
        return lns_sa(*args, **kwargs)
    _bsplit = 0.0
    _both_rounds = None
    _both_rounds_completed = None
    _both_rounds_planned_s = None
    _sampler = False

    def _publish_route_meta():
        if run_meta is None:
            return
        actual_route = 'both' if _both else 'heavy' if _slr else 'light'
        run_meta.update({'route': actual_route, 'forced_route': _route or None, 'raw_dr': float(_dr), 'pressure': float(_P), 'abar': float(_abar), 'workable_s': float(_wk), 'pup_enabled': True, 'pup_gate': bool(_pup), 'pup_forced': bool(_pup_forced), 'pup_promoted': bool(_pup and (not _route) and _both), 'pup_forced_light': bool(_pup and (not _route) and _pup_force_light), 'both_split': None if _factory_producer or _both_rounds is not None else float(_bsplit) if _both else None, 'both_rounds': int(_both_rounds) if _both and _both_rounds is not None else None, 'both_rounds_completed': int(_both_rounds_completed) if _both_rounds_completed is not None else None, 'both_rounds_planned_s': float(_both_rounds_planned_s) if _both_rounds_planned_s is not None else None, 'multi_share': float(_multi_share), 'lns_share': float(_lns_share), 'sampler': bool(_sampler), 'heavy_factory': False, 'factory_producer': bool(_factory_producer), 'factory_role': _factory_role or None, 'factory_handoff_start': bool(_factory_shared_start), 'lr_relative_dispatch': bool(_lr_relative_dispatch), 'lr_blocked_weight': float(_lr_blocked_weight), 'lr_zobrist_dedup': bool(_lr_zobrist_dedup), 'mseed_start': bool(start_assign is not None and (not _factory_shared_start)), 'mseed_baseline_start': bool(baseline_start_assign is not None and (not _factory_shared_start)), 'mseed_start_source': 'objective' if start_assign is not None and (not _factory_shared_start) else None, 'light_profile': _light_profile_requested if _light_alap_nonstrict else 'search', 'alap_nonstrict': _alap_ns_meta})
        if route_meta_cb is not None:
            route_meta_cb(run_meta)
    _slr_snap = None
    w1 = float(prob_info.get('weights', {}).get('w1', 1.0))
    if not _slr:
        _publish_route_meta()
    if _slr:
        from plan_lr import solve_plan_lr
        _rc_shadow = os.environ.get('OGC_RC_SHADOW', '') == '1'
        _rc_on = _rc_shadow
        if _rc_on:
            from rc_variants import discrepant_plan, reduced_cost_options
        if not _both:
            multi_budget = 0.0
            lns_budget = 0.0
        _native_border = not _route and 0.6 <= _dr < 0.7 and True
        _heavy_factory_sampler = not _both and os.environ.get('OGC_HEAVY_FACTORY_SAMPLER', '') == '1'
        _factory_signal = None
        if _heavy_factory_sampler and _incumbent_trace.get('enabled', False) and (run_meta is not None):
            _factory_signal = {'enabled': True, 'schema_version': 1, 'clock': _incumbent_trace['clock'], 'profile': os.environ.get('OGC_HEAVY_FACTORY_PROFILE', 'kappa'), 'aeff_scale': float(os.environ.get('OGC_AEFF_SCALE', '1.0')), 'planned_rounds': None, 'rounds_completed': 0, 'round_trace': [], 'fixed_point_events': [], 'first_fixed_point': None, 'first_strict_improvement': None, 'last_strict_improvement': None, 'last_round_strict_improvement': None, 'round_loop_end_elapsed_s': None, 'seconds_since_last_round_strict_improvement_at_round_loop_end': None, 'rounds_after_last_strict_improvement': None, 'slr_end_elapsed_s': None, 'seconds_since_last_strict_improvement_at_slr_end': None, 'seconds_since_last_round_strict_improvement_at_slr_end': None}
            run_meta['factory_signal'] = _factory_signal

        def _factory_register_improvement(strict, event, objective, elapsed_s, phase, round_index=None):
            if _factory_signal is None or not strict:
                return
            row = {'event': str(event), 'elapsed_s': round(float(elapsed_s), 3), 'objective': float(objective), 'phase': str(phase)}
            if round_index is not None:
                row['round'] = int(round_index)
            if _factory_signal['first_strict_improvement'] is None:
                _factory_signal['first_strict_improvement'] = dict(row)
            _factory_signal['last_strict_improvement'] = dict(row)
            if event == 'slr_round':
                _factory_signal['last_round_strict_improvement'] = dict(row)
        _sampler = _both and (os.environ.get('OGC_ARM_SAMPLER', '') == '1' or _native_border) or _heavy_factory_sampler
        if _sampler and os.environ.get('OGC_NO_GHAT', '') != '1':
            from plan_core import _layer0_area as _l0a
            _bays_A = [b['width'] * b['height'] for b in prob_info['bays']]
            if hasattr(_aeff, '__len__'):
                _cap = float(sum((a * x for a, x in zip(_bays_A, _aeff))))
            else:
                _cap = float(_aeff) * float(sum(_bays_A))
            _items = [(b['due_date'] - b['processing_time'], b['release_time'], _l0a(b) * b['processing_time'] / _cap) for b in prob_info['blocks']]
            _items.sort(key=lambda x: (x[0], x[1]))
            _C = 0.0
            _nl = 0
            for _ls, _R, _s in _items:
                _st = _R if _R > _C else _C
                _C = _st + _s
                if _st - _ls > 1e-09:
                    _nl += 1
            _qhat = _nl / max(1, len(_items))
            _R0 = min((x[1] for x in _items))
            _R1 = max((x[1] for x in _items))
            _rho = len(_items) / max(1.0, _R1 - _R0) * (sum((x[2] for x in _items)) / max(1, len(_items)))
            _sampler = _qhat > 0.35 and _rho < 1.5
        _both_rounds_raw = os.environ.get('OGC_BOTH_ROUNDS', '').strip()
        if _both and _both_rounds_raw:
            _both_rounds = max(1, int(_both_rounds_raw))
        if _factory_producer:
            _bsplit = 1.0
            _leg = _wk
        elif _both_rounds is not None:
            _bsplit = 0.0
            _leg = _wk
        else:
            _bsplit = float(os.environ.get('OGC_BOTH_SPLIT', '0.8' if _sampler else '0.35' if _pup else '0.5'))
            _leg = _wk * _bsplit if _both else _wk
        _slr_end = time.time() + _leg
        _publish_route_meta()

        def _slr_left():
            return _slr_end - time.time() if _both else remaining() - tail_reserve
        u, e_prev, lam = ({}, None, None)
        _rc_slip = {}
        _ww_on = _factory_producer or (not _both and True) or os.environ.get('OGC_WHO_WAITS', '') == '1'

        def _ww_prob(_plan):
            if not _ww_on:
                return prob_info
            return _who_waits_locked_problem(prob_info, _plan)
        arm_wall = max(5.0, probe_wall)
        pull_min = 6.0 if _both and _both_rounds is not None else min(18.0, max(6.0, 0.2 * _leg)) if _both else 18.0
        _plan_last = None
        rnd = 0
        rcap = int(os.environ.get('OGC_RCAP', '200'))
        _rd_leg = _rd * 0.55 if _sampler else _rd
        if _both_rounds is not None:
            det_rounds = min(rcap, int(_both_rounds))
            _both_rounds_planned_s = min(_wk, det_rounds * _rd_leg + pull_min + 4.0)
        else:
            det_rounds = max(2, min(rcap, int(max(0.0, _leg - pull_min - 4.0) / _rd_leg)))
        _factory_handoff_state = {'best': None, 'sent': False}

        def _factory_offer_handoff(_plan, _assign, _rounds_completed, plan_bucket=0, force=False):
            if not _factory_producer:
                return
            objective = float(obj_of(_assign))
            prior = _factory_handoff_state['best']
            if prior is None or objective < prior['objective'] - 1e-09:
                _factory_handoff_state['best'] = {'plan': {int(block_id): dict(row) for block_id, row in _plan.items()}, 'assignment': {int(block_id): dict(row) for block_id, row in _assign.items()}, 'objective': objective, 'rounds_completed': int(_rounds_completed), 'plan_bucket': int(plan_bucket)}
            if _factory_handoff_state['sent'] or factory_handoff_cb is None or (not force and _rounds_completed < 2):
                return
            payload = dict(_factory_handoff_state['best'])
            payload['rounds_completed'] = int(_rounds_completed)
            try:
                factory_handoff_cb(payload)
            except Exception as exc:
                if run_meta is not None:
                    run_meta['factory_handoff_error'] = f'{type(exc).__name__}: {exc}'
                return
            _factory_handoff_state['sent'] = True
            if run_meta is not None:
                run_meta.update({'factory_handoff_sent': True, 'factory_handoff_rounds': int(_rounds_completed), 'factory_handoff_objective': float(payload['objective']), 'factory_handoff_protected': True})
        if _factory_signal is not None:
            _factory_signal['planned_rounds'] = int(det_rounds)
            _factory_signal['planned_leg_s'] = float(_leg)
        _rc_rounds = set()
        if _rc_shadow:
            _rc_rounds.update((1, max(2, det_rounds // 2)))
        _msch_sub = max(2, min(3, det_rounds // 2))
        _msch = not _both and os.environ.get('OGC_NO_HEAVY_MS', '') != '1' and (det_rounds >= _msch_sub + 1)
        _msch_L0 = None
        _msch_menu = ['tpure', ('bkt', 5), 2.0, ('asc', 0.92), 'smooth', ('bkt', 8), 4.0, ('asc', 0.85), 0.5, 0.25, 8.0]
        _msch_r240 = det_rounds
        _rr240 = min(90.0, max(20.0, 0.12 * 240.0))
        _wk240 = max(0.0, 240.0 - _rr240 - _est_pass - _oh_full)
        _leg240 = _wk240 * _bsplit if _both else _wk240
        _msch_r240 = max(2, min(rcap, int(max(0.0, _leg240 - pull_min - 4.0) / _rd_leg)))
        _msch_bucket = 0
        _msch_asc = 1.0
        _msch_pool = []
        _msch_ci = 0
        _msch_cr = 0
        _karm_on = os.environ.get('OGC_NO_KARM', '') != '1' and _dr < 0.95
        _karm_k, _karm_at = (0.9, max(1, det_rounds // 3))
        _karm_spec = os.environ.get('OGC_KARM', '')
        if _karm_spec:
            _kp = _karm_spec.split(':')
            _karm_k = float(_kp[0])
            if len(_kp) > 1:
                _karm_at = int(_kp[1])
            _karm_on = os.environ.get('OGC_NO_KARM', '') != '1'
        _karm_hi = False
        if not _karm_on and (not _karm_spec) and (os.environ.get('OGC_NO_KARM', '') != '1') and (os.environ.get('OGC_NO_KARM_HI', '') != '1'):
            _karm_on = True
            _karm_hi = True
            _karm_k = 0.95
            _karm_at = max(2, _msch_r240)
        _karm_done = False
        _msch_walk_iters = 0

        def _tpure_lam0(_L0):
            _H = _L0.shape[1]
            _cap_tot = float((_aeff * _bA).sum()) if hasattr(_aeff, '__len__') else float(_aeff) * float(_bA.sum())
            _idx = sorted(range(len(_bl)), key=lambda i: (_bl[i]['due_date'] - _bl[i]['processing_time'], _bl[i]['release_time']))
            _Bhat = _np.zeros(_H)
            _Cc = 0.0
            for _i in _idx:
                _b = _bl[_i]
                _ls = _b['due_date'] - _b['processing_time']
                _s = _l0a(_b) * _b['processing_time'] / _cap_tot
                _st = _b['release_time'] if _b['release_time'] > _Cc else _Cc
                _Cc = _st + _s
                if _st - _ls > 1e-09:
                    _t0 = max(0, int(_ls))
                    _t1 = min(_H, int(_np.ceil(_st + _s)))
                    if _t1 > _t0:
                        _Bhat[_t0:_t1] += _l0a(_b)
            _Bhat /= max(float(_Bhat.max()), 1e-09)
            return 0.5 * float(_L0.max()) * _np.tile(_Bhat, (_nbay, 1))
        _fp_break = os.environ.get('OGC_NO_FPBREAK', '') != '1'
        _fp_streak = 0
        _fp_u_prev = None
        _rearmed = False
        _factory_phase = 'cold'
        if not _both and os.environ.get('OGC_NO_HEAVY_GREEDY', '') != '1':
            from plan_core import greedy_plan
            _ab = float(_np.mean(_aeff)) if hasattr(_aeff, '__len__') else float(_aeff)
            _gp = greedy_plan(prob_info, alpha=_ab)
            _gplan = {i: {'bay_id': j, 'start': t} for i, (j, t) in _gp.items()}
            _g_order = _primal_order(_gplan, 0, 'core-greedy')
            _g_assign, _ = construct(_ww_prob(_gplan), timelimit=60.0, plan=_gplan, plan_bays=False, plan_target_w=0.3 * w1, insert_ids=_g_order, insert_order='given', geo_cache=geo_cache, pair_memo=pair_memo, rel_memo=rel_memo, verbose=False)
            candidates.append(_g_assign)
            if _incumbent_trace.get('enabled', False):
                _g_elapsed = time.time() - trace_origin
                _g_obj = float(obj_of(_g_assign))
                _g_strict = _trace_objective(_g_obj, 'slr_greedy', elapsed_s=_g_elapsed, stage='slr', phase=_factory_phase)
                _factory_register_improvement(_g_strict, 'slr_greedy', _g_obj, _g_elapsed, _factory_phase)
        while rnd < det_rounds and _slr_left() > pull_min + 6.0:
            if _karm_on and (not _karm_done) and (rnd >= _karm_at) and (_msch_ci == 0):
                _karm_done = True
                _msch = False
                _msch_asc = _karm_k
                lam = None
                _factory_phase = f'kappa:{_karm_k:.3g}'
                u, e_prev = ({}, None)
                _fp_streak, _fp_u_prev = (0, None)
            _rc_take = _rc_shadow and rnd in _rc_rounds
            _rc_state = {} if _rc_take else None
            _lr_round_stats = {} if _lr_feedback_meta['enabled'] else None
            plan, lam = solve_plan_lr(prob_info, alpha=_np.clip(_aeff * _msch_asc, 0.25, 1.0) if _msch_asc != 1.0 else _aeff, timelimit=10.0 ** 9, iters=100 if _sampler else 300, price=u if u else None, prox=e_prev if e_prev else None, prox_w=0.2 * w1, lam0=lam, rc_state=_rc_state, stats=_lr_round_stats, relative_dispatch=_lr_relative_dispatch, blocked_weight=_lr_blocked_weight, zobrist_dedup=_lr_zobrist_dedup)
            if _lr_round_stats is not None:
                _lr_feedback_meta['rounds'].append({'round': int(rnd), **_lr_round_stats})
            if _msch and _msch_L0 is None:
                _msch_L0 = lam
            _rc_meta = None
            _rc_plan = None
            if _rc_take:
                try:
                    _rc_ranked = reduced_cost_options(prob_info, plan, _rc_state['lam'], price=u if u else None, prox=e_prev if e_prev else None, prox_w=0.2 * w1, top_k=4)
                    _rc_plan, _rc_meta = discrepant_plan(prob_info, plan, _rc_ranked, _rc_slip, rnd, max_changes=1, pool_width=8, gap_fraction=0.25)
                except Exception as _rc_exc:
                    _rc_meta = None
                finally:
                    pass
            _realization_bucket = _msch_bucket if _msch_bucket else (5, 0, 3)[rnd % 3] if _sampler else int(os.environ.get('OGC_PLAN_BUCKET', '0'))
            _realization_order = _primal_order(plan, _realization_bucket, 'core')
            r_assign, rstats = construct(_ww_prob(plan), timelimit=60.0, plan=plan, plan_bays=False, plan_target_w=0.3 * w1, plan_bucket=_realization_bucket, insert_ids=_realization_order, insert_order='given', geo_cache=geo_cache, pair_memo=pair_memo, rel_memo=rel_memo, verbose=False)
            candidates.append(r_assign)
            _factory_offer_handoff(plan, r_assign, rnd + 1, plan_bucket=_realization_bucket)
            _round_obj = None
            _round_elapsed = None
            _round_local_elapsed = None
            _round_strict = False
            if _incumbent_trace.get('enabled', False):
                _round_elapsed = time.time() - trace_origin
                _round_local_elapsed = time.time() - t_start
                _round_obj = float(obj_of(r_assign))
                _round_strict = _trace_objective(_round_obj, 'slr_round', elapsed_s=_round_elapsed, stage='slr', round=int(rnd), phase=_factory_phase)
            if _rc_shadow and _rc_meta is not None and _rc_meta['changed']:
                try:
                    _rc_assign, _ = construct(_ww_prob(_rc_plan), timelimit=60.0, plan=_rc_plan, plan_bays=True, plan_target_w=0.3 * w1, plan_bucket=_realization_bucket, insert_ids=_primal_order(_rc_plan, _realization_bucket, 'core-shadow'), insert_order='given', geo_cache=geo_cache, pair_memo=pair_memo, rel_memo=rel_memo, verbose=False)
                    candidates.append(_rc_assign)
                    if _incumbent_trace.get('enabled', False):
                        _trace_objective(float(obj_of(_rc_assign)), 'slr_shadow', stage='slr', round=int(rnd), phase=_factory_phase)
                except Exception as _rc_exc:
                    pass
                finally:
                    pass
            arm_wall = rstats['wall_s']
            if _rc_on:
                _rc_slip = {i: max(0, int(r_assign[i]['entry_time']) - int(p['start'])) for i, p in plan.items() if i in r_assign}
            cap = 0.5 * w1
            for i, p in plan.items():
                if i in r_assign:
                    v = u.get(i, 0.0) + 0.03 * w1 * (p['start'] - r_assign[i]['entry_time'])
                    u[i] = max(-cap, min(cap, v))
            _plan_last = plan
            e_new = {i: a['entry_time'] for i, a in r_assign.items()}
            if e_prev == e_new and _fp_u_prev == u:
                _fp_streak += 1
            else:
                _fp_streak = 0
            _fp_u_prev = dict(u)
            e_prev = e_new
            if _factory_signal is not None:
                if _round_obj is None:
                    _round_obj = float(obj_of(r_assign))
                    _round_elapsed = time.time() - trace_origin
                    _round_local_elapsed = time.time() - t_start
                _factory_signal['round_trace'].append({'round': int(rnd), 'elapsed_s': round(float(_round_elapsed), 3), 'arm_elapsed_s': round(float(_round_local_elapsed), 3), 'phase': _factory_phase, 'objective': float(_round_obj), 'incumbent_objective': float(_incumbent_trace['best_objective']) if isinstance(_incumbent_trace.get('best_objective'), (int, float)) else None, 'strict_incumbent_improvement': bool(_round_strict), 'fixed_point_streak': int(_fp_streak)})
                _factory_signal['rounds_completed'] = int(len(_factory_signal['round_trace']))
                _factory_register_improvement(_round_strict, 'slr_round', _round_obj, _round_elapsed, _factory_phase, round_index=rnd)
                if _fp_streak == 2:
                    _fp_event = {'round': int(rnd), 'elapsed_s': round(float(_round_elapsed), 3), 'phase': _factory_phase}
                    _factory_signal['fixed_point_events'].append(_fp_event)
                    if _factory_signal['first_fixed_point'] is None:
                        _factory_signal['first_fixed_point'] = dict(_fp_event)
            rnd += 1
            if _msch:
                _msch_pool.append((lam, obj_of(r_assign)))
                _msch_cr += 1
                _msch_ready = _fp_streak >= 2 or rnd >= _msch_r240 if _msch_ci == 0 else _msch_cr >= _msch_sub
                if _msch_ready:
                    if _karm_hi and _karm_on and (not _karm_done) and (_msch_ci == 0) and (rnd >= _karm_at):
                        continue
                    if _msch_ci >= len(_msch_menu):
                        break
                    _pert = _msch_menu[_msch_ci]
                    _msch_ci += 1
                    _factory_phase = f'menu:{_pert}'
                    _msch_cr = 0
                    _msch_bucket = 0
                    _msch_asc = 1.0
                    _lam_hat, _lh_obj = min(_msch_pool, key=lambda x: x[1])
                    if _pert == 'tpure':
                        lam = _tpure_lam0(_msch_L0)
                    elif _pert == 'smooth':
                        _k5 = _np.ones(5) / 5.0
                        lam = _np.apply_along_axis(lambda v: _np.convolve(v, _k5, mode='same'), 1, _lam_hat)
                    elif isinstance(_pert, tuple) and _pert[0] == 'bkt':
                        lam = _np.array(_lam_hat, copy=True)
                        _msch_bucket = int(_pert[1])
                    elif isinstance(_pert, tuple) and _pert[0] == 'asc':
                        lam = None
                        _msch_asc = float(_pert[1])
                    else:
                        lam = _lam_hat * _pert
                    u, e_prev = ({}, None)
                    _fp_streak, _fp_u_prev = (0, None)
            if _fp_break and _fp_streak >= 2 and (not _msch):
                if not _both and (not _rearmed) and (rnd < det_rounds):
                    _av2, _seen2 = _sense_av(min(candidates, key=obj_of))
                    _raised = _np.where(_seen2, _np.maximum(_aeff, _av2), _aeff)
                    if float(_np.max(_raised - _aeff)) >= 0.01:
                        _aeff = _raised
                        lam = None
                        _fp_streak = 0
                        _fp_u_prev = None
                        _rearmed = True
                        continue
                break
        if _both_rounds is not None:
            _both_rounds_completed = int(rnd)
            _publish_route_meta()
        if run_meta is not None:
            run_meta.update({'lr_round_cap': int(rcap), 'lr_rounds_planned': int(det_rounds), 'lr_rounds_completed': int(rnd)})
        if _factory_producer and (not _factory_handoff_state['sent']) and (_factory_handoff_state['best'] is not None):
            _factory_offer_handoff(_factory_handoff_state['best']['plan'], _factory_handoff_state['best']['assignment'], max(1, rnd), plan_bucket=_factory_handoff_state['best'].get('plan_bucket', 0), force=True)
        if _factory_signal is not None:
            _factory_signal['round_loop_end_elapsed_s'] = round(time.time() - trace_origin, 3)
            _last_round = _factory_signal['last_round_strict_improvement']
            if _last_round is not None:
                _factory_signal['seconds_since_last_round_strict_improvement_at_round_loop_end'] = round(_factory_signal['round_loop_end_elapsed_s'] - float(_last_round['elapsed_s']), 3)
                _factory_signal['rounds_after_last_strict_improvement'] = max(0, int(rnd) - int(_last_round['round']) - 1)
        if _msch and _slr_left() > pull_min + 10.0:
            _msch_walk_iters = max(0, min(2000, int((_slr_left() - pull_min - 6.0 - 60.0) / _it)))
        if _msch and _msch_walk_iters >= 100 and (_slr_left() > pull_min + 6.0):
            from lns import lns_sa
            try:
                _msch_seed = min(candidates, key=obj_of)
                _msch_walked, _ = _lns_walk(prob_info, _msch_seed, timelimit=max(10.0, _slr_left() - pull_min - 60.0), max_iters=_msch_walk_iters, seed=_arm_seed, geo_cache=geo_cache, pair_memo=pair_memo, rel_memo=rel_memo, verbose=False)
                candidates.append(_msch_walked)
                if _incumbent_trace.get('enabled', False):
                    _walk_elapsed = time.time() - trace_origin
                    _walk_obj = float(obj_of(_msch_walked))
                    _walk_strict = _trace_objective(_walk_obj, 'slr_walk', elapsed_s=_walk_elapsed, stage='slr', phase=_factory_phase)
                    _factory_register_improvement(_walk_strict, 'slr_walk', _walk_obj, _walk_elapsed, _factory_phase)
            except Exception:
                pass
            finally:
                pass
        if _both:
            _slr_snap = {'u': dict(u), 'lam': lam, 'e_prev': dict(e_prev) if e_prev else e_prev}
        if os.environ.get('OGC_NO_SLR_PULL', '') != '1' and candidates:
            from pull import pull_forward
            try:
                p_assign, _ = pull_forward(prob_info, min(candidates, key=obj_of), timelimit=20.0, pref=True, try_cost=_pt, geo_cache=geo_cache, pair_memo=pair_memo, rel_memo=rel_memo, verbose=False)
                candidates.append(p_assign)
                if _incumbent_trace.get('enabled', False):
                    _pull_elapsed = time.time() - trace_origin
                    _pull_obj = float(obj_of(p_assign))
                    _pull_strict = _trace_objective(_pull_obj, 'slr_pull', elapsed_s=_pull_elapsed, stage='slr', phase=_factory_phase)
                    _factory_register_improvement(_pull_strict, 'slr_pull', _pull_obj, _pull_elapsed, _factory_phase)
            except Exception as _pe:
                pass
            finally:
                pass
        if _both and _factory_producer:
            lns_budget = 0.0
            multi_budget = 0.0
        elif _both:
            _w2 = max(0.0, remaining() - tail_reserve)
            lns_budget = _w2 * 0.55
            multi_budget = max(0.0, _w2 - lns_budget)
        if _factory_signal is not None:
            _factory_signal['slr_end_elapsed_s'] = round(time.time() - trace_origin, 3)
            _last_factory = _factory_signal['last_strict_improvement']
            if _last_factory is not None:
                _factory_signal['seconds_since_last_strict_improvement_at_slr_end'] = round(_factory_signal['slr_end_elapsed_s'] - float(_last_factory['elapsed_s']), 3)
            _last_round = _factory_signal['last_round_strict_improvement']
            if _last_round is not None:
                _factory_signal['seconds_since_last_round_strict_improvement_at_slr_end'] = round(_factory_signal['slr_end_elapsed_s'] - float(_last_round['elapsed_s']), 3)
    if _factory_role == 'bucket_lns' and _factory_shared_start and isinstance(factory_handoff.get('plan'), dict):
        try:
            _factory_plan = factory_handoff['plan']
            _factory_bucket = max(1, int(os.environ.get('OGC_FACTORY_PLAN_BUCKET', '5')))
            _factory_source_bucket = int(factory_handoff.get('plan_bucket', 0))
            if _factory_bucket == _factory_source_bucket:
                _factory_bucket = 3 if _factory_source_bucket != 3 else 8
            _bucket_assign, _bucket_stats = construct(_factory_locked_prob, timelimit=60.0, plan=_factory_plan, plan_bays=False, plan_target_w=0.3 * w1, plan_bucket=_factory_bucket, insert_ids=None, insert_order='given', geo_cache=geo_cache, pair_memo=pair_memo, rel_memo=rel_memo, verbose=False)
            candidates.append(_bucket_assign)
            _factory_lns_start = _bucket_assign
            if run_meta is not None:
                run_meta.update({'factory_bucket': int(_factory_bucket), 'factory_source_bucket': int(_factory_source_bucket), 'factory_bucket_objective': float(obj_of(_bucket_assign)), 'factory_bucket_wall_s': float(_bucket_stats.get('wall_s', 0.0))})
            _stage('factory-bucket')
        except Exception as _factory_bucket_exc:
            if run_meta is not None:
                run_meta['factory_bucket_error'] = f'{type(_factory_bucket_exc).__name__}: {_factory_bucket_exc}'
        finally:
            pass
    _stage('slr')
    det_passes = det_iters = 0
    if _factory_role:
        _leg_lt = max(20.0, _wk - 10.0)
        det_passes = 0
        det_iters = max(100, min(4000, int(0.85 * _leg_lt / _it))) if _factory_deep else 0
    else:
        if _both and _both_rounds is not None:
            _leg_lt = max(20.0, _wk - float(_both_rounds_planned_s or 0.0) - 25.0)
        else:
            _leg_lt = max(20.0, _wk * (1.0 - _bsplit if _both else 1.0) - 25.0)
        det_passes = max(3, min(400, int(_multi_share * _leg_lt / _ps)))
        det_iters = max(100, min(4000, int(_lns_share * _leg_lt / _it)))
    if start_assign is not None:
        candidates.append(start_assign)
        if run_meta is not None:
            if _factory_shared_start:
                run_meta['factory_shared_start_objective'] = float(obj_of(start_assign))
                run_meta['factory_shared_start_protected'] = True
            else:
                run_meta['mseed_start_objective'] = float(obj_of(start_assign))
                run_meta['mseed_baseline_start_objective'] = float(obj_of(baseline_start_assign)) if baseline_start_assign is not None else None
    elif multi_budget > 5.0:
        try:
            m_assign, _ = construct_multi(prob_info, timelimit=2.0 * _multi_share * _leg_lt + 30.0, use_legacy=True, geo_cache=geo_cache, pair_memo=pair_memo, rel_memo=rel_memo, scout_skip=None if probe_degraded else ('edd', None), max_passes=det_passes, var_offset=_arm_seed * 17, verbose=False)
            candidates.append(m_assign)
        except Exception:
            pass
        finally:
            pass
    best = _trajectory_best()
    _stage('multi')
    if _light_alap_nonstrict:
        _alap_seed = _trajectory_best()
        _alap_budget = max(0.0, remaining() - 4.0)
        _alap_ns_meta['budget_s'] = float(_alap_budget)
        if _alap_budget > 2.0:
            _alap_ns_meta['attempted'] = True
            try:
                from alap_batch import alap_batch_walk
                _alap_best, _alap_stats, _alap_terminal = alap_batch_walk(prob_info, _alap_seed, timelimit=_alap_budget, quality_gap=_alap_ns_meta['quality_gap'], batch_size=_alap_ns_meta['batch_size'], anchors=_alap_ns_meta['anchors'], max_rounds=_alap_ns_meta['max_rounds'], max_proposals_per_round=_alap_ns_meta['max_proposals_per_round'], candidate_timelimit=_alap_ns_meta['candidate_timelimit_s'], latest_scaffold=_alap_ns_meta['latest_scaffold'], repair_mode=_alap_ns_meta['repair_mode'], cpsat_det_units=_alap_ns_meta['cpsat_det_units'], cpsat_workers=_alap_ns_meta['cpsat_workers'], seed=_arm_seed, geo_cache=geo_cache, pair_memo=pair_memo, rel_memo=rel_memo, verbose=False)
                candidates.append(_alap_best)
                _alap_ns_meta['result'] = _alap_stats
                _alap_ns_meta['seed_objective'] = float(obj_of(_alap_seed))
                _alap_ns_meta['protected_objective'] = float(obj_of(_alap_best))
                _alap_ns_meta['terminal_objective'] = float(obj_of(_alap_terminal))
            except Exception as _alap_exc:
                _alap_ns_meta['error'] = f'{type(_alap_exc).__name__}: {_alap_exc}'
            finally:
                pass
        _stage('alap-ns')
    w3 = float(prob_info.get('weights', {}).get('w3', 1.0))
    w2 = float(prob_info.get('weights', {}).get('w2', 1.0))
    tot_b, (_, o2_b, o3_b) = _objective_of(prob_info, best)
    pref_gate = tot_b > 0 and w3 * o3_b >= 0.3 * tot_b
    seat_ran = False
    if (not _light_alap_nonstrict) and (not _factory_deep) and pref_gate and remaining() > 25.0:
        from plan_core import solve_seat_plan
        from pull import seat_move
        try:
            target, sstat = solve_seat_plan(prob_info, timelimit=6.0, incumbent=best, workers=4)
            _seatgap_log(ev='chart', stage='seat', arm=_arm_seed, status=sstat['status'], promise=sstat['obj23'], inc23=w2 * o2_b + w3 * o3_b, admit=bool(target and sstat['obj23'] < w2 * o2_b + w3 * o3_b))
            if target and sstat['obj23'] < w2 * o2_b + w3 * o3_b:
                sc_assign, _sc_st = seat_move(prob_info, best, target, timelimit=25.0, try_cost=_st, drain_cost=_pt, geo_cache=geo_cache, pair_memo=pair_memo, rel_memo=rel_memo, verbose=False)
                if sc_assign is not None:
                    candidates.append(sc_assign)
                    best = _trajectory_best()
                    if _sc_st.get('accepted', 0) > 0:
                        seat_ran = True
        except Exception as _pe:
            pass
        finally:
            pass
    _stage('seat')
    if not _light_alap_nonstrict and lns_budget > 0.0 and (remaining() > 12.0) and (not seat_ran):
        try:
            if _factory_deep:
                _lns_timelimit = 2.0 * 0.85 * _leg_lt + 20.0
            else:
                _lns_timelimit = 2.0 * _lns_share * _leg_lt + 30.0
            _lns_prob = _factory_locked_prob if _factory_deep and _factory_locked_prob is not None else prob_info
            _lns_start = _factory_lns_start if _factory_role == 'bucket_lns' and _factory_lns_start is not None else best
            improved, _ = _lns_walk(_lns_prob, _lns_start, timelimit=_lns_timelimit, max_iters=det_iters, seed=_arm_seed, geo_cache=geo_cache, pair_memo=pair_memo, rel_memo=rel_memo, verbose=False)
            candidates.append(improved)
            best = _trajectory_best()
        except Exception:
            pass
        finally:
            pass
    _stage('lns')
    _pp_seed = _trajectory_best()
    _light_alap = not _slr and _objective_of(prob_info, _pp_seed)[1][0] <= 1e-09
    if (not _light_alap_nonstrict) and (not _factory_deep) and (pref_gate or _light_alap):
        pp_tl = min(60.0, 0.25 * total_budget)
        if _light_alap and (not pref_gate):
            pp_tl = min(pp_tl, max(0.0, (remaining() - 4.0) / 2.0))
        if pp_tl > 5.0:
            from pull import pull_forward
            try:
                pp_assign, _pp_st = pull_forward(prob_info, _pp_seed, timelimit=pp_tl, pref=True, try_cost=_pt, geo_cache=geo_cache, pair_memo=pair_memo, rel_memo=rel_memo, verbose=False, alap_when_ontime=_light_alap)
                candidates.append(pp_assign)
            except Exception as _pe:
                pass
            finally:
                pass
    if (not _light_alap_nonstrict) and (os.environ.get('OGC_NO_EXT', '') != '1') and remaining() > 15.0:
        _ext_base = _trajectory_best()
        if _slr and (not _both):
            from plan_lr import solve_plan_lr as _ext_plr
            _menu = [(5, 1.0), (0, 0.92), (5, 0.92), (3, 0.96), (8, 1.0), (0, 0.85)]
            _ext_stop = max(12.0, 0.8 * arm_wall + 6.0)
            for _ext_index, (_bkt, _asc) in enumerate(_menu):
                if remaining() < _ext_stop:
                    break
                try:
                    _al = _np.clip(_aeff * _asc, 0.25, 1.0)
                    _pl, _ = _ext_plr(prob_info, alpha=_al, timelimit=4.0)
                    _ra, _ = construct(prob_info, timelimit=min(60.0, max(10.0, remaining() - 4.0)), plan=_pl, plan_bays=False, plan_target_w=0.3 * w1, plan_bucket=_bkt, insert_ids=_primal_order(_pl, _bkt, 'extension'), insert_order='given', geo_cache=geo_cache, pair_memo=pair_memo, rel_memo=rel_memo, verbose=False)
                    candidates.append(_ra)
                except Exception:
                    break
                finally:
                    pass
            _tie_in = None
            if remaining() > 20.0 and obj_of(_trajectory_best()) < obj_of(_ext_base) - 1e-09:
                from pull import pull_forward
                try:
                    _pin = _trajectory_best()
                    _tie_in = _pin
                    _pa, _ = pull_forward(prob_info, _pin, timelimit=min(30.0, remaining() - 8.0), pref=True, try_cost=_pt, geo_cache=geo_cache, pair_memo=pair_memo, rel_memo=rel_memo, verbose=False)
                    candidates.append(_pa)
                except Exception as _pe:
                    pass
                finally:
                    pass
            if remaining() > 16.0:
                from pull import pull_forward
                try:
                    _ta, _ = pull_forward(prob_info, _tie_in if _tie_in is not None else _trajectory_best(), timelimit=min(30.0, remaining() - 8.0), pref=True, try_cost=_pt, geo_cache=geo_cache, pair_memo=pair_memo, rel_memo=rel_memo, verbose=False, tie_open=True)
                    candidates.append(_ta)
                except Exception as _pe:
                    pass
                finally:
                    pass
        else:
            _menu_led = False
            _pre_pool = candidates
            _menu_pool = []
            if _both and _slr_snap is not None and True:
                from plan_lr import solve_plan_lr as _ext_plr
                _pre_pool = list(candidates)
                _u0 = _slr_snap['u']
                _lam0 = _slr_snap['lam']
                _e0 = _slr_snap['e_prev']
                _ext_low = obj_of(_ext_base)
                _menu = [(1.0, 1.0, 5, True), (None, 1.0, 0, False), (1.0, 0.92, 0, False), (1.0, 1.0, 3, True), (0.5, 1.0, 0, True), (1.0, 1.08, 0, False), (0.25, 1.0, 0, True)]
                _miss = 0
                for _msc, _asc, _bkt, _warm in _menu:
                    if _miss >= 3 or remaining() <= max(10.0, _ps + 6.0):
                        break
                    if _msc is not None and _msc < 1.0 and (not _u0):
                        continue
                    try:
                        _al = _np.clip(_aeff * _asc, 0.25, 1.0) if _asc != 1.0 else _aeff
                        _pr = {i: _msc * v for i, v in _u0.items()} if _msc is not None and _u0 else None
                        _pl, _ = _ext_plr(prob_info, alpha=_al, timelimit=10.0 ** 9, iters=100, price=_pr, prox=_e0 if _msc is not None and _e0 else None, prox_w=0.2 * w1, lam0=_lam0 if _warm else None)
                        _ra, _ = construct(prob_info, timelimit=60.0, plan=_pl, plan_bays=False, plan_target_w=0.3 * w1, plan_bucket=_bkt, geo_cache=geo_cache, pair_memo=pair_memo, rel_memo=rel_memo, verbose=False)
                    except Exception:
                        break
                    finally:
                        pass
                    candidates.append(_ra)
                    _menu_pool.append(_ra)
                    _o = obj_of(_ra)
                    if _o < _ext_low - 1e-09:
                        _ext_low = _o
                        _miss = 0
                    else:
                        _miss += 1
                _u = dict(_u0)
                _lam = _lam0
                _e = _e0
                for _rn in (1, 2):
                    if _miss >= 3 or remaining() <= max(10.0, _ps + 6.0):
                        break
                    try:
                        _pl, _lam = _ext_plr(prob_info, alpha=_aeff, timelimit=10.0 ** 9, iters=100, price=_u if _u else None, prox=_e if _e else None, prox_w=0.2 * w1, lam0=_lam)
                        _ra, _ = construct(prob_info, timelimit=60.0, plan=_pl, plan_bays=False, plan_target_w=0.3 * w1, plan_bucket=int(os.environ.get('OGC_PLAN_BUCKET', '0')), geo_cache=geo_cache, pair_memo=pair_memo, rel_memo=rel_memo, verbose=False)
                    except Exception:
                        break
                    finally:
                        pass
                    candidates.append(_ra)
                    _menu_pool.append(_ra)
                    _ucap = 0.5 * w1
                    for i, p in _pl.items():
                        if i in _ra:
                            v = _u.get(i, 0.0) + 0.03 * w1 * (p['start'] - _ra[i]['entry_time'])
                            _u[i] = max(-_ucap, min(_ucap, v))
                    _e = {i: a['entry_time'] for i, a in _ra.items()}
                    _o = obj_of(_ra)
                    if _o < _ext_low - 1e-09:
                        _ext_low = _o
                        _miss = 0
                    else:
                        _miss += 1
                _menu_led = bool(_menu_pool) and obj_of(min(_menu_pool, key=obj_of)) < obj_of(_ext_base) - 1e-09
                if remaining() > 20.0 and _menu_led:
                    from pull import pull_forward
                    try:
                        _pa, _ = pull_forward(prob_info, _trajectory_best(), timelimit=min(30.0, remaining() - 8.0), pref=True, try_cost=_pt, geo_cache=geo_cache, pair_memo=pair_memo, rel_memo=rel_memo, verbose=False)
                        candidates.append(_pa)
                        _menu_pool.append(_pa)
                    except Exception as _pe:
                        pass
                    finally:
                        pass
            if not _menu_led:
                _ext_w = remaining() - 4.0
                _mtl = 0.45 * _ext_w
                if not _both and (not _factory_role) and (_mtl > max(4.0, 1.05 * _est_pass)):
                    try:
                        _mx, _ = construct_multi(prob_info, timelimit=_mtl, use_legacy=False, geo_cache=geo_cache, pair_memo=pair_memo, rel_memo=rel_memo, var_offset=_arm_seed * 17 + det_passes + 4, verbose=False)
                        if _mx is not None:
                            candidates.append(_mx)
                    except Exception:
                        pass
                    finally:
                        pass
                _ltl = remaining() - 4.0
                if _ltl > 8.0:
                    try:
                        _seed = _trajectory_best()
                        _ext_walk_prob = _factory_locked_prob if _factory_deep and _factory_locked_prob is not None else prob_info
                        _lx, _ = _lns_walk(_ext_walk_prob, _seed, timelimit=_ltl, seed=_arm_seed + 101, geo_cache=geo_cache, pair_memo=pair_memo, rel_memo=rel_memo, verbose=False)
                        candidates.append(_lx)
                    except Exception:
                        pass
                    finally:
                        pass
            else:
                _ltl = 0.6 * (remaining() - 4.0)
                if _ltl > 8.0:
                    try:
                        _seed = min(_pre_pool, key=obj_of)
                        _lx, _ = _lns_walk(prob_info, _seed, timelimit=_ltl, seed=_arm_seed + 101, geo_cache=geo_cache, pair_memo=pair_memo, rel_memo=rel_memo, verbose=False)
                        candidates.append(_lx)
                    except Exception:
                        pass
                    finally:
                        pass
                _ltl = remaining() - 4.0
                if _ltl > 8.0:
                    try:
                        _seed = _trajectory_best()
                        _lx, _ = _lns_walk(prob_info, _seed, timelimit=_ltl, seed=_arm_seed + 101, geo_cache=geo_cache, pair_memo=pair_memo, rel_memo=rel_memo, verbose=False)
                        candidates.append(_lx)
                    except Exception:
                        pass
                    finally:
                        pass
        _stage('ext')
    if os.environ.get('OGC_ARM', '') == '' and _slr and (not _both) and _plan_last and (remaining() > 24.0 or False):
        try:
            from winrep import window_stage, base_lateness
            _whb = _trajectory_best()
            _whrows = sorted(((_whb[_i]['entry_time'] - _wp['start'], _i, _wp['bay_id']) for _i, _wp in _plan_last.items() if _i in _whb and _whb[_i]['entry_time'] > _wp['start']), reverse=True)
            _whrows = [[_i, _j] for _, _i, _j in _whrows[:8]]
            if _whrows:
                _wh_t0 = time.time()
                _wh_b0 = remaining() - 8.0
                _wh = window_stage(prob_info, _whb, obj_of(_whb), base_lateness(prob_info, _whb), budget_fn=lambda: min(_wh_b0 - (time.time() - _wh_t0), remaining() - 6.0), chart=False, seed_rows=_whrows, debug=False)
                if _wh is not None:
                    from gpr_core import assignments_from_solution
                    candidates.append(assignments_from_solution(_wh[0]))
        except Exception as _pe:
            pass
        finally:
            pass
    _stage('tail')
    ranked = sorted(candidates, key=obj_of)
    hard_deadline = t_start + timelimit - 2.0
    last_sol = None

    def _emit_with_meta(assign, sol, verified):
        if run_meta is not None:
            winning_stage = candidate_sources.get(id(assign), 'unknown')
            run_meta.update({'winning_stage': winning_stage, 'winning_internal_objective': float(obj_of(assign)), 'emit_checker_verified': bool(verified)})
            if _slr and _plan_last and (os.environ.get('OGC_ARM', '') != ''):
                run_meta['_parent_plan_targets'] = {int(_bid): {'bay_id': int(_target.get('bay_id', -1)), 'start': int(_target['start'])} for _bid, _target in _plan_last.items() if 'start' in _target}
        return sol
    for assign in ranked:
        try:
            sol, _ = retime(prob_info, assign, verbose=False)
        except Exception:
            continue
        finally:
            pass
        last_sol = sol if last_sol is None else last_sol
        if time.time() + max(3.0, n_blocks * 0.03) < hard_deadline:
            try:
                res = check_feasibility(prob_info, sol)
                if res['feasible']:
                    return _emit_with_meta(assign, sol, True)
                continue
            except Exception:
                return _emit_with_meta(assign, sol, False)
            finally:
                pass
        return _emit_with_meta(assign, sol, False)
    if run_meta is not None:
        run_meta.update({'winning_stage': None, 'winning_internal_objective': None, 'emit_checker_verified': False})
    return last_sol
