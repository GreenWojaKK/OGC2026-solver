import math
import time
from collections import defaultdict
from ortools.sat.python import cp_model
from utils import _poly_from_verts

def _layer0_area(blk):
    p = _poly_from_verts(blk['shape'][0]['layers'][0])
    return float(p.area) if p is not None else 0.0

def demand_ratio(prob) -> float:
    blocks = prob['blocks']
    bay_area = sum((b['width'] * b['height'] for b in prob['bays']))
    events = []
    for b in blocks:
        lo = b['due_date'] - b['processing_time']
        hi = b['release_time'] + b['processing_time']
        if lo < hi:
            a = _layer0_area(b)
            events.append((lo, a))
            events.append((hi, -a))
    if not events:
        return 0.0
    events.sort()
    cur = best = 0.0
    for _, d in events:
        cur += d
        if cur > best:
            best = cur
    return best / bay_area

def _fits_bay(blk, bay) -> bool:
    W, H = (bay['width'], bay['height'])
    for orient in blk['shape']:
        vs = [v for l in orient['layers'] for v in l]
        xs = [v[0] for v in vs]
        ys = [v[1] for v in vs]
        if math.ceil(-min(xs)) <= math.floor(W - max(xs)) and math.ceil(-min(ys)) <= math.floor(H - max(ys)):
            return True
    return False

def greedy_plan(prob, alpha: float=0.72):
    blocks, bays = (prob['blocks'], prob['bays'])
    nb = len(bays)
    a100 = [int(round(_layer0_area(b) * 100)) for b in blocks]
    cap = [int(alpha * b['width'] * b['height'] * 100) for b in bays]
    used = [defaultdict(int) for _ in range(nb)]
    order = sorted(range(len(blocks)), key=lambda i: (blocks[i]['due_date'], -a100[i]))
    plan = {}
    for i in order:
        b = blocks[i]
        R, P = (b['release_time'], b['processing_time'])
        smax = max(b['bay_preferences'])
        best = None
        for j in range(nb):
            if not _fits_bay(b, bays[j]) or a100[i] > cap[j]:
                continue
            t = R
            while True:
                ok = True
                for d in range(t, t + P):
                    if used[j][d] + a100[i] > cap[j]:
                        t = d + 1
                        ok = False
                        break
                if ok:
                    break
            key = (t, smax - b['bay_preferences'][j])
            if best is None or key < best[0]:
                best = (key, j, t)
        if best is None:
            j = max(range(nb), key=lambda jj: cap[jj])
            plan[i] = (j, R)
            continue
        _, j, t = best
        for d in range(t, t + P):
            used[j][d] += a100[i]
        plan[i] = (j, t)
    return plan

def solve_seat_plan(prob, timelimit: float=5.0, alpha: float=0.8, alpha_smooth: float=0.65, incumbent: dict | None=None, workers: int=4):
    t0 = time.time()
    blocks, bays = (prob['blocks'], prob['bays'])
    w2 = int(prob['weights']['w2'])
    w3 = int(prob['weights']['w3'])
    n, m = (len(blocks), len(bays))
    areas = [b['width'] * b['height'] for b in bays]
    avg_area = sum(areas) / m
    uu = [int(round(avg_area / a * 10000)) for a in areas]
    model = cp_model.CpModel()
    x = {}
    for i, blk in enumerate(blocks):
        fits = [j for j in range(m) if _fits_bay(blk, bays[j])]
        for j in fits:
            x[i, j] = model.NewBoolVar(f'x{i}_{j}')
        model.AddExactlyOne((x[i, j] for j in fits))
    pref_terms = []
    for i, blk in enumerate(blocks):
        prefs = blk['bay_preferences']
        smax = max(prefs)
        for j in range(m):
            if (i, j) in x and smax - prefs[j] > 0:
                pref_terms.append((smax - prefs[j]) * x[i, j])
    scaled_load = []
    for j in range(m):
        scaled_load.append(sum((uu[j] * blk['workload'] * x[i, j] for i, blk in enumerate(blocks) if (i, j) in x)))
    z_ub = max(uu) * sum((blk['workload'] for blk in blocks))
    z = model.NewIntVar(0, z_ub, 'z')
    for a in range(m):
        for b in range(m):
            if a != b:
                model.Add(z >= scaled_load[a] - scaled_load[b])
    a100 = [int(round(_layer0_area(blk) * 100)) for blk in blocks]
    cores = [(i, blk['due_date'] - blk['processing_time'], blk['release_time'] + blk['processing_time']) for i, blk in enumerate(blocks) if blk['due_date'] - blk['processing_time'] < blk['release_time'] + blk['processing_time']]
    for t in sorted({lo for _, lo, _ in cores}):
        live = [i for i, lo, hi in cores if lo <= t < hi]
        for j in range(m):
            cap = int(alpha * areas[j] * 100)
            terms = [a100[i] * x[i, j] for i in live if (i, j) in x]
            if terms:
                model.Add(sum(terms) <= cap)
    spans = [(i, blk['release_time'], max(blk['due_date'], blk['release_time'] + blk['processing_time'])) for i, blk in enumerate(blocks)]
    d100 = {i: int(round(a100[i] * blocks[i]['processing_time'] / (hi - lo))) for i, lo, hi in spans}
    for t in sorted({lo for _, lo, _ in spans}):
        live = [i for i, lo, hi in spans if lo <= t < hi]
        for j in range(m):
            cap = int(alpha_smooth * areas[j] * 100)
            terms = [d100[i] * x[i, j] for i in live if (i, j) in x]
            if terms:
                model.Add(sum(terms) <= cap)
    model.Minimize(w2 * z + w3 * 10000 * sum(pref_terms))
    if incumbent:
        for i, a in incumbent.items():
            if (i, a['bay_id']) in x:
                model.AddHint(x[i, a['bay_id']], 1)
    solver = cp_model.CpSolver()
    solver.parameters.max_deterministic_time = max(1.0, timelimit)
    solver.parameters.max_time_in_seconds = 10.0 * max(1.0, timelimit)
    solver.parameters.random_seed = 0
    solver.parameters.num_search_workers = 1
    status = solver.Solve(model)
    target = {}
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for (i, j), v in x.items():
            if solver.Value(v):
                target[i] = j
    return (target, {'status': solver.StatusName(status), 'obj23': solver.ObjectiveValue() / 10000 if target else None, 'wall_s': round(time.time() - t0, 1)})
