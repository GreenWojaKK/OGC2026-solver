from __future__ import annotations
import ctypes
import time
from collections import defaultdict, deque
import numpy as np
from utils import Block, _poly_from_verts, _bb_overlap, _bounding_box
FREE = 'FREE'
EXCLUSIVE = 'EXCLUSIVE'
_NATIVE = None
_NAT_H = None
_NAT_REG: dict = {}
_NAT_N = [0]
_NAT_CNT = None
_NAT_CNT_P = None
_REL_VERIFY = False
_REL_FULL = True
_REL_VER_CNT = [0, 0]
_PD = ctypes.POINTER(ctypes.c_double)
_PI32 = ctypes.POINTER(ctypes.c_int32)
_REL_MEMO: dict = {}
_REL_REFS: dict = {}

def _set_native(lib):
    global _NATIVE, _NAT_H, _NAT_CNT, _NAT_CNT_P, _REL_VERIFY, _REL_FULL
    lib.geo_pair_ov.restype = ctypes.c_int32
    lib.geo_pair_ov.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_int64, ctypes.c_int64, ctypes.c_int32, ctypes.c_int64, ctypes.c_int64, ctypes.POINTER(ctypes.c_int64)]
    _NATIVE = lib
    _NAT_H = lib.geo_new()
    _NAT_CNT = np.zeros(2, dtype=np.int64)
    _NAT_CNT_P = _NAT_CNT.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
    _REL_VERIFY = False
    _REL_FULL = False

def _nat_gids_of(blk: Block):
    global _NAT_H
    _sh = blk.block_data['shape']
    ent = _NAT_REG.get(id(_sh))
    if ent is None:
        ent = (_sh, {})
        _NAT_REG[id(_sh)] = ent
    gids = ent[1].get(blk.orient_idx)
    if gids is None:
        if _NAT_N[0] >= 15000:
            _NAT_H = _NATIVE.geo_new()
            _NAT_REG.clear()
            _NAT_N[0] = 0
            ent = (_sh, {})
            _NAT_REG[id(_sh)] = ent
        layers = blk.resolved_layers()
        if not layers or not layers[0]:
            gids = ()
        else:
            rx, ry = layers[0][0]
            if rx != 0.0 or ry != 0.0:
                gids = ()
            else:
                out = []
                for verts in layers:
                    if len(verts) < 3:
                        out.append(-1)
                        continue
                    xy = np.ascontiguousarray(np.asarray(verts, dtype=np.float64))
                    if not np.array_equal(xy[0], xy[-1]):
                        xy = np.vstack((xy, xy[:1]))
                    n = len(xy)
                    rp = np.array([0, n], dtype=np.int32)
                    tv = np.array([0], dtype=np.int32)
                    out.append(_NATIVE.geo_add(_NAT_H, xy.ctypes.data_as(_PD), n, rp.ctypes.data_as(_PI32), 1, tv.ctypes.data_as(_PI32), 1))
                    _NAT_N[0] += 1
                gids = tuple(out)
        ent[1][blk.orient_idx] = gids
    return gids if gids else None

def _interior_overlap(pa, pb) -> bool:
    if pa is None or pb is None:
        return False
    try:
        inter = pa.intersection(pb)
    except Exception:
        return False
    return not inter.is_empty and inter.area > 0

def pair_relation(A: Block, B: Block) -> tuple:
    bdA = A.block_data
    bdB = B.block_data
    key = (id(bdA), A.orient_idx, id(bdB), B.orient_idx, A.x, A.y, B.x, B.y)
    r = _REL_MEMO.get(key)
    if r is None:
        r = _pair_relation_compute(A, B)
        if len(_REL_MEMO) >= 262144:
            _REL_MEMO.clear()
            _REL_REFS.clear()
        _REL_MEMO[key] = r
        _REL_REFS[id(bdA)] = bdA
        _REL_REFS[id(bdB)] = bdB
    return r

def _pair_relation_compute(A: Block, B: Block) -> tuple:
    if not _bb_overlap(A.bounding_rect(), B.bounding_rect()):
        return ('FREE',)
    if _NAT_H is not None:
        _h0 = _NAT_H
        ga = _nat_gids_of(A)
        gb = _nat_gids_of(B) if ga is not None else None
        if gb is not None and _NAT_H is not _h0:
            ga = _nat_gids_of(A)
        if gb is not None and ga is not None:
            return _pair_relation_nat(A, B, ga, gb)
    la, lb = (A.layers_at_pos(), B.layers_at_pos())
    pa = [_poly_from_verts(l) for l in la]
    pb = [_poly_from_verts(l) for l in lb]
    ba = [_bounding_box(l) for l in la]
    bb = [_bounding_box(l) for l in lb]
    blocked_A = blocked_B = False
    for a in range(len(la)):
        for b in range(len(lb)):
            if not (a >= b and (not blocked_B)) and (not (b >= a and (not blocked_A))):
                continue
            if not _bb_overlap(ba[a], bb[b]):
                continue
            if _interior_overlap(pa[a], pb[b]):
                if a >= b:
                    blocked_B = True
                if b >= a:
                    blocked_A = True
                if blocked_A and blocked_B:
                    return ('EXCLUSIVE',)
    if not blocked_A and (not blocked_B):
        return ('FREE',)
    if blocked_A and blocked_B:
        return ('EXCLUSIVE',)
    if blocked_B:
        return ('UNDER', B.block_id, A.block_id)
    return ('UNDER', A.block_id, B.block_id)

def _pair_relation_nat(A: Block, B: Block, ga: tuple, gb: tuple) -> tuple:
    la, lb = (A.layers_at_pos(), B.layers_at_pos())
    ax, ay, bx, by = (A.x, A.y, B.x, B.y)
    pa: list = [None] * len(la)
    pb: list = [None] * len(lb)
    blocked_A = blocked_B = False
    for a in range(len(la)):
        gaa = ga[a]
        for b in range(len(lb)):
            if not (a >= b and (not blocked_B)) and (not (b >= a and (not blocked_A))):
                continue
            hit = None
            if gaa >= 0 and gb[b] >= 0:
                if _REL_FULL:
                    hit = bool(_NATIVE.geo_pair_ov(_NAT_H, gaa, ax, ay, gb[b], bx, by, _NAT_CNT_P))
                elif not _NATIVE.geo_pair(_NAT_H, gaa, ax, ay, gb[b], bx, by, _NAT_CNT_P):
                    hit = False
            if _REL_VERIFY and hit is not None:
                v = pa[a]
                if v is None:
                    v = pa[a] = _poly_from_verts(la[a])
                w = pb[b]
                if w is None:
                    w = pb[b] = _poly_from_verts(lb[b])
                ref = _interior_overlap(v, w)
                _REL_VER_CNT[0] += 1
                if ref != hit:
                    _REL_VER_CNT[1] += 1
                    print(f'[RELVER] mismatch blocks {A.block_id}/{B.block_id} layers {a}/{b} nat={hit} geos={ref}', flush=True)
                hit = ref
            if hit is None:
                v = pa[a]
                if v is None:
                    v = pa[a] = _poly_from_verts(la[a])
                w = pb[b]
                if w is None:
                    w = pb[b] = _poly_from_verts(lb[b])
                hit = _interior_overlap(v, w)
            if hit:
                if a >= b:
                    blocked_B = True
                if b >= a:
                    blocked_A = True
                if blocked_A and blocked_B:
                    return ('EXCLUSIVE',)
    if not blocked_A and (not blocked_B):
        return ('FREE',)
    if blocked_A and blocked_B:
        return ('EXCLUSIVE',)
    if blocked_B:
        return ('UNDER', B.block_id, A.block_id)
    return ('UNDER', A.block_id, B.block_id)

def compute_relations(prob_info: dict, assignments: dict) -> dict:
    blocks_data = prob_info['blocks']
    by_bay: dict[int, list[int]] = defaultdict(list)
    for bid, a in assignments.items():
        by_bay[a['bay_id']].append(bid)
    placed: dict[int, Block] = {bid: Block(block_id=bid, block_data=blocks_data[bid], x=int(a['x']), y=int(a['y']), orient_idx=a['orient_idx']) for bid, a in assignments.items()}
    bboxes = {bid: block.bounding_rect() for bid, block in placed.items()}
    relations: dict[tuple, tuple] = {}
    for ids in by_bay.values():
        sweep = sorted(ids, key=lambda bid: (bboxes[bid][0], bid))
        active: list[int] = []
        candidates: list[tuple[int, int]] = []
        for j in sweep:
            bj = bboxes[j]
            active = [i for i in active if bboxes[i][2] > bj[0]]
            for i in active:
                bi = bboxes[i]
                if bi[1] < bj[3] and bj[1] < bi[3]:
                    candidates.append((i, j) if i < j else (j, i))
            active.append(j)
        for i, j in sorted(candidates):
            rel = pair_relation(placed[i], placed[j])
            if rel[0] != 'FREE':
                relations[i, j] = rel
    return relations

def derive_choice_edges(relations: dict, assignments: dict):
    edges: list[tuple] = []
    containments: list[tuple] = []
    violations: list[str] = []
    for (i, j), rel in relations.items():
        Si, Fi = (assignments[i]['entry_time'], assignments[i]['exit_time'])
        Sj, Fj = (assignments[j]['entry_time'], assignments[j]['exit_time'])
        if rel[0] == 'EXCLUSIVE':
            if Fi <= Sj:
                edges.append(('FS', i, j))
            elif Fj <= Si:
                edges.append(('FS', j, i))
            else:
                violations.append(f'EXCLUSIVE pair ({i},{j}) has overlapping times [{Si},{Fi}) vs [{Sj},{Fj}) in reference schedule')
        else:
            u, o = (rel[1], rel[2])
            Su, Fu = (assignments[u]['entry_time'], assignments[u]['exit_time'])
            So, Fo = (assignments[o]['entry_time'], assignments[o]['exit_time'])
            if Fu <= So:
                edges.append(('FS', u, o))
            elif Fo <= Su:
                edges.append(('FS', o, u))
            elif Su <= So and Fo <= Fu:
                edges.append(('SS', u, o))
                edges.append(('FF', o, u))
                containments.append((u, o))
            else:
                violations.append(f'UNDER pair (under={u}, over={o}) is neither disjoint nor contained: [{Su},{Fu}) vs [{So},{Fo})')
    return (edges, containments, violations)

class PositiveCycleError(Exception):
    pass

class TemporalNetwork:

    def __init__(self, block_ids: list[int], release: dict, proc: dict):
        self.ids = list(block_ids)
        self.pos = {bid: k for k, bid in enumerate(self.ids)}
        self.n = len(self.ids)
        self.release = release
        self.proc = proc
        self.adj: list[list[tuple[int, int]]] = [[] for _ in range(2 * self.n)]
        for bid in self.ids:
            k = self.pos[bid]
            self.adj[k].append((self.n + k, proc[bid]))

    def S(self, bid):
        return self.pos[bid]

    def F(self, bid):
        return self.n + self.pos[bid]

    def add(self, kind: str, i: int, j: int, lag: int=0):
        if kind == 'FS':
            self.adj[self.F(i)].append((self.S(j), lag))
        elif kind == 'SS':
            self.adj[self.S(i)].append((self.S(j), lag))
        elif kind == 'FF':
            self.adj[self.F(i)].append((self.F(j), lag))
        else:
            raise ValueError(kind)

    def earliest(self) -> tuple[dict, dict]:
        N = 2 * self.n
        val = [0] * N
        for bid in self.ids:
            k = self.pos[bid]
            val[k] = int(self.release[bid])
            val[self.n + k] = int(self.release[bid]) + int(self.proc[bid])
        inq = [True] * N
        q = deque(range(N))
        pops = 0
        limit = 4 * N * max(1, sum((len(a) for a in self.adj)))
        while q:
            u = q.popleft()
            inq[u] = False
            pops += 1
            if pops > limit:
                raise PositiveCycleError('positive cycle: no feasible schedule')
            for v, w in self.adj[u]:
                nv = val[u] + w
                if nv > val[v]:
                    val[v] = nv
                    if not inq[v]:
                        inq[v] = True
                        q.append(v)
        S = {bid: val[self.pos[bid]] for bid in self.ids}
        F = {bid: val[self.n + self.pos[bid]] for bid in self.ids}
        return (S, F)

    def latest(self, due: dict) -> tuple[dict, dict]:
        N = 2 * self.n
        inf = 10 ** 30
        val = [inf] * N
        rev: list[list[tuple[int, int]]] = [[] for _ in range(N)]
        for u, out in enumerate(self.adj):
            for v, w in out:
                rev[v].append((u, w))
        q = deque()
        inq = [False] * N
        for bid in self.ids:
            v = self.F(bid)
            val[v] = int(due[bid])
            q.append(v)
            inq[v] = True
        pops = 0
        limit = 4 * N * max(1, sum((len(a) for a in self.adj)))
        while q:
            v = q.popleft()
            inq[v] = False
            pops += 1
            if pops > limit:
                raise PositiveCycleError('positive cycle: no feasible schedule')
            for u, w in rev[v]:
                nv = val[v] - w
                if nv < val[u]:
                    val[u] = nv
                    if not inq[u]:
                        inq[u] = True
                        q.append(u)
        bad = [bid for bid in self.ids if val[self.S(bid)] < int(self.release[bid])]
        if bad:
            raise ValueError(f'no zero-tardy ALAP schedule for fixed pair choices: release violated by block {bad[0]}')
        S = {bid: val[self.S(bid)] for bid in self.ids}
        F = {bid: val[self.F(bid)] for bid in self.ids}
        return (S, F)

def _topo_order(nodes: list[int], edges: list[tuple[int, int]]):
    indeg = {v: 0 for v in nodes}
    adj = defaultdict(list)
    eset = set()
    for a, b in edges:
        if a in indeg and b in indeg and ((a, b) not in eset):
            eset.add((a, b))
            adj[a].append(b)
            indeg[b] += 1
    q = deque(sorted((v for v in nodes if indeg[v] == 0)))
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    if len(order) < len(nodes):
        rem = [v for v in nodes if v not in set(order)]
        for a, b in eset:
            if a in rem and b in rem:
                return (None, (a, b))
        return (None, (rem[0], rem[0]))
    return (order, None)

def build_operations(assignments: dict, containments: list[tuple]):
    entry_prec = defaultdict(list)
    exit_prec = defaultdict(list)
    for u, o in containments:
        au, ao = (assignments[u], assignments[o])
        if au['entry_time'] == ao['entry_time']:
            entry_prec[au['entry_time'], au['bay_id']].append((u, o))
        if au['exit_time'] == ao['exit_time']:
            exit_prec[au['exit_time'], au['bay_id']].append((o, u))
    entries = defaultdict(list)
    exits = defaultdict(list)
    for bid, a in assignments.items():
        entries[a['entry_time'], a['bay_id']].append(bid)
        exits[a['exit_time'], a['bay_id']].append(bid)
    days = sorted({d for d, _ in entries} | {d for d, _ in exits})
    bays_of = lambda groups, day: sorted((b for d, b in groups if d == day))
    ops: dict[str, list] = {}
    for day in days:
        day_ops: list[dict] = []
        for bay in bays_of(exits, day):
            ids = exits[day, bay]
            order, cyc = _topo_order(sorted(ids), exit_prec.get((day, bay), []))
            if order is None:
                o, u = cyc
                return (None, ('exit', u, o))
            for bid in order:
                day_ops.append({'type': 'EXIT', 'block_id': bid, 'bay_id': bay})
        for bay in bays_of(entries, day):
            ids = entries[day, bay]
            order, cyc = _topo_order(sorted(ids), entry_prec.get((day, bay), []))
            if order is None:
                u, o = cyc
                return (None, ('entry', u, o))
            for bid in order:
                a = assignments[bid]
                day_ops.append({'type': 'ENTRY', 'block_id': bid, 'bay_id': bay, 'x': int(a['x']), 'y': int(a['y']), 'orient_idx': int(a['orient_idx'])})
        ops[str(int(day))] = day_ops
    return (ops, None)

def retime(prob_info: dict, assignments: dict, relations: dict | None=None, verbose: bool=True, timing: str='earliest', latest_exit_bounds: dict | None=None):
    t0 = time.time()
    blocks_data = prob_info['blocks']
    if timing not in ('earliest', 'latest'):
        raise ValueError(f'unknown retime timing mode: {timing}')
    if latest_exit_bounds is not None and timing != 'latest':
        raise ValueError("latest_exit_bounds requires timing='latest'")
    if relations is None:
        relations = compute_relations(prob_info, assignments)
    t_rel = time.time() - t0
    kinds = defaultdict(int)
    for rel in relations.values():
        kinds[rel[0]] += 1
    edges, containments, violations = derive_choice_edges(relations, assignments)
    if violations:
        raise ValueError('GPR model contradicted by reference schedule:\n  ' + '\n  '.join(violations[:10]))
    release = {bid: blocks_data[bid]['release_time'] for bid in assignments}
    proc = {bid: blocks_data[bid]['processing_time'] for bid in assignments}
    due = {bid: blocks_data[bid]['due_date'] for bid in assignments}
    if latest_exit_bounds is not None:
        missing = [bid for bid in assignments if bid not in latest_exit_bounds]
        if missing:
            raise ValueError(f'latest_exit_bounds missing block {missing[0]}')
        due = {bid: int(latest_exit_bounds[bid]) for bid in assignments}
    net = TemporalNetwork(sorted(assignments), release, proc)
    for kind, i, j in edges:
        net.add(kind, i, j, 0)
    new_assign = {bid: dict(a) for bid, a in assignments.items()}
    n_bumps = 0
    t1 = time.time()
    for _ in range(200):
        S, F = net.latest(due) if timing == 'latest' else net.earliest()
        for bid in new_assign:
            new_assign[bid]['entry_time'] = S[bid]
            new_assign[bid]['exit_time'] = F[bid]
        operations, bump = build_operations(new_assign, containments)
        if operations is not None:
            break
        n_bumps += 1
        kind, u, o = bump
        if kind == 'entry':
            net.add('SS', u, o, 1)
        else:
            net.add('FF', o, u, 1)
    else:
        raise RuntimeError('could not resolve same-day ordering cycles')
    t_lp = time.time() - t1
    info = {'relations': relations, 'n_pairs_exclusive': kinds['EXCLUSIVE'], 'n_pairs_under': kinds['UNDER'], 'n_containments_realised': len(containments), 'n_edges': len(edges), 'n_bumps': n_bumps, 'timing': timing, 'custom_latest_exit_bounds': bool(latest_exit_bounds is not None), 'assignments': new_assign, 't_relations_s': t_rel, 't_retime_s': t_lp}
    if verbose:
        print(f'[GPR] pairs: exclusive={kinds['EXCLUSIVE']}  under={kinds['UNDER']}  containment-realised={len(containments)}  edges={len(edges)}')
        print(f'[GPR] relations {t_rel:.2f}s  retime {t_lp:.3f}s  bumps={n_bumps}')
    return ({'operations': operations}, info)

def assignments_from_solution(solution: dict) -> dict:
    assign: dict[int, dict] = {}
    for t_str, ops in solution['operations'].items():
        t = int(t_str)
        for op in ops:
            bid = op['block_id']
            if op['type'] == 'ENTRY':
                assign.setdefault(bid, {}).update(block_id=bid, bay_id=op['bay_id'], x=int(op['x']), y=int(op['y']), orient_idx=int(op.get('orient_idx', 0)), entry_time=t)
            else:
                assign.setdefault(bid, {}).update(block_id=bid, exit_time=t)
                assign[bid].setdefault('bay_id', op['bay_id'])
    return assign
