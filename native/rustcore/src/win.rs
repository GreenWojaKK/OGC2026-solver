// win.rs -- seatrep window enumeration (candidate generation + relation
// classification), the cost-dominant phase of the windowed exact
// re-assignment (TECHNICAL_REPORT 10.32).  The selection semantics
// mirror dev/native/seatrep_core.py's canonical (ordered) enumeration
// bit for bit: same linspace-round-half-even grids, same (blocked,
// n_excl, gy, gx) stable sort, same per-group quotas, same conflict and
// pair orders.  Relations come from Engine::relation -- the same
// pair_relation walk the insert loop uses (stage-E parity lineage).

use crate::engine::Engine;

const KIND_FREE: u8 = 3;
const KIND_EXCL: u8 = 0;

pub struct WinRes {
    // per candidate row: [blk_slot, bay, x, y, oi, is_cur, conf_off, conf_len]
    pub cands: Vec<[i64; 8]>,
    // per conflict row: [s, f, kind]  kind 0=EXCL 1=B_UNDER 2=B_OVER
    pub confs: Vec<[i64; 3]>,
    // per pair row: [slot_i, ci, slot_j, cj, kind] kind 0=EXCL 1=U12 2=U21
    pub pairs: Vec<[i64; 5]>,
}

// np.linspace(lo, hi, n): step = (hi-lo)/(n-1), v_i = i*step + lo,
// v_last = hi exactly; python int(round(v)) = round-half-even; then
// sorted(set(..)) ascending.
fn grid(lo: i64, hi: i64, n: usize) -> Vec<i64> {
    if hi <= lo {
        return vec![lo];
    }
    let (lof, hif) = (lo as f64, hi as f64);
    let step = (hif - lof) / (n - 1) as f64;
    let mut vals: Vec<i64> = Vec::with_capacity(n);
    for i in 0..n {
        let v = if i == n - 1 { hif } else { i as f64 * step + lof };
        vals.push(v.round_ties_even() as i64);
    }
    vals.sort_unstable();
    vals.dedup();
    vals
}

struct Scored {
    key: [i64; 4], // blocked, n_excl, gy, gx
    bay: i32,
    x: i64,
    y: i64,
    oi: i32,
    confs: Vec<[i64; 3]>,
}

/// Stable top-k indices for the window ordering key.  Window enumeration can
/// create thousands of rows but the CP-SAT model consumes at most `cand_m-1`
/// exploratory rows (normally 11).  Keeping only that ordered prefix avoids a
/// full O(n log n) sort while preserving enumeration order on exact ties.
fn topk_scored<F>(rows: &[Scored], k: usize, mut keep: F) -> Vec<usize>
where
    F: FnMut(&Scored) -> bool,
{
    let mut best: Vec<usize> = Vec::with_capacity(k.min(rows.len()));
    for (i, row) in rows.iter().enumerate() {
        if !keep(row) {
            continue;
        }
        let mut lo = 0usize;
        let mut hi = best.len();
        while lo < hi {
            let mid = (lo + hi) / 2;
            let j = best[mid];
            if rows[j].key < row.key || (rows[j].key == row.key && j < i) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        if lo < k {
            best.insert(lo, i);
            if best.len() > k {
                best.pop();
            }
        }
    }
    best
}

#[allow(clippy::too_many_arguments)]
pub fn win_enum(
    e: &mut Engine,
    // frozen rows, caller-canonical order (sorted block id)
    fz_gk: &[usize], fz_x: &[i64], fz_y: &[i64], fz_bay: &[i32],
    fz_s: &[i64], fz_f: &[i64],
    // window rows
    wn_bid: &[i32], wn_r: &[i64], _wn_p: &[i64], wn_d: &[i64],
    wn_is_off: &[u8], wn_cur_bay: &[i32], wn_cur_x: &[i64],
    wn_cur_y: &[i64], wn_cur_oi: &[i32],
    wn_allowed_ptr: &[i32], wn_allowed_flat: &[i32],
    grid_nx: usize, grid_ny: usize, cand_m: usize, force_off: bool,
) -> WinRes {
    let m = e.bay_w.len();
    // per-bay frozen index lists, input order preserved
    let mut by_bay: Vec<Vec<usize>> = vec![Vec::new(); m];
    for (fi, &b) in fz_bay.iter().enumerate() {
        by_bay[b as usize].push(fi);
    }

    let mut res = WinRes { cands: Vec::new(), confs: Vec::new(),
                           pairs: Vec::new() };
    // final candidate geometry per slot for the pair pass:
    // (bay, gk, x, y)
    let mut final_geo: Vec<Vec<(i32, usize, i64, i64)>> =
        Vec::with_capacity(wn_bid.len());

    for slot in 0..wn_bid.len() {
        let bid = wn_bid[slot];
        let (b_r, b_d) = (wn_r[slot], wn_d[slot]);
        let mut scored: Vec<Scored> = Vec::new();
        let a0 = wn_allowed_ptr[slot] as usize;
        let a1 = wn_allowed_ptr[slot + 1] as usize;
        for &bay in &wn_allowed_flat[a0..a1] {
            let bw = e.bay_w[bay as usize];
            let bh = e.bay_h[bay as usize];
            let mut oi = 0i32;
            while let Some(&gk) = e.gidx.get(&(bid, oi)) {
                let bb = e.geos[gk].bbox;
                let xlo = (-bb[0]).ceil() as i64;
                let xhi = (bw as f64 - bb[2]).floor() as i64;
                let ylo = (-bb[1]).ceil() as i64;
                let yhi = (bh as f64 - bb[3]).floor() as i64;
                if xhi >= xlo && yhi >= ylo {
                    for &gx in &grid(xlo, xhi, grid_nx) {
                        for &gy in &grid(ylo, yhi, grid_ny) {
                            let (confs, blocked, n_excl) = classify(
                                e, &by_bay[bay as usize], fz_gk, fz_x,
                                fz_y, fz_s, fz_f, gk, gx, gy, b_r, b_d);
                            scored.push(Scored {
                                key: [blocked, n_excl, gy, gx],
                                bay, x: gx, y: gy, oi, confs });
                        }
                    }
                }
                oi += 1;
            }
        }
        let is_off = wn_is_off[slot] != 0;
        let explore_n = cand_m.saturating_sub(1);
        let only_current_bay = a1 == a0 + 1
            && wn_allowed_flat[a0] == wn_cur_bay[slot];
        let selected_idx = if is_off || only_current_bay {
            // A same-bay mover is not forced away from its incumbent, but all
            // exploratory slots should still be spent inside its requested
            // bay rather than applying the resident own/alternative quota.
            topk_scored(&scored, explore_n, |_| true)
        } else {
            // own-bay quota then alternatives fill (seatrep_core)
            let own_q = if explore_n == 0 {
                0
            } else {
                std::cmp::max(1, explore_n * 2 / 3)
            };
            let cur_bay = wn_cur_bay[slot];
            let mut own = topk_scored(&scored, own_q,
                                      |s| s.bay == cur_bay);
            let alt_q = explore_n - own.len();
            own.extend(topk_scored(&scored, alt_q,
                                   |s| s.bay != cur_bay));
            own
        };
        // Move only the retained rows (and their conflict vectors) into the
        // model order; no cloning of the discarded relation payloads.
        let mut slots: Vec<Option<Scored>> =
            scored.into_iter().map(Some).collect();
        let scored: Vec<Scored> = selected_idx.into_iter()
            .map(|i| slots[i].take().unwrap()).collect();

        let mut slot_geo: Vec<(i32, usize, i64, i64)> = Vec::new();
        let include_cur = !(force_off && is_off);
        let (cb, cx, cy, coi) = (wn_cur_bay[slot], wn_cur_x[slot],
                                 wn_cur_y[slot], wn_cur_oi[slot]);
        let mut current_seen = false;
        for s in &scored {
            let gk = e.gidx[&(bid, s.oi)];
            let is_cur = include_cur && s.bay == cb && s.x == cx
                && s.y == cy && s.oi == coi;
            current_seen |= is_cur;
            push_cand(&mut res, slot, s.bay, s.x, s.y, s.oi,
                      if is_cur { 1 } else { 0 }, &s.confs);
            slot_geo.push((s.bay, gk, s.x, s.y));
        }
        if include_cur && !current_seen {
            let gk = e.gidx[&(bid, coi)];
            let (confs, _, _) = classify(
                e, &by_bay[cb as usize], fz_gk, fz_x, fz_y, fz_s, fz_f,
                gk, cx, cy, b_r, b_d);
            push_cand(&mut res, slot, cb, cx, cy, coi, 1, &confs);
            slot_geo.push((cb, gk, cx, cy));
        }
        final_geo.push(slot_geo);
    }

    // candidate-vs-candidate pairs, same-bay only, seatrep_core order
    for i in 0..final_geo.len() {
        for j in (i + 1)..final_geo.len() {
            for (ci, &(bay1, gk1, x1, y1)) in final_geo[i].iter()
                                                          .enumerate() {
                for (cj, &(bay2, gk2, x2, y2)) in final_geo[j].iter()
                                                              .enumerate() {
                    if bay1 != bay2 {
                        continue;
                    }
                    let (kind, under_is_b) =
                        e.relation(gk1, x1, y1, gk2, x2, y2);
                    if kind == KIND_FREE {
                        continue;
                    }
                    let k = if kind == KIND_EXCL {
                        0
                    } else if under_is_b {
                        2 // second block under -> U21
                    } else {
                        1 // first block under -> U12
                    };
                    res.pairs.push([i as i64, ci as i64, j as i64,
                                    cj as i64, k]);
                }
            }
        }
    }
    res
}

fn classify(
    e: &mut Engine, bay_frozen: &[usize], fz_gk: &[usize], fz_x: &[i64],
    fz_y: &[i64], fz_s: &[i64], fz_f: &[i64], gk: usize, x: i64, y: i64,
    b_r: i64, b_d: i64,
) -> (Vec<[i64; 3]>, i64, i64) {
    let mut confs: Vec<[i64; 3]> = Vec::new();
    let mut blocked = 0i64;
    let mut n_excl = 0i64;
    for &fi in bay_frozen {
        // relation(placed=frozen, cand=window block) -- under_is_b true
        // means the window block is the under side (B_UNDER)
        let (kind, under_is_b) = e.relation(fz_gk[fi], fz_x[fi],
                                            fz_y[fi], gk, x, y);
        if kind == KIND_FREE {
            continue;
        }
        let k = if kind == KIND_EXCL {
            n_excl += 1;
            let ov = std::cmp::min(fz_f[fi], b_d)
                - std::cmp::max(fz_s[fi], b_r);
            if ov > 0 {
                blocked += ov;
            }
            0
        } else if under_is_b {
            1
        } else {
            2
        };
        confs.push([fz_s[fi], fz_f[fi], k]);
    }
    (confs, blocked, n_excl)
}

fn push_cand(res: &mut WinRes, slot: usize, bay: i32, x: i64, y: i64,
             oi: i32, is_cur: i64, confs: &[[i64; 3]]) {
    let off = res.confs.len() as i64;
    res.confs.extend_from_slice(confs);
    res.cands.push([slot as i64, bay as i64, x, y, oi as i64, is_cur,
                    off, confs.len() as i64]);
}
