// engine.rs -- stage E: the construct() insert loop (default kernel-path
// mode: full-order insertion, no plan / no partial), ported for
// final-assignment parity with constructor.construct.
//
// Everything python-specific is reproduced deliberately: stable sorts,
// numpy linspace + round-half-even subsampling, dict.fromkeys dedup,
// candidate-set construction via integer ceil/floor of the same float
// expressions, kernel CSR order, and the exact float evaluation order of
// every score expression.  Exact geometry verdicts come from the stage-B
// primitives (geom_intersects for the co-presence screen, per-layer
// geom_overlap for the crane relation walk -- the same verdict source as
// the production pair_relation full-native mode).

use crate::{geom_intersects, geom_overlap, Geom};
use std::collections::{BTreeSet, HashMap};

const KIND_PAST: u8 = 0; // EXCLUSIVE -> disjoint options only
const KIND_INSIDE: u8 = 1; // e under b: b may nest inside e's window
const KIND_ENCLOSE: u8 = 2; // b under e: b may enclose e's window
const MEMO_CAP: usize = 262_144;

struct LayerBits {
    ox: i64,
    oy: i64,
    mh: usize,
    mw: usize,
    stride: usize,
    bits: Vec<u64>,
}

pub(crate) struct EngGeo {
    pub union: Option<Geom>,
    pub layers: Vec<Option<Geom>>, // per layer; None when degenerate
    layer_bits: Vec<Option<LayerBits>>,
    pub bbox: [f64; 4],            // RAW layer-vertex bbox (constructor _Geo.bbox)
    pub ox: i64,
    pub oy: i64,
    pub mh: usize,
    pub mw: usize,
    pub bit_stride: usize,
    pub mask_bits: Vec<u64>,
    pub amask_bits: Vec<u64>,
}

struct Placed {
    x: i64,
    y: i64,
    s: i64,
    f: i64,
    gk: usize, // geo index
    bbox: [f64; 4], // world bbox = raw bbox + (x, y)
    bit_stride: usize,
    mask_bits: Vec<u64>,  // packed bay-grid world occupancy
    amask_bits: Vec<u64>,
}

fn pack_rows(mask: &[u8], h: usize, w: usize) -> (usize, Vec<u64>) {
    let stride = (w + 63) / 64;
    let mut bits = vec![0u64; h * stride];
    for r in 0..h {
        for c in 0..w {
            if mask[r * w + c] != 0 {
                bits[r * stride + c / 64] |= 1u64 << (c & 63);
            }
        }
    }
    (stride, bits)
}

fn blit_bits(src: &[u64], src_stride: usize, h: usize,
             dst_h: usize, dst_w: usize, r0: usize, c0: usize)
             -> (usize, Vec<u64>) {
    let dst_stride = (dst_w + 63) / 64;
    let mut dst = vec![0u64; dst_h * dst_stride];
    let shift = c0 & 63;
    let word0 = c0 / 64;
    for r in 0..h {
        let db = (r0 + r) * dst_stride + word0;
        let sb = r * src_stride;
        for sw in 0..src_stride {
            let v = src[sb + sw];
            if v == 0 {
                continue;
            }
            let di = db + sw;
            dst[di] |= v << shift;
            if shift != 0 && di + 1 < (r0 + r + 1) * dst_stride {
                dst[di + 1] |= v >> (64 - shift);
            }
        }
    }
    (dst_stride, dst)
}

fn blit_or_bits(src: &[u64], src_stride: usize, h: usize,
                dst: &mut [u64], dst_stride: usize,
                r0: usize, c0: usize) {
    let shift = c0 & 63;
    let word0 = c0 / 64;
    for r in 0..h {
        let db = (r0 + r) * dst_stride + word0;
        let sb = r * src_stride;
        for sw in 0..src_stride {
            let v = src[sb + sw];
            if v == 0 {
                continue;
            }
            let di = db + sw;
            dst[di] |= v << shift;
            if shift != 0 && di + 1 < (r0 + r + 1) * dst_stride {
                dst[di + 1] |= v >> (64 - shift);
            }
        }
    }
}

#[inline]
fn shifted_overlap(src: &[u64], src_stride: usize, h: usize,
                   r0: usize, c0: usize,
                   dst: &[u64], dst_stride: usize) -> bool {
    // Lossless representation change only: this is exactly the old
    // byte-cell nested loop, including cross-u64 shifts.  It is not a
    // local-coordinate or approximate collision verdict.
    let shift = c0 & 63;
    let word0 = c0 / 64;
    for r in 0..h {
        let db = (r0 + r) * dst_stride + word0;
        let sb = r * src_stride;
        for sw in 0..src_stride {
            let v = src[sb + sw];
            if v == 0 {
                continue;
            }
            let di = db + sw;
            if dst[di] & (v << shift) != 0 {
                return true;
            }
            if shift != 0 && di + 1 < (r0 + r + 1) * dst_stride
                && dst[di + 1] & (v >> (64 - shift)) != 0 {
                return true;
            }
        }
    }
    false
}

#[inline]
fn shifted_popcount(src: &[u64], src_stride: usize, h: usize,
                    r0: usize, c0: usize,
                    dst: &[u64], dst_stride: usize) -> u32 {
    let shift = c0 & 63;
    let word0 = c0 / 64;
    let mut count = 0u32;
    for r in 0..h {
        let db = (r0 + r) * dst_stride + word0;
        let sb = r * src_stride;
        for sw in 0..src_stride {
            let v = src[sb + sw];
            if v == 0 {
                continue;
            }
            let di = db + sw;
            count += (dst[di] & (v << shift)).count_ones();
            if shift != 0 && di + 1 < (r0 + r + 1) * dst_stride {
                count += (dst[di + 1] & (v >> (64 - shift))).count_ones();
            }
        }
    }
    count
}

fn raster_layer(g: &Geom, res: i64) -> LayerBits {
    let ox = (g.bb[0] * res as f64).floor() as i64;
    let oy = (g.bb[1] * res as f64).floor() as i64;
    let mw = ((g.bb[2] * res as f64).ceil() as i64 - ox).max(1) as usize;
    let mh = ((g.bb[3] * res as f64).ceil() as i64 - oy).max(1) as usize;
    let mut mask = vec![0u8; mh * mw];
    let cs = 1.0 / res as f64;
    for r in 0..mh {
        let y0 = (oy + r as i64) as f64 * cs;
        let y1 = y0 + cs;
        for c in 0..mw {
            let x0 = (ox + c as i64) as f64 * cs;
            let x1 = x0 + cs;
            let idx = r * mw + c;
            if g.bb[0] > x1 || g.bb[2] < x0 || g.bb[1] > y1
                || g.bb[3] < y0 {
                continue;
            }
            if crate::box_contact(g, x0, y0, x1, y1)
                || crate::point_in(g, 0.0, 0.0, x0, y0) {
                mask[idx] = 1;
            }
        }
    }
    let (stride, bits) = pack_rows(&mask, mh, mw);
    LayerBits { ox, oy, mh, mw, stride, bits }
}

pub struct Engine {
    pub res: i64,
    pub bay_w: Vec<i64>,
    pub bay_h: Vec<i64>,
    pub(crate) geos: Vec<EngGeo>,
    pub gidx: HashMap<(i32, i32), usize>, // (bid, oi) -> geos index
    pair_memo: HashMap<(usize, usize, i64, i64, i64, i64), bool>,
    rel_memo: HashMap<(usize, usize, i64, i64, i64, i64), (u8, bool)>,
    layer_lane_k: usize,
}

impl Engine {
    pub fn new(res: i64, bay_w: Vec<i64>, bay_h: Vec<i64>) -> Self {
        let layer_lane_enabled = std::env::var("OGC_RUST_LAYER_LANE")
            .map_or(false, |value| value != "0")
            && std::env::var("OGC_RUST_VERIFY")
                .map_or(true, |value| value != "1");
        let layer_lane_k = if layer_lane_enabled {
            std::env::var("OGC_RUST_LAYER_LANE_K").ok()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(4)
        } else {
            0
        };
        Engine {
            res,
            bay_w,
            bay_h,
            geos: Vec::new(),
            gidx: HashMap::new(),
            pair_memo: HashMap::new(),
            rel_memo: HashMap::new(),
            layer_lane_k,
        }
    }

    /// Register a (bid, oi) geometry: the shapely-constructed union rings
    /// plus the RAW layer bbox; packed masks are computed here with the
    /// stage-C cell predicates (bit-identical to _Geo).
    pub(crate) fn add_geo(&mut self, bid: i32, oi: i32,
                          union: Option<Geom>, bbox: [f64; 4]) -> usize {
        let (ox, oy, mw, mh) = if union.is_some() {
            let ox = (bbox[0] * self.res as f64).floor() as i64;
            let oy = (bbox[1] * self.res as f64).floor() as i64;
            let w = ((bbox[2] * self.res as f64).ceil() as i64 - ox).max(1);
            let h = ((bbox[3] * self.res as f64).ceil() as i64 - oy).max(1);
            (ox, oy, w as usize, h as usize)
        } else {
            (0, 0, 1, 1)
        };
        let mut mask = vec![0u8; mh * mw];
        let mut amask = vec![0u8; mh * mw];
        if let Some(u) = &union {
            let cs = 1.0 / self.res as f64;
            for r in 0..mh {
                let y0 = (oy + r as i64) as f64 * cs;
                let y1 = y0 + cs;
                for c in 0..mw {
                    let x0 = (ox + c as i64) as f64 * cs;
                    let x1 = x0 + cs;
                    let idx = r * mw + c;
                    if u.bb[0] > x1 || u.bb[2] < x0 || u.bb[1] > y1
                        || u.bb[3] < y0 {
                        continue;
                    }
                    if crate::box_contact(u, x0, y0, x1, y1) {
                        mask[idx] = 1;
                    } else if crate::point_in(u, 0.0, 0.0, x0, y0) {
                        mask[idx] = 1;
                        amask[idx] = 1;
                    }
                }
            }
        }
        let (bit_stride, mask_bits) = pack_rows(&mask, mh, mw);
        let (amask_stride, amask_bits) = pack_rows(&amask, mh, mw);
        debug_assert_eq!(bit_stride, amask_stride);
        let g = EngGeo {
            union,
            layers: Vec::new(),
            layer_bits: Vec::new(),
            bbox,
            ox,
            oy,
            mh,
            mw,
            bit_stride,
            mask_bits,
            amask_bits,
        };
        self.geos.push(g);
        let k = self.geos.len() - 1;
        self.gidx.insert((bid, oi), k);
        k
    }

    pub(crate) fn add_layer(&mut self, bid: i32, oi: i32,
                            g: Option<Geom>) {
        let k = self.gidx[&(bid, oi)];
        let bits = if self.layer_lane_k > 0 {
            g.as_ref().map(|layer| raster_layer(layer, self.res))
        } else {
            None
        };
        self.geos[k].layers.push(g);
        self.geos[k].layer_bits.push(bits);
    }

    fn pair_exact(&mut self, ek: usize, ex: i64, ey: i64,
                  e_mask: &[u64], e_amask: &[u64], e_stride: usize,
                  bay_w: usize, ck: usize, x: i64, y: i64) -> bool {
        // Conservative mask miss -> definitely disjoint; inner-mask hit ->
        // definitely intersecting; every residual case still reaches
        // geom_intersects.  Packed rows change none of those verdicts.
        let g = &self.geos[ck];
        let res = self.res;
        let r0 = y * res + g.oy;
        let c0 = x * res + g.ox;
        if r0 >= 0 && c0 >= 0 {
            let (r0, c0) = (r0 as usize, c0 as usize);
            let bay_h = e_mask.len() / e_stride;
            if r0 + g.mh <= bay_h && c0 + g.mw <= bay_w {
                if !shifted_overlap(
                    &g.mask_bits, g.bit_stride, g.mh, r0, c0,
                    e_mask, e_stride)
                {
                    return false;
                }
                if shifted_overlap(
                    &g.amask_bits, g.bit_stride, g.mh, r0, c0,
                    e_amask, e_stride)
                {
                    return true;
                }
            }
        }
        let key = (ek, ck, ex, ey, x, y);
        if let Some(&v) = self.pair_memo.get(&key) {
            return v;
        }
        let ge = self.geos[ek].union.as_ref().unwrap();
        let gc = self.geos[ck].union.as_ref().unwrap();
        let v = geom_intersects(ge, ex as f64, ey as f64, gc, x as f64,
                                y as f64);
        if self.pair_memo.len() >= MEMO_CAP {
            self.pair_memo.clear();
        }
        self.pair_memo.insert(key, v);
        v
    }

    /// Crane relation of placed (ek at ex, ey) vs candidate (ck at x, y):
    /// the pair_relation M[a][b] walk with geom_overlap cell verdicts.
    /// Returns (kind, under_is_cand) with kind 3 = FREE.
    pub(crate) fn relation(&mut self, ek: usize, ex: i64, ey: i64,
                           ck: usize, x: i64, y: i64) -> (u8, bool) {
        let key = (ek, ck, ex, ey, x, y);
        if let Some(&v) = self.rel_memo.get(&key) {
            return v;
        }
        let (la, lb) = {
            let a = &self.geos[ek];
            let b = &self.geos[ck];
            (a.layers.len(), b.layers.len())
        };
        let mut blocked_a = false;
        let mut blocked_b = false;
        'walk: for a in 0..la {
            for b in 0..lb {
                if !(a >= b && !blocked_b) && !(b >= a && !blocked_a) {
                    continue;
                }
                let ga = self.geos[ek].layers[a].as_ref();
                let gb = self.geos[ck].layers[b].as_ref();
                let (ga, gb) = match (ga, gb) {
                    (Some(x), Some(y)) => (x, y),
                    _ => continue, // degenerate layer: verdict False
                };
                if geom_overlap(ga, ex as f64, ey as f64, gb, x as f64,
                                y as f64) {
                    if a >= b {
                        blocked_b = true;
                    }
                    if b >= a {
                        blocked_a = true;
                    }
                    if blocked_a && blocked_b {
                        break 'walk;
                    }
                }
            }
        }
        let v = if !blocked_a && !blocked_b {
            (3u8, false)
        } else if blocked_a && blocked_b {
            (KIND_PAST, false)
        } else if blocked_b {
            // cand's crane path blocked: cand under the placed overhang
            (255, true)
        } else {
            (255, false)
        };
        if self.rel_memo.len() >= MEMO_CAP {
            self.rel_memo.clear();
        }
        self.rel_memo.insert(key, v);
        v
    }
}

struct LayerAggregate {
    stride: usize,
    plane: usize,
    prefix: Vec<u64>,
    suffix: Vec<u64>,
}

fn build_layer_aggregate(eng: &Engine, bay_placed: &[Placed],
                         act_idx: &[usize], bay_wpx: usize,
                         bay_hpx: usize, max_layers: usize)
                         -> LayerAggregate {
    let stride = (bay_wpx + 63) / 64;
    let plane = bay_hpx * stride;
    let mut layers = vec![0u64; max_layers * plane];
    for &ei in act_idx {
        let e = &bay_placed[ei];
        let g = &eng.geos[e.gk];
        for (layer, bits) in g.layer_bits.iter().enumerate() {
            let bits = match bits {
                Some(value) => value,
                None => continue,
            };
            let r0 = e.y * eng.res + bits.oy;
            let c0 = e.x * eng.res + bits.ox;
            debug_assert!(r0 >= 0 && c0 >= 0);
            debug_assert!(r0 as usize + bits.mh <= bay_hpx);
            debug_assert!(c0 as usize + bits.mw <= bay_wpx);
            let dst = &mut layers[layer * plane..(layer + 1) * plane];
            blit_or_bits(
                &bits.bits, bits.stride, bits.mh, dst, stride,
                r0 as usize, c0 as usize);
        }
    }

    let mut prefix = layers.clone();
    for layer in 1..max_layers {
        let base = layer * plane;
        let prev = (layer - 1) * plane;
        for word in 0..plane {
            prefix[base + word] |= prefix[prev + word];
        }
    }
    let mut suffix = layers;
    for layer in (0..max_layers.saturating_sub(1)).rev() {
        let base = layer * plane;
        let next = (layer + 1) * plane;
        for word in 0..plane {
            suffix[base + word] |= suffix[next + word];
        }
    }
    LayerAggregate { stride, plane, prefix, suffix }
}

fn layer_xor_position(eng: &Engine, gk: usize, agg: &LayerAggregate,
                      x: i64, y: i64) -> (bool, u32) {
    let g = &eng.geos[gk];
    let mut blocked_candidate = 0u32;
    let mut blocked_resident = 0u32;
    for (layer, bits) in g.layer_bits.iter().enumerate() {
        let bits = match bits {
            Some(value) => value,
            None => continue,
        };
        let r0 = y * eng.res + bits.oy;
        let c0 = x * eng.res + bits.ox;
        debug_assert!(r0 >= 0 && c0 >= 0);
        let lo = layer * agg.plane;
        let hi = lo + agg.plane;
        blocked_candidate += shifted_popcount(
            &bits.bits, bits.stride, bits.mh,
            r0 as usize, c0 as usize,
            &agg.suffix[lo..hi], agg.stride);
        blocked_resident += shifted_popcount(
            &bits.bits, bits.stride, bits.mh,
            r0 as usize, c0 as usize,
            &agg.prefix[lo..hi], agg.stride);
    }
    ((blocked_candidate != 0) ^ (blocked_resident != 0),
     blocked_candidate.max(blocked_resident))
}

// -- python-ism reproductions ------------------------------------------------

fn linspace_round_dedup(len: usize, num: usize) -> Vec<usize> {
    // np.linspace(0, len-1, num).round().astype(int) then dict.fromkeys
    let start = 0.0f64;
    let stop = (len - 1) as f64;
    let div = (num - 1) as f64;
    let step = (stop - start) / div;
    let mut out = Vec::with_capacity(num);
    let mut seen = vec![false; len];
    for i in 0..num {
        let v = if i == num - 1 { stop } else { start + i as f64 * step };
        let r = v.round_ties_even() as usize;
        if !seen[r] {
            seen[r] = true;
            out.push(r);
        }
    }
    out
}

/// Stable top-k of score indices without sorting the entire candidate grid.
/// `top_b` and `nest_lane` are small (normally 8 and 4), so maintaining a
/// sorted prefix costs O(n*k) and preserves the former stable `(score, index)`
/// ordering exactly.
fn topk_scores(vals: &[f64], k: usize) -> Vec<usize> {
    let k = k.min(vals.len());
    let mut best: Vec<usize> = Vec::with_capacity(k);
    for i in 0..vals.len() {
        let mut lo = 0usize;
        let mut hi = best.len();
        while lo < hi {
            let mid = (lo + hi) / 2;
            let j = best[mid];
            if vals[j] < vals[i] || (vals[j] == vals[i] && j < i) {
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

/// Stable nesting prefix.  The old implementation first stable-sorted by
/// normal score, filtered improving nesting candidates, then stable-sorted by
/// optimistic score.  Its exact total order is therefore
/// `(optimistic_score, normal_score, original_index)`.
fn topk_nest(scores: &[f64], optimistic: &[f64], k: usize) -> Vec<usize> {
    let mut best: Vec<usize> = Vec::with_capacity(k.min(scores.len()));
    for i in 0..scores.len() {
        if optimistic[i] >= scores[i] - 1e-9 {
            continue;
        }
        let mut lo = 0usize;
        let mut hi = best.len();
        while lo < hi {
            let mid = (lo + hi) / 2;
            let j = best[mid];
            let before = optimistic[j] < optimistic[i]
                || (optimistic[j] == optimistic[i]
                    && (scores[j] < scores[i]
                        || (scores[j] == scores[i] && j < i)));
            if before {
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

/// Reserve XOR candidates without displacing the ordinary/nesting floor.
/// One robust high-popcount direction is kept first, then the best optimistic
/// scores fill the remaining lane in stable generation order.
fn topk_layer_xor(indices: &[usize], optimistic: &[f64], pop: &[u32],
                  k: usize) -> Vec<usize> {
    if k == 0 || indices.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(k.min(indices.len()));
    let mut strongest = indices[0];
    for &idx in &indices[1..] {
        if pop[idx] > pop[strongest]
            || (pop[idx] == pop[strongest]
                && (optimistic[idx] < optimistic[strongest]
                    || (optimistic[idx] == optimistic[strongest]
                        && idx < strongest))) {
            strongest = idx;
        }
    }
    out.push(strongest);
    while out.len() < k && out.len() < indices.len() {
        let mut best: Option<usize> = None;
        for &idx in indices {
            if out.contains(&idx) {
                continue;
            }
            if best.map_or(true, |cur| {
                optimistic[idx] < optimistic[cur]
                    || (optimistic[idx] == optimistic[cur]
                        && (pop[idx] > pop[cur]
                            || (pop[idx] == pop[cur] && idx < cur)))
            }) {
                best = Some(idx);
            }
        }
        out.push(best.unwrap());
    }
    out
}

fn earliest_slot(r_b: i64, p_b: i64, conflicts: &[(i64, i64, u8)])
                 -> (i64, i64) {
    if conflicts.is_empty() {
        return (r_b, r_b + p_b);
    }
    let feasible = |t: i64| -> Option<i64> {
        let mut f_b = t + p_b;
        for _ in 0..conflicts.len() + 2 {
            let mut changed = false;
            for &(s_e, f_e, kind) in conflicts {
                if f_e <= t {
                    continue;
                }
                if f_b <= s_e {
                    continue;
                }
                if kind == KIND_INSIDE && s_e <= t && f_b <= f_e {
                    continue;
                }
                if kind == KIND_ENCLOSE && t <= s_e {
                    if f_b < f_e {
                        f_b = f_e;
                        changed = true;
                    }
                    continue;
                }
                return None;
            }
            if !changed {
                return Some(f_b);
            }
        }
        None
    };
    let mut t_after = r_b;
    for &(_, f_e, _) in conflicts {
        if f_e > t_after {
            t_after = f_e;
        }
    }
    let mut cands: BTreeSet<i64> = BTreeSet::new();
    cands.insert(r_b);
    for &(s_e, f_e, kind) in conflicts {
        if f_e > r_b {
            cands.insert(f_e);
        }
        if kind == KIND_INSIDE && s_e > r_b {
            cands.insert(s_e);
        }
    }
    let mut best = (t_after, t_after + p_b);
    for &t in cands.iter() {
        if t >= t_after {
            break;
        }
        if let Some(f_b) = feasible(t) {
            if f_b < best.1 || (f_b == best.1 && t < best.0) {
                best = (t, f_b);
            }
            break;
        }
    }
    best
}

// -- the insert loop ---------------------------------------------------------

pub struct Params {
    pub max_xs: usize,
    pub max_ys: usize,
    pub top_b: usize,
    pub nest_lane: usize,
    pub slide: bool,
    pub w1: f64,
    pub w2: f64,
    pub w3: f64,
    pub bay_prior: Vec<f64>,
    pub bay_prior_strength: f64,
    pub bay_prior_active: bool,
    pub timelimit: f64,
}

/// Plan-mode inputs (heavy realization): per-block planned start
/// (i64::MIN = absent) and planned bay (-1 = absent); target_w > 0 arms
/// the ADMM L2 proximal terms, bays restricts the bay order.
pub struct PlanIn<'a> {
    pub start: &'a [i64],
    pub bay: &'a [i32],
    pub target_w: f64,
    pub bays: bool,
}

pub struct BlockMeta {
    pub r: i64,
    pub d: i64,
    pub p: i64,
    pub wl: f64,
    pub prefs: Vec<f64>,
    pub oi_uniq: Vec<i32>,
    pub n_orient: usize,
}

struct Grid {
    bay: usize,
    _oi: i32,
    gk: usize,
    xs: Vec<i64>,
    ys: Vec<i64>,
    scores: Vec<f64>,
    scores_opt: Vec<f64>,
    cptr: Vec<i32>,
    cflat: Vec<i32>,
    rptr: Vec<i32>,
    rflat: Vec<i32>,
    rok: Vec<u8>,
    xor_bits: Vec<u64>,
    xor_pop: Vec<u32>,
}

pub fn run(eng: &mut Engine, blocks: &[BlockMeta], order: &[i32],
           fixed: &[[i64; 7]], plan: Option<PlanIn>,
           prm: &Params, out: &mut [i64]) -> i32 {
    let t0 = std::time::Instant::now();
    let m = eng.bay_w.len();
    let n_ord = if order.is_empty() { 1 } else { order.len() };
    let areas: Vec<f64> = (0..m)
        .map(|j| eng.bay_w[j] as f64 * eng.bay_h[j] as f64)
        .collect();
    let total: f64 = areas.iter().fold(0.0, |acc, &a| acc + a);
    let u: Vec<f64> = areas.iter().map(|&a| total / m as f64 / a).collect();
    let layer_lane_k = eng.layer_lane_k;
    let layer_debug = std::env::var("OGC_RUST_LAYER_DEBUG")
        .map_or(false, |value| value == "1");
    let max_layers = eng.geos.iter()
        .map(|g| g.layer_bits.len()).max().unwrap_or(0);
    let mut layer_positions = 0usize;
    let mut layer_xor_count = 0usize;
    let mut layer_selected = 0usize;
    // Phase-B lane supply census (OGC_LANE_CENSUS=1).  Answers whether the
    // lane WIDTHS are the binding constraint at all, per band: how many
    // candidates are nest-eligible, how many the width admits, how often the
    // width truncates the supply, and which lane produced the placement that
    // actually won Phase B.
    let lane_census = std::env::var("OGC_LANE_CENSUS")
        .map_or(false, |value| value == "1");
    let mut cs_blocks = 0usize;
    let mut cs_cand = 0usize;
    let mut cs_elig = 0usize;
    let mut cs_nest_admitted = 0usize;
    let mut cs_nest_capped = 0usize;
    let mut cs_ord = 0usize;
    let mut cs_win_ord = 0usize;
    let mut cs_win_nest = 0usize;
    let mut cs_ord_saturated = 0usize;

    let mut placed: Vec<Vec<Placed>> = (0..m).map(|_| Vec::new()).collect();
    let mut loads = vec![0.0f64; m];

    // partial mode: pre-place frozen blocks in caller (dict) order
    for f in fixed {
        let (bid, bay_id, x, y, oi) =
            (f[0] as i32, f[1] as usize, f[2], f[3], f[4] as i32);
        let gk = eng.gidx[&(bid, oi)];
        let g = &eng.geos[gk];
        let bay_wpx = (eng.bay_w[bay_id] * eng.res) as usize;
        let bay_hpx = (eng.bay_h[bay_id] * eng.res) as usize;
        let r0 = (y * eng.res + g.oy) as usize;
        let c0 = (x * eng.res + g.ox) as usize;
        let (bit_stride, wm) = blit_bits(
            &g.mask_bits, g.bit_stride, g.mh,
            bay_hpx, bay_wpx, r0, c0);
        let (abit_stride, wa) = blit_bits(
            &g.amask_bits, g.bit_stride, g.mh,
            bay_hpx, bay_wpx, r0, c0);
        debug_assert_eq!(bit_stride, abit_stride);
        placed[bay_id].push(Placed {
            x,
            y,
            s: f[5],
            f: f[6],
            gk,
            bbox: [x as f64 + g.bbox[0], y as f64 + g.bbox[1],
                   x as f64 + g.bbox[2], y as f64 + g.bbox[3]],
            bit_stride,
            mask_bits: wm,
            amask_bits: wa,
        });
        loads[bay_id] += blocks[bid as usize].wl;
    }

    let mut cur_xs = prm.max_xs;
    let mut cur_ys = prm.max_ys;
    let mut cur_b = prm.top_b;
    let mut cur_nest = prm.nest_lane;
    let mut orient_cap = 99usize;

    for (rank, &bid) in order.iter().enumerate() {
        let frac_used = t0.elapsed().as_secs_f64() / prm.timelimit;
        let frac_done = rank as f64 / n_ord as f64;
        if frac_used > 0.85 {
            cur_xs = 4;
            cur_ys = 3;
            cur_b = 2;
            orient_cap = 2;
            cur_nest = 0;
        } else if frac_used > frac_done + 0.12 {
            cur_xs = (cur_xs * 2 / 3).max(8);
            cur_ys = (cur_ys * 2 / 3).max(6);
            orient_cap = 4;
        }

        let bm = &blocks[bid as usize];
        let (r_b, d_b, p_b) = (bm.r, bm.d, bm.p);
        let wl = bm.wl;
        let (tgt_b, ptw_b) = match &plan {
            Some(pl) if pl.target_w > 0.0
                && pl.start[bid as usize] != i64::MIN => {
                (pl.start[bid as usize], pl.target_w)
            }
            _ => (0i64, 0.0f64),
        };
        let smax = bm.prefs.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let prior_penalties: Vec<f64> = if prm.bay_prior_active {
            (0..m).map(|j| {
                prm.w1 * prm.bay_prior_strength
                    * prm.bay_prior[bid as usize * m + j]
            }).collect()
        } else {
            vec![0.0; m]
        };
        let mut oi_list: Vec<i32> = bm.oi_uniq.clone();
        if oi_list.len() > orient_cap {
            let step = oi_list.len() as f64 / orient_cap as f64;
            oi_list = (0..orient_cap)
                .map(|i| oi_list[(i as f64 * step) as usize])
                .collect();
        }

        let mut bay_order: Vec<usize> = (0..m).collect();
        if prm.bay_prior_active {
            bay_order.sort_by(|&a, &b| {
                let ca = prm.w3 * (smax - bm.prefs[a])
                    + prior_penalties[a];
                let cb = prm.w3 * (smax - bm.prefs[b])
                    + prior_penalties[b];
                ca.partial_cmp(&cb).unwrap()
            });
        } else {
            bay_order.sort_by(|&a, &b| {
                bm.prefs[b].partial_cmp(&bm.prefs[a]).unwrap()
            });
        }
        if let Some(pl) = &plan {
            if pl.bays && pl.bay[bid as usize] >= 0 {
                bay_order = vec![pl.bay[bid as usize] as usize];
            }
        }

        let mut best_cons = f64::INFINITY;
        let mut grids: Vec<Grid> = Vec::new();

        for &bay_id in &bay_order {
            if prm.w3 * (smax - bm.prefs[bay_id])
                + prior_penalties[bay_id] > best_cons {
                continue;
            }
            let act_idx: Vec<usize> = (0..placed[bay_id].len())
                .filter(|&i| placed[bay_id][i].f > r_b)
                .collect();
            let layer_agg = if layer_lane_k > 0
                && max_layers > 0 && !act_idx.is_empty() {
                let bay_wpx = (eng.bay_w[bay_id] * eng.res) as usize;
                let bay_hpx = (eng.bay_h[bay_id] * eng.res) as usize;
                Some(build_layer_aggregate(
                    eng, &placed[bay_id], &act_idx,
                    bay_wpx, bay_hpx, max_layers))
            } else {
                None
            };
            let d2 = if m > 1 {
                let new_load = loads[bay_id] + wl;
                let mut mx = f64::NEG_INFINITY;
                for j in 0..m {
                    if j != bay_id {
                        let v = (u[bay_id] * new_load - u[j] * loads[j]).abs();
                        mx = mx.max(v);
                    }
                }
                mx
            } else { 0.0 };
            let base_score = prm.w2 * d2
                + prm.w3 * (smax - bm.prefs[bay_id])
                + prior_penalties[bay_id];

            for &oi in &oi_list {
                let gk = match eng.gidx.get(&(bid, oi)) {
                    Some(&k) if eng.geos[k].union.is_some() => k,
                    _ => continue,
                };
                let (lx0, ly0, lx1, ly1) = {
                    let b = eng.geos[gk].bbox;
                    (b[0], b[1], b[2], b[3])
                };
                let px_lo = (-lx0).ceil() as i64;
                let px_hi = (eng.bay_w[bay_id] as f64 - lx1).floor() as i64;
                let py_lo = (-ly0).ceil() as i64;
                let py_hi = (eng.bay_h[bay_id] as f64 - ly1).floor() as i64;
                if px_lo > px_hi || py_lo > py_hi {
                    continue;
                }
                let mut xset: BTreeSet<i64> = BTreeSet::new();
                let mut yset: BTreeSet<i64> = BTreeSet::new();
                xset.insert(px_lo);
                xset.insert(px_hi);
                yset.insert(py_lo);
                yset.insert(py_hi);
                for &ei in &act_idx {
                    let e = &placed[bay_id][ei];
                    let (ex0, ey0, ex1, ey1) =
                        (e.bbox[0], e.bbox[1], e.bbox[2], e.bbox[3]);
                    xset.insert((ex1 - lx0).ceil() as i64);
                    xset.insert((ex0 - lx1).floor() as i64);
                    xset.insert((ex0 - lx0).ceil() as i64);
                    yset.insert((ey1 - ly0).ceil() as i64);
                    yset.insert((ey0 - ly1).floor() as i64);
                    yset.insert((ey0 - ly0).ceil() as i64);
                }
                let mut xs: Vec<i64> = xset.into_iter()
                    .filter(|&v| px_lo <= v && v <= px_hi).collect();
                let mut ys: Vec<i64> = yset.into_iter()
                    .filter(|&v| py_lo <= v && v <= py_hi).collect();
                if xs.len() > cur_xs {
                    xs = linspace_round_dedup(xs.len(), cur_xs)
                        .into_iter().map(|i| xs[i]).collect();
                }
                if ys.len() > cur_ys {
                    ys = linspace_round_dedup(ys.len(), cur_ys)
                        .into_iter().map(|i| ys[i]).collect();
                }
                if xs.is_empty() || ys.is_empty() {
                    continue;
                }
                let nx = xs.len();
                let p_total = nx * ys.len();

                // ---- tiers (rc_k_tiers semantics on per-active masks) ----
                let mut cptr = vec![0i32; p_total + 1];
                let mut rptr = vec![0i32; p_total + 1];
                let mut cflat: Vec<i32> = Vec::new();
                let mut rflat: Vec<i32> = Vec::new();
                {
                    let g = &eng.geos[gk];
                    for p in 0..p_total {
                        let y = ys[p / nx];
                        let x = xs[p % nx];
                        let r0 = y * eng.res + g.oy;
                        let c0 = x * eng.res + g.ox;
                        let wy0 = y as f64 + ly0;
                        let wy1 = y as f64 + ly1;
                        let wx0 = x as f64 + lx0;
                        let wx1 = x as f64 + lx1;
                        for (k, &ei) in act_idx.iter().enumerate() {
                            let e = &placed[bay_id][ei];
                            if !(wx0 < e.bbox[2] && e.bbox[0] < wx1
                                 && wy0 < e.bbox[3] && e.bbox[1] < wy1) {
                                continue;
                            }
                            if !shifted_overlap(
                                &g.mask_bits, g.bit_stride, g.mh,
                                r0 as usize, c0 as usize,
                                &e.mask_bits, e.bit_stride)
                            {
                                continue;
                            }
                            if shifted_overlap(
                                &g.amask_bits, g.bit_stride, g.mh,
                                r0 as usize, c0 as usize,
                                &e.amask_bits, e.bit_stride)
                            {
                                cflat.push(k as i32);
                            } else {
                                rflat.push(k as i32);
                            }
                        }
                        cptr[p + 1] = cflat.len() as i32;
                        rptr[p + 1] = rflat.len() as i32;
                    }
                }

                // ---- resolve the residual band (memoised exact) ----------
                let mut rok = vec![0u8; rflat.len()];
                for p in 0..p_total {
                    let y = ys[p / nx];
                    let x = xs[p % nx];
                    for ti in rptr[p] as usize..rptr[p + 1] as usize {
                        let k = rflat[ti] as usize;
                        let e = &placed[bay_id][act_idx[k]];
                        let key = (e.gk, gk, e.x, e.y, x, y);
                        let v = if let Some(&v) = eng.pair_memo.get(&key) {
                            v
                        } else {
                            let ge = eng.geos[e.gk].union.as_ref().unwrap();
                            let gc = eng.geos[gk].union.as_ref().unwrap();
                            let v = geom_intersects(
                                ge, e.x as f64, e.y as f64,
                                gc, x as f64, y as f64);
                            if eng.pair_memo.len() >= MEMO_CAP {
                                eng.pair_memo.clear();
                            }
                            eng.pair_memo.insert(key, v);
                            v
                        };
                        if v {
                            rok[ti] = 1;
                        }
                    }
                }

                // ---- scores (rc_k_score semantics) -----------------------
                let mut scores = vec![0.0f64; p_total];
                let mut scores_opt = vec![0.0f64; p_total];
                {
                    let act_s: Vec<i64> = act_idx.iter()
                        .map(|&ei| placed[bay_id][ei].s).collect();
                    let act_f: Vec<i64> = act_idx.iter()
                        .map(|&ei| placed[bay_id][ei].f).collect();
                    let mut keys: Vec<i64> = Vec::new();
                    let mut hard: Vec<i64> = Vec::new();
                    for p in 0..p_total {
                        let y = ys[p / nx];
                        keys.clear();
                        for i in cptr[p]..cptr[p + 1] {
                            let a = cflat[i as usize] as usize;
                            keys.push((act_s[a] << 20) | act_f[a]);
                        }
                        for i in rptr[p]..rptr[p + 1] {
                            if rok[i as usize] != 0 {
                                let a = rflat[i as usize] as usize;
                                keys.push((act_s[a] << 20) | act_f[a]);
                            }
                        }
                        let n_iv = keys.len();
                        for i in 1..n_iv {
                            let kk = keys[i];
                            let mut j = i;
                            while j > 0 && keys[j - 1] > kk {
                                keys[j] = keys[j - 1];
                                j -= 1;
                            }
                            keys[j] = kk;
                        }
                        let mut t = r_b;
                        for &key in keys.iter() {
                            let s_e = key >> 20;
                            let f_e = key & 0xFFFFF;
                            if t + p_b <= s_e {
                                break;
                            }
                            if f_e > t {
                                t = f_e;
                            }
                        }
                        let entry = t;
                        let tard = (entry + p_b - d_b).max(0);
                        let dtg = (entry - tgt_b).abs();
                        let sc = base_score + prm.w1 * tard as f64
                            + 1e-4 * (y as f64 + ly1)
                            + 1e-6 * entry as f64
                            + ptw_b * dtg as f64;
                        let mut sco = sc;
                        if cur_nest > 0 && n_iv > 0 {
                            hard.clear();
                            let rpb = r_b + p_b;
                            for &key in keys.iter() {
                                let s_e = key >> 20;
                                let f_e = key & 0xFFFFF;
                                if !(s_e <= r_b && rpb <= f_e) {
                                    hard.push(key);
                                }
                            }
                            if hard.len() < n_iv {
                                let mut t = r_b;
                                for &key in hard.iter() {
                                    let s_e = key >> 20;
                                    let f_e = key & 0xFFFFF;
                                    if t + p_b <= s_e {
                                        break;
                                    }
                                    if f_e > t {
                                        t = f_e;
                                    }
                                }
                                let t_opt = (t + p_b - d_b).max(0);
                                let dtg = (t - tgt_b).abs();
                                sco = base_score + prm.w1 * t_opt as f64
                                    + 1e-4 * (y as f64 + ly1)
                                    + 1e-6 * t as f64
                                    + ptw_b * dtg as f64;
                            }
                        }
                        scores[p] = sc;
                        scores_opt[p] = sco;
                    }
                }
                for &s in &scores {
                    if s < best_cons {
                        best_cons = s;
                    }
                }
                let mut xor_bits = vec![0u64; (p_total + 63) / 64];
                let mut xor_pop = vec![0u32; p_total];
                if let Some(agg) = &layer_agg {
                    for p in 0..p_total {
                        let y = ys[p / nx];
                        let x = xs[p % nx];
                        let (is_xor, count) =
                            layer_xor_position(eng, gk, agg, x, y);
                        layer_positions += 1;
                        if is_xor {
                            xor_bits[p / 64] |= 1u64 << (p & 63);
                            xor_pop[p] = count;
                            layer_xor_count += 1;
                        }
                    }
                }
                grids.push(Grid {
                    bay: bay_id,
                    _oi: oi,
                    gk,
                    xs,
                    ys,
                    scores,
                    scores_opt,
                    cptr,
                    cflat,
                    rptr,
                    rflat,
                    rok,
                    xor_bits,
                    xor_pop,
                });
            }
        }

        // ---- selection ---------------------------------------------------
        struct Sel {
            bay: usize,
            gk: usize,
            oi: i32,
            x: i64,
            y: i64,
            conf: Vec<i32>,
        }
        let mut selected: Vec<Sel> = Vec::new();
        let mut n_ordinary = 0usize;
        if !grids.is_empty() {
            let mut sc_all: Vec<f64> = Vec::new();
            let mut so_all: Vec<f64> = Vec::new();
            let mut pop_all: Vec<u32> = Vec::new();
            let mut xor_indices: Vec<usize> = Vec::new();
            for gr in &grids {
                let base = sc_all.len();
                sc_all.extend_from_slice(&gr.scores);
                so_all.extend_from_slice(&gr.scores_opt);
                pop_all.extend_from_slice(&gr.xor_pop);
                for p in 0..gr.scores.len() {
                    if gr.xor_bits[p / 64] & (1u64 << (p & 63)) != 0 {
                        xor_indices.push(base + p);
                    }
                }
            }
            let mut sel_idx = topk_scores(&sc_all, cur_b);
            n_ordinary = sel_idx.len();
            if lane_census {
                let elig = (0..sc_all.len())
                    .filter(|&i| so_all[i] < sc_all[i] - 1e-9)
                    .count();
                cs_blocks += 1;
                cs_cand += sc_all.len();
                cs_elig += elig;
                cs_ord += n_ordinary;
                if sc_all.len() > cur_b {
                    cs_ord_saturated += 1;
                }
                if elig > cur_nest {
                    cs_nest_capped += 1;
                }
            }
            if cur_nest > 0 {
                for idx in topk_nest(&sc_all, &so_all, cur_nest) {
                    if !sel_idx.contains(&idx) {
                        sel_idx.push(idx);
                        if lane_census {
                            cs_nest_admitted += 1;
                        }
                    }
                }
            }
            for idx in topk_layer_xor(
                    &xor_indices, &so_all, &pop_all, layer_lane_k) {
                if !sel_idx.contains(&idx) {
                    sel_idx.push(idx);
                    layer_selected += 1;
                }
            }
            let mut gstart = vec![0usize; grids.len() + 1];
            for (i, gr) in grids.iter().enumerate() {
                gstart[i + 1] = gstart[i] + gr.scores.len();
            }
            for idx in sel_idx {
                let gi = match gstart.binary_search_by(|v| {
                    if *v <= idx { std::cmp::Ordering::Less }
                    else { std::cmp::Ordering::Greater }
                }) {
                    Ok(i) | Err(i) => i - 1,
                };
                let gr = &grids[gi];
                let p = idx - gstart[gi];
                let nxl = gr.xs.len();
                let x = gr.xs[p % nxl];
                let y = gr.ys[p / nxl];
                let mut conf: Vec<i32> = gr.cflat
                    [gr.cptr[p] as usize..gr.cptr[p + 1] as usize].to_vec();
                for ti in gr.rptr[p] as usize..gr.rptr[p + 1] as usize {
                    if gr.rok[ti] != 0 {
                        conf.push(gr.rflat[ti]);
                    }
                }
                conf.sort_unstable();
                selected.push(Sel {
                    bay: gr.bay,
                    gk: gr.gk,
                    oi: gr._oi,
                    x,
                    y,
                    conf,
                });
            }
        } else {
            // safety net: wall candidates over ALL bays and orientations
            for &bay_id in &bay_order {
                let act_idx: Vec<usize> = (0..placed[bay_id].len())
                    .filter(|&i| placed[bay_id][i].f > r_b)
                    .collect();
                for oi in 0..bm.n_orient as i32 {
                    let gk = match eng.gidx.get(&(bid, oi)) {
                        Some(&k) if eng.geos[k].union.is_some() => k,
                        _ => continue,
                    };
                    let b = eng.geos[gk].bbox;
                    let (lx0, ly0, lx1, ly1) = (b[0], b[1], b[2], b[3]);
                    let px_lo = (-lx0).ceil() as i64;
                    let py_lo = (-ly0).ceil() as i64;
                    if px_lo > (eng.bay_w[bay_id] as f64 - lx1).floor() as i64
                        || py_lo
                        > (eng.bay_h[bay_id] as f64 - ly1).floor() as i64 {
                        continue;
                    }
                    let wbb = (px_lo as f64 + lx0, py_lo as f64 + ly0,
                               px_lo as f64 + lx1, py_lo as f64 + ly1);
                    let conf: Vec<i32> = act_idx.iter().enumerate()
                        .filter(|(_, &ei)| {
                            let e = &placed[bay_id][ei];
                            wbb.0 < e.bbox[2] && e.bbox[0] < wbb.2
                                && wbb.1 < e.bbox[3] && e.bbox[1] < wbb.3
                        })
                        .map(|(k, _)| k as i32)
                        .collect();
                    selected.push(Sel {
                        bay: bay_id,
                        gk,
                        oi,
                        x: px_lo,
                        y: py_lo,
                        conf,
                    });
                }
            }
            if selected.is_empty() {
                return -(bid as i32) - 1; // no valid placement
            }
            // python: cands sorted by score (all inf -> stable order kept),
            // top cur_b; the nest lane never fires on inf scores
            selected.truncate(cur_b);
        }

        // ---- Phase B: exact refinement ----------------------------------
        let mut best: Option<(usize, i64, i64, i32, usize, i64, i64,
                              Vec<(i64, i64, u8)>)> = None;
        let mut best_score = f64::INFINITY;
        let mut best_lane = usize::MAX;
        for (sel_rank, sel) in selected.iter().enumerate() {
            let bay_id = sel.bay;
            let act_idx: Vec<usize> = (0..placed[bay_id].len())
                .filter(|&i| placed[bay_id][i].f > r_b)
                .collect();
            let mut x = sel.x;
            let mut y = sel.y;
            let mut conflicts: Vec<(i64, i64, u8)> = Vec::new();
            for &k in &sel.conf {
                let ei = act_idx[k as usize];
                let (ek, ex, ey, es, ef) = {
                    let e = &placed[bay_id][ei];
                    (e.gk, e.x, e.y, e.s, e.f)
                };
                let (kind_raw, under_is_b) =
                    eng.relation(ek, ex, ey, sel.gk, x, y);
                if kind_raw == 3 {
                    continue; // FREE: the co-presence screen was touch-only
                }
                let kind = if kind_raw == KIND_PAST {
                    KIND_PAST
                } else if under_is_b {
                    KIND_ENCLOSE
                } else {
                    KIND_INSIDE
                };
                conflicts.push((es, ef, kind));
            }
            let (entry, exit_t) = earliest_slot(r_b, p_b, &conflicts);
            if prm.slide && conflicts.iter().all(|&(_, f_e, _)| f_e <= entry) {
                // bottom-left polygon-contact compaction
                let g = &eng.geos[sel.gk];
                let (lx0, ly0, lx1, ly1) =
                    (g.bbox[0], g.bbox[1], g.bbox[2], g.bbox[3]);
                let px_lo = (-lx0).ceil() as i64;
                let py_lo = (-ly0).ceil() as i64;
                let obs: Vec<usize> = act_idx.iter().copied()
                    .filter(|&ei| placed[bay_id][ei].f > entry).collect();
                let bay_wpx = (eng.bay_w[bay_id] * eng.res) as usize;
                let mut steps = 0;
                let mut moved = true;
                while moved && steps < 80 {
                    moved = false;
                    loop {
                        if !(y > py_lo && steps < 80) {
                            break;
                        }
                        let (nx, ny) = (x, y - 1);
                        let mut ok = true;
                        let wbb = (nx as f64 + lx0, ny as f64 + ly0,
                                   nx as f64 + lx1, ny as f64 + ly1);
                        for &ei in &obs {
                            let (ebb, ex, ey, ek) = {
                                let e = &placed[bay_id][ei];
                                (e.bbox, e.x, e.y, e.gk)
                            };
                            if !(wbb.0 < ebb[2] && ebb[0] < wbb.2
                                 && wbb.1 < ebb[3] && ebb[1] < wbb.3) {
                                continue;
                            }
                            let e = &placed[bay_id][ei];
                            if eng.pair_exact(
                                ek, ex, ey, &e.mask_bits, &e.amask_bits,
                                e.bit_stride, bay_wpx, sel.gk, nx, ny)
                            {
                                ok = false;
                                break;
                            }
                        }
                        if !ok {
                            break;
                        }
                        y = ny;
                        steps += 1;
                        moved = true;
                    }
                    loop {
                        if !(x > px_lo && steps < 80) {
                            break;
                        }
                        let (nx, ny) = (x - 1, y);
                        let mut ok = true;
                        let wbb = (nx as f64 + lx0, ny as f64 + ly0,
                                   nx as f64 + lx1, ny as f64 + ly1);
                        for &ei in &obs {
                            let (ebb, ex, ey, ek) = {
                                let e = &placed[bay_id][ei];
                                (e.bbox, e.x, e.y, e.gk)
                            };
                            if !(wbb.0 < ebb[2] && ebb[0] < wbb.2
                                 && wbb.1 < ebb[3] && ebb[1] < wbb.3) {
                                continue;
                            }
                            let e = &placed[bay_id][ei];
                            if eng.pair_exact(
                                ek, ex, ey, &e.mask_bits, &e.amask_bits,
                                e.bit_stride, bay_wpx, sel.gk, nx, ny)
                            {
                                ok = false;
                                break;
                            }
                        }
                        if !ok {
                            break;
                        }
                        x = nx;
                        steps += 1;
                        moved = true;
                    }
                }
            }
            let tard = (exit_t - d_b).max(0);
            let d2 = if m > 1 {
                let new_load = loads[bay_id] + wl;
                let mut mx = f64::NEG_INFINITY;
                for j in 0..m {
                    if j != bay_id {
                        let v = (u[bay_id] * new_load - u[j] * loads[j]).abs();
                        mx = mx.max(v);
                    }
                }
                mx
            } else { 0.0 };
            let ly1g = eng.geos[sel.gk].bbox[3];
            let score = prm.w1 * tard as f64 + prm.w2 * d2
                + prm.w3 * (smax - bm.prefs[bay_id])
                + prior_penalties[bay_id]
                + 1e-4 * (y as f64 + ly1g) + 1e-6 * entry as f64
                + ptw_b * (entry - tgt_b).abs() as f64;
            if score < best_score {
                best_score = score;
                best_lane = sel_rank;
                best = Some((bay_id, x, y, sel.oi, sel.gk, entry, exit_t,
                             conflicts));
            }
        }
        if lane_census && !selected.is_empty() {
            if best_lane < n_ordinary {
                cs_win_ord += 1;
            } else {
                cs_win_nest += 1;
            }
        }

        let (bay_id, x, y, oi, gk, entry, exit_t, _conflicts) = best.unwrap();
        // world mask blit (in-bay placement guarantees non-negative origin)
        let bay_wpx = (eng.bay_w[bay_id] * eng.res) as usize;
        let bay_hpx = (eng.bay_h[bay_id] * eng.res) as usize;
        let g = &eng.geos[gk];
        let r0 = (y * eng.res + g.oy) as usize;
        let c0 = (x * eng.res + g.ox) as usize;
        let (bit_stride, wm) = blit_bits(
            &g.mask_bits, g.bit_stride, g.mh,
            bay_hpx, bay_wpx, r0, c0);
        let (abit_stride, wa) = blit_bits(
            &g.amask_bits, g.bit_stride, g.mh,
            bay_hpx, bay_wpx, r0, c0);
        debug_assert_eq!(bit_stride, abit_stride);
        placed[bay_id].push(Placed {
            x,
            y,
            s: entry,
            f: exit_t,
            gk,
            bbox: [x as f64 + g.bbox[0], y as f64 + g.bbox[1],
                   x as f64 + g.bbox[2], y as f64 + g.bbox[3]],
            bit_stride,
            mask_bits: wm,
            amask_bits: wa,
        });
        loads[bay_id] += wl;
        let o = &mut out[6 * bid as usize..6 * bid as usize + 6];
        o[0] = bay_id as i64;
        o[1] = x;
        o[2] = y;
        o[3] = oi as i64;
        o[4] = entry;
        o[5] = exit_t;
    }
    if lane_census {
        eprintln!(
            "[LANES] blocks={} cand/blk={:.1} ord/blk={:.2} elig/blk={:.2} nest_admitted/blk={:.2} nest_capped={:.1}% ord_saturated={:.1}% win_ordinary={} win_nest={} ({:.1}%)",
            cs_blocks,
            cs_cand as f64 / cs_blocks.max(1) as f64,
            cs_ord as f64 / cs_blocks.max(1) as f64,
            cs_elig as f64 / cs_blocks.max(1) as f64,
            cs_nest_admitted as f64 / cs_blocks.max(1) as f64,
            100.0 * cs_nest_capped as f64 / cs_blocks.max(1) as f64,
            100.0 * cs_ord_saturated as f64 / cs_blocks.max(1) as f64,
            cs_win_ord, cs_win_nest,
            100.0 * cs_win_nest as f64
                / (cs_win_ord + cs_win_nest).max(1) as f64);
    }
    if layer_debug {
        eprintln!(
            "[RLAYER] positions={} xor={} selected={} wall={:.3}",
            layer_positions, layer_xor_count, layer_selected,
            t0.elapsed().as_secs_f64());
    }
    0
}

#[cfg(test)]
mod bitmask_tests {
    use super::{
        blit_bits, pack_rows, shifted_overlap, shifted_popcount, topk_layer_xor,
    };

    fn unpack(bits: &[u64], stride: usize, h: usize, w: usize) -> Vec<u8> {
        let mut out = vec![0u8; h * w];
        for r in 0..h {
            for c in 0..w {
                out[r * w + c] =
                    ((bits[r * stride + c / 64] >> (c & 63)) & 1) as u8;
            }
        }
        out
    }

    fn byte_overlap(
        src: &[u8],
        h: usize,
        w: usize,
        r0: usize,
        c0: usize,
        dst: &[u8],
        dst_w: usize,
    ) -> bool {
        for r in 0..h {
            for c in 0..w {
                if src[r * w + c] != 0
                    && dst[(r0 + r) * dst_w + c0 + c] != 0
                {
                    return true;
                }
            }
        }
        false
    }

    fn byte_popcount(
        src: &[u8],
        h: usize,
        w: usize,
        r0: usize,
        c0: usize,
        dst: &[u8],
        dst_w: usize,
    ) -> u32 {
        let mut count = 0u32;
        for r in 0..h {
            for c in 0..w {
                if src[r * w + c] != 0
                    && dst[(r0 + r) * dst_w + c0 + c] != 0
                {
                    count += 1;
                }
            }
        }
        count
    }

    #[test]
    fn xor_lane_keeps_strongest_then_best_optimistic_scores() {
        let indices = [1usize, 3, 4, 6];
        let optimistic = [0.0, 8.0, 0.0, 2.0, 1.0, 0.0, 3.0];
        let pop = [0u32, 2, 0, 4, 1, 0, 9];
        assert_eq!(
            topk_layer_xor(&indices, &optimistic, &pop, 3),
            vec![6, 4, 3],
        );
    }

    #[test]
    fn packed_rows_preserve_every_cell_and_shifted_overlap() {
        let widths = [1usize, 2, 31, 63, 64, 65, 95, 127, 130];
        let shifts = [0usize, 1, 31, 63, 64, 65, 127];
        let mut state = 0x9e37_79b9_7f4a_7c15u64;

        for &w in &widths {
            for h in 1usize..=5 {
                let mut src = vec![0u8; h * w];
                for v in &mut src {
                    state ^= state << 7;
                    state ^= state >> 9;
                    state ^= state << 8;
                    *v = ((state >> 61) == 0) as u8;
                }
                // Always exercise both ends of every packed row.
                for r in 0..h {
                    src[r * w] = 1;
                    src[r * w + w - 1] = 1;
                }

                let (src_stride, src_bits) = pack_rows(&src, h, w);
                assert_eq!(unpack(&src_bits, src_stride, h, w), src);

                for &c0 in &shifts {
                    let r0 = 2usize;
                    let dst_h = h + r0 + 2;
                    let dst_w = c0 + w + 67;
                    let mut dst = vec![0u8; dst_h * dst_w];
                    for v in &mut dst {
                        state ^= state << 7;
                        state ^= state >> 9;
                        state ^= state << 8;
                        *v = ((state >> 60) == 0) as u8;
                    }
                    let (dst_stride, dst_bits) =
                        pack_rows(&dst, dst_h, dst_w);
                    assert_eq!(
                        shifted_overlap(
                            &src_bits,
                            src_stride,
                            h,
                            r0,
                            c0,
                            &dst_bits,
                            dst_stride,
                        ),
                        byte_overlap(&src, h, w, r0, c0, &dst, dst_w),
                        "w={w} h={h} c0={c0}",
                    );
                    assert_eq!(
                        shifted_popcount(
                            &src_bits,
                            src_stride,
                            h,
                            r0,
                            c0,
                            &dst_bits,
                            dst_stride,
                        ),
                        byte_popcount(&src, h, w, r0, c0, &dst, dst_w),
                        "popcount w={w} h={h} c0={c0}",
                    );

                    let (blit_stride, blit) = blit_bits(
                        &src_bits,
                        src_stride,
                        h,
                        dst_h,
                        dst_w,
                        r0,
                        c0,
                    );
                    let unpacked = unpack(&blit, blit_stride, dst_h, dst_w);
                    for r in 0..dst_h {
                        for c in 0..dst_w {
                            let expected = if r0 <= r
                                && r < r0 + h
                                && c0 <= c
                                && c < c0 + w
                            {
                                src[(r - r0) * w + c - c0]
                            } else {
                                0
                            };
                            assert_eq!(
                                unpacked[r * dst_w + c],
                                expected,
                                "blit w={w} h={h} c0={c0} r={r} c={c}",
                            );
                        }
                    }
                }
            }
        }
    }

}
