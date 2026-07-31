// rustcore -- Rust port of the OGC2026 native geometry engine (stage 1:
// exact predicates + polygon verdicts).  The constructor insert engine
// lands here in later stages; this stage must be verdict-identical to
// geoexact.cpp, which the parity fuzz harness (test_rust_parity.py)
// checks pair by pair.
//
// Float discipline: every operation below rounds to f64 exactly as the
// C++ does (Rust never contracts a*b+c into fma on its own), and the
// Shewchuk expansions are transcribed operation for operation.  World
// coordinates are vertex + offset, one IEEE add per coordinate -- the
// same values numpy/GEOS/geoexact see.

#![allow(clippy::too_many_arguments)]

mod engine;
mod win;

// ---------------------------------------------------------------------------
// Shewchuk exact predicates (orient2d), operation-for-operation port of
// the geoexact.cpp transcription.
// ---------------------------------------------------------------------------

const SPLITTER: f64 = 134217729.0; // 2^27 + 1
const EPSILON: f64 = 1.110_223_024_625_156_5e-16; // 2^-53
const RESULTERRBOUND: f64 = (3.0 + 8.0 * EPSILON) * EPSILON;
const CCWERRBOUND_A: f64 = (3.0 + 16.0 * EPSILON) * EPSILON;
const CCWERRBOUND_B: f64 = (2.0 + 12.0 * EPSILON) * EPSILON;
const CCWERRBOUND_C: f64 = (9.0 + 64.0 * EPSILON) * EPSILON * EPSILON;

#[inline(always)]
fn fast_two_sum(a: f64, b: f64) -> (f64, f64) {
    let x = a + b;
    let bvirt = x - a;
    (x, b - bvirt)
}

#[inline(always)]
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let x = a + b;
    let bvirt = x - a;
    let avirt = x - bvirt;
    let bround = b - bvirt;
    let around = a - avirt;
    (x, around + bround)
}

#[inline(always)]
fn two_diff(a: f64, b: f64) -> (f64, f64) {
    let x = a - b;
    let bvirt = a - x;
    let avirt = x + bvirt;
    let bround = bvirt - b;
    let around = a - avirt;
    (x, around + bround)
}

#[inline(always)]
fn split(a: f64) -> (f64, f64) {
    let c = SPLITTER * a;
    let abig = c - a;
    let ahi = c - abig;
    (ahi, a - ahi)
}

#[inline(always)]
fn two_product(a: f64, b: f64) -> (f64, f64) {
    let x = a * b;
    let (ahi, alo) = split(a);
    let (bhi, blo) = split(b);
    let err1 = x - (ahi * bhi);
    let err2 = err1 - (alo * bhi);
    let err3 = err2 - (ahi * blo);
    (x, (alo * blo) - err3)
}

#[inline(always)]
fn two_one_diff(a1: f64, a0: f64, b: f64) -> (f64, f64, f64) {
    let (i, x0) = two_diff(a0, b);
    let (x2, x1) = two_sum(a1, i);
    (x2, x1, x0)
}

#[inline(always)]
fn two_two_diff(a1: f64, a0: f64, b1: f64, b0: f64) -> (f64, f64, f64, f64) {
    let (j, r0, x0) = two_one_diff(a1, a0, b0);
    let (x3, x2, x1) = two_one_diff(j, r0, b1);
    (x3, x2, x1, x0)
}

fn fast_expansion_sum_zeroelim(e: &[f64], f: &[f64], h: &mut [f64]) -> usize {
    let elen = e.len();
    let flen = f.len();
    let mut enow = e[0];
    let mut fnow = f[0];
    let mut eindex = 0usize;
    let mut findex = 0usize;
    let mut hindex = 0usize;
    let mut q: f64;

    if (fnow > enow) == (fnow > -enow) {
        q = enow;
        eindex += 1;
        if eindex < elen {
            enow = e[eindex];
        }
    } else {
        q = fnow;
        findex += 1;
        if findex < flen {
            fnow = f[findex];
        }
    }
    if eindex < elen && findex < flen {
        let (qnew, hh) = if (fnow > enow) == (fnow > -enow) {
            let r = fast_two_sum(enow, q);
            eindex += 1;
            if eindex < elen {
                enow = e[eindex];
            }
            r
        } else {
            let r = fast_two_sum(fnow, q);
            findex += 1;
            if findex < flen {
                fnow = f[findex];
            }
            r
        };
        q = qnew;
        if hh != 0.0 {
            h[hindex] = hh;
            hindex += 1;
        }
        while eindex < elen && findex < flen {
            let (qnew, hh) = if (fnow > enow) == (fnow > -enow) {
                let r = two_sum(q, enow);
                eindex += 1;
                if eindex < elen {
                    enow = e[eindex];
                }
                r
            } else {
                let r = two_sum(q, fnow);
                findex += 1;
                if findex < flen {
                    fnow = f[findex];
                }
                r
            };
            q = qnew;
            if hh != 0.0 {
                h[hindex] = hh;
                hindex += 1;
            }
        }
    }
    while eindex < elen {
        let (qnew, hh) = two_sum(q, enow);
        eindex += 1;
        if eindex < elen {
            enow = e[eindex];
        }
        q = qnew;
        if hh != 0.0 {
            h[hindex] = hh;
            hindex += 1;
        }
    }
    while findex < flen {
        let (qnew, hh) = two_sum(q, fnow);
        findex += 1;
        if findex < flen {
            fnow = f[findex];
        }
        q = qnew;
        if hh != 0.0 {
            h[hindex] = hh;
            hindex += 1;
        }
    }
    if q != 0.0 || hindex == 0 {
        h[hindex] = q;
        hindex += 1;
    }
    hindex
}

fn estimate(e: &[f64]) -> f64 {
    let mut q = e[0];
    for v in &e[1..] {
        q += v;
    }
    q
}

fn orient2d_adapt(pax: f64, pay: f64, pbx: f64, pby: f64, pcx: f64, pcy: f64,
                  detsum: f64) -> f64 {
    let acx = pax - pcx;
    let bcx = pbx - pcx;
    let acy = pay - pcy;
    let bcy = pby - pcy;

    let (detleft, detlefttail) = two_product(acx, bcy);
    let (detright, detrighttail) = two_product(acy, bcx);

    let (b3, b2, b1, b0) = two_two_diff(detleft, detlefttail,
                                        detright, detrighttail);
    let b = [b0, b1, b2, b3];

    let mut det = estimate(&b);
    let errbound = CCWERRBOUND_B * detsum;
    if det >= errbound || -det >= errbound {
        return det;
    }

    let (_, acxtail) = two_diff(pax, pcx);
    let (_, bcxtail) = two_diff(pbx, pcx);
    let (_, acytail) = two_diff(pay, pcy);
    let (_, bcytail) = two_diff(pby, pcy);

    if acxtail == 0.0 && acytail == 0.0 && bcxtail == 0.0 && bcytail == 0.0 {
        return det;
    }

    let errbound = CCWERRBOUND_C * detsum + RESULTERRBOUND * det.abs();
    det += (acx * bcytail + bcy * acxtail) - (acy * bcxtail + bcx * acytail);
    if det >= errbound || -det >= errbound {
        return det;
    }

    let (s1, s0) = two_product(acxtail, bcy);
    let (t1, t0) = two_product(acytail, bcx);
    let (u3, u2, u1, u0) = two_two_diff(s1, s0, t1, t0);
    let u = [u0, u1, u2, u3];
    let mut c1 = [0.0f64; 8];
    let c1len = fast_expansion_sum_zeroelim(&b, &u, &mut c1);

    let (s1, s0) = two_product(acx, bcytail);
    let (t1, t0) = two_product(acy, bcxtail);
    let (u3, u2, u1, u0) = two_two_diff(s1, s0, t1, t0);
    let u = [u0, u1, u2, u3];
    let mut c2 = [0.0f64; 12];
    let c2len = fast_expansion_sum_zeroelim(&c1[..c1len], &u, &mut c2);

    let (s1, s0) = two_product(acxtail, bcytail);
    let (t1, t0) = two_product(acytail, bcxtail);
    let (u3, u2, u1, u0) = two_two_diff(s1, s0, t1, t0);
    let u = [u0, u1, u2, u3];
    let mut d = [0.0f64; 16];
    let dlen = fast_expansion_sum_zeroelim(&c2[..c2len], &u, &mut d);

    d[dlen - 1]
}

pub fn orient2d(pax: f64, pay: f64, pbx: f64, pby: f64,
                pcx: f64, pcy: f64) -> f64 {
    let detleft = (pax - pcx) * (pby - pcy);
    let detright = (pay - pcy) * (pbx - pcx);
    let det = detleft - detright;

    let detsum;
    if detleft > 0.0 {
        if detright <= 0.0 {
            return det;
        }
        detsum = detleft + detright;
    } else if detleft < 0.0 {
        if detright >= 0.0 {
            return det;
        }
        detsum = -detleft - detright;
    } else {
        return det;
    }
    let errbound = CCWERRBOUND_A * detsum;
    if det >= errbound || -det >= errbound {
        return det;
    }
    orient2d_adapt(pax, pay, pbx, pby, pcx, pcy, detsum)
}

#[inline(always)]
fn osign(ax: f64, ay: f64, bx: f64, by: f64, cx: f64, cy: f64) -> i32 {
    let d = orient2d(ax, ay, bx, by, cx, cy);
    (d > 0.0) as i32 - (d < 0.0) as i32
}

// GEOS, used by the official checker for `intersection.area > 0`, collapses
// intended decimal-rotation contacts that land a few ulps across a supporting
// line to a line or point.  Exact orientation alone instead sees a microscopic
// sliver.  Apply this narrow filter only to the touch-exclusive overlap path;
// touch-inclusive intersection retains the exact predicate above.
#[inline(always)]
fn osign_overlap(ax: f64, ay: f64, bx: f64, by: f64,
                 cx: f64, cy: f64) -> i32 {
    let acx = ax - cx;
    let bcy = by - cy;
    let acy = ay - cy;
    let bcx = bx - cx;
    let d = orient2d(ax, ay, bx, by, cx, cy);
    let tol = 64.0 * EPSILON * ((acx * bcy).abs() + (acy * bcx).abs());
    if d.abs() <= tol {
        0
    } else {
        (d > 0.0) as i32 - (d < 0.0) as i32
    }
}

// ---------------------------------------------------------------------------
// Segment / point primitives (ports of geoexact.cpp)
// ---------------------------------------------------------------------------

#[inline(always)]
fn onseg(px: f64, py: f64, qx: f64, qy: f64, rx: f64, ry: f64) -> bool {
    (if px < qx { px <= rx && rx <= qx } else { qx <= rx && rx <= px })
        && (if py < qy { py <= ry && ry <= qy } else { qy <= ry && ry <= py })
}

fn seg_intersects(ax: f64, ay: f64, bx: f64, by: f64,
                  cx: f64, cy: f64, dx: f64, dy: f64) -> bool {
    let d1 = osign(cx, cy, dx, dy, ax, ay);
    let d2 = osign(cx, cy, dx, dy, bx, by);
    if d1 != 0 && d1 == d2 {
        return false;
    }
    let d3 = osign(ax, ay, bx, by, cx, cy);
    let d4 = osign(ax, ay, bx, by, dx, dy);
    if d3 != 0 && d3 == d4 {
        return false;
    }
    if d1 != 0 && d2 != 0 && d3 != 0 && d4 != 0 {
        return true; // proper crossing
    }
    if d1 == 0 && onseg(cx, cy, dx, dy, ax, ay) {
        return true;
    }
    if d2 == 0 && onseg(cx, cy, dx, dy, bx, by) {
        return true;
    }
    if d3 == 0 && onseg(ax, ay, bx, by, cx, cy) {
        return true;
    }
    if d4 == 0 && onseg(ax, ay, bx, by, dx, dy) {
        return true;
    }
    false
}

// ---------------------------------------------------------------------------
// Geometry registry (mirrors geoexact's Geom/Ring)
// ---------------------------------------------------------------------------

pub(crate) struct Ring {
    pub(crate) start: usize,
    pub(crate) npts: usize, // INCLUDING the closing duplicate
    pub(crate) ccw: i32,    // shoelace: +1 CCW, -1 CW, 0 degenerate
    pub(crate) bb: [f64; 4],
}

pub(crate) struct Geom {
    pub(crate) xy: Vec<f64>,
    pub(crate) rings: Vec<Ring>,
    pub(crate) testvtx: Vec<usize>, // one exterior-ring vertex per part
    pub(crate) bb: [f64; 4],
}

pub struct Handle {
    geoms: Vec<Geom>,
}

// Even-odd membership, touch-inclusive engine variant: exact for points
// not on the boundary (callers resolve boundary contact by seg tests).
pub(crate) fn point_in(g: &Geom, gx: f64, gy: f64, px: f64, py: f64) -> bool {
    let mut inside = false;
    for r in &g.rings {
        if py < r.bb[1] + gy || py > r.bb[3] + gy || px > r.bb[2] + gx {
            continue;
        }
        let p = &g.xy[2 * r.start..];
        let nseg = r.npts - 1;
        for i in 0..nseg {
            let x1 = p[2 * i] + gx;
            let y1 = p[2 * i + 1] + gy;
            let x2 = p[2 * i + 2] + gx;
            let y2 = p[2 * i + 3] + gy;
            if (y1 > py) != (y2 > py) {
                let o = osign(x1, y1, x2, y2, px, py);
                if o == 0 {
                    continue;
                }
                if if y2 > y1 { o > 0 } else { o < 0 } {
                    inside = !inside;
                }
            }
        }
    }
    inside
}

// Strict membership: +1 strictly inside, 0 on the boundary, -1 outside.
fn point_in_strict(g: &Geom, gx: f64, gy: f64, px: f64, py: f64) -> i32 {
    if px < g.bb[0] + gx || px > g.bb[2] + gx
        || py < g.bb[1] + gy || py > g.bb[3] + gy {
        return -1;
    }
    let mut inside = false;
    for r in &g.rings {
        let p = &g.xy[2 * r.start..];
        let nseg = r.npts - 1;
        let span = !(py < r.bb[1] + gy || py > r.bb[3] + gy
                     || px > r.bb[2] + gx);
        for i in 0..nseg {
            let x1 = p[2 * i] + gx;
            let y1 = p[2 * i + 1] + gy;
            let x2 = p[2 * i + 2] + gx;
            let y2 = p[2 * i + 3] + gy;
            if onseg(x1, y1, x2, y2, px, py)
                && osign_overlap(x1, y1, x2, y2, px, py) == 0 {
                return 0;
            }
            if span && (y1 > py) != (y2 > py) {
                let o = osign_overlap(x1, y1, x2, y2, px, py);
                if o == 0 {
                    continue;
                }
                if if y2 > y1 { o > 0 } else { o < 0 } {
                    inside = !inside;
                }
            }
        }
    }
    if inside { 1 } else { -1 }
}

// Exact touch-inclusive intersects of A at (ax, ay) and B at (bx, by).
pub(crate) fn geom_intersects(a: &Geom, ax: f64, ay: f64,
                   b: &Geom, bx: f64, by: f64) -> bool {
    if a.bb[2] + ax < b.bb[0] + bx || b.bb[2] + bx < a.bb[0] + ax
        || a.bb[3] + ay < b.bb[1] + by || b.bb[3] + by < a.bb[1] + ay {
        return false;
    }
    for ra in &a.rings {
        let a0 = ra.bb[0] + ax;
        let a1 = ra.bb[1] + ay;
        let a2 = ra.bb[2] + ax;
        let a3 = ra.bb[3] + ay;
        for rb in &b.rings {
            if a2 < rb.bb[0] + bx || rb.bb[2] + bx < a0
                || a3 < rb.bb[1] + by || rb.bb[3] + by < a1 {
                continue;
            }
            let pa = &a.xy[2 * ra.start..];
            let pb = &b.xy[2 * rb.start..];
            let na = ra.npts - 1;
            let nb = rb.npts - 1;
            for i in 0..na {
                let x1 = pa[2 * i] + ax;
                let y1 = pa[2 * i + 1] + ay;
                let x2 = pa[2 * i + 2] + ax;
                let y2 = pa[2 * i + 3] + ay;
                let slo = if x1 < x2 { x1 } else { x2 };
                let shi = if x1 < x2 { x2 } else { x1 };
                let tlo = if y1 < y2 { y1 } else { y2 };
                let thi = if y1 < y2 { y2 } else { y1 };
                if shi < rb.bb[0] + bx || rb.bb[2] + bx < slo
                    || thi < rb.bb[1] + by || rb.bb[3] + by < tlo {
                    continue;
                }
                for j in 0..nb {
                    let x3 = pb[2 * j] + bx;
                    let y3 = pb[2 * j + 1] + by;
                    let x4 = pb[2 * j + 2] + bx;
                    let y4 = pb[2 * j + 3] + by;
                    let ulo = if x3 < x4 { x3 } else { x4 };
                    let uhi = if x3 < x4 { x4 } else { x3 };
                    if uhi < slo || shi < ulo {
                        continue;
                    }
                    let vlo = if y3 < y4 { y3 } else { y4 };
                    let vhi = if y3 < y4 { y4 } else { y3 };
                    if vhi < tlo || thi < vlo {
                        continue;
                    }
                    if seg_intersects(x1, y1, x2, y2, x3, y3, x4, y4) {
                        return true;
                    }
                }
            }
        }
    }
    for &t in &b.testvtx {
        if point_in(a, ax, ay, b.xy[2 * t] + bx, b.xy[2 * t + 1] + by) {
            return true;
        }
    }
    for &t in &a.testvtx {
        if point_in(b, bx, by, a.xy[2 * t] + ax, a.xy[2 * t + 1] + ay) {
            return true;
        }
    }
    false
}

// Exact positive-area interior intersection (touch-exclusive) -- port of
// geoexact geom_overlap, same certificate set: proper crossing, collinear
// overlap with same interior side, vertex / edge-midpoint strictly inside.
pub(crate) fn geom_overlap(a: &Geom, ax: f64, ay: f64,
                b: &Geom, bx: f64, by: f64) -> bool {
    if a.bb[2] + ax <= b.bb[0] + bx || b.bb[2] + bx <= a.bb[0] + ax
        || a.bb[3] + ay <= b.bb[1] + by || b.bb[3] + by <= a.bb[1] + ay {
        return false;
    }
    for ra in &a.rings {
        let a0 = ra.bb[0] + ax;
        let a1 = ra.bb[1] + ay;
        let a2 = ra.bb[2] + ax;
        let a3 = ra.bb[3] + ay;
        for rb in &b.rings {
            if a2 < rb.bb[0] + bx || rb.bb[2] + bx < a0
                || a3 < rb.bb[1] + by || rb.bb[3] + by < a1 {
                continue;
            }
            let pa = &a.xy[2 * ra.start..];
            let pb = &b.xy[2 * rb.start..];
            let na = ra.npts - 1;
            let nb = rb.npts - 1;
            for i in 0..na {
                let x1 = pa[2 * i] + ax;
                let y1 = pa[2 * i + 1] + ay;
                let x2 = pa[2 * i + 2] + ax;
                let y2 = pa[2 * i + 3] + ay;
                let slo = if x1 < x2 { x1 } else { x2 };
                let shi = if x1 < x2 { x2 } else { x1 };
                let tlo = if y1 < y2 { y1 } else { y2 };
                let thi = if y1 < y2 { y2 } else { y1 };
                if shi < rb.bb[0] + bx || rb.bb[2] + bx < slo
                    || thi < rb.bb[1] + by || rb.bb[3] + by < tlo {
                    continue;
                }
                for j in 0..nb {
                    let x3 = pb[2 * j] + bx;
                    let y3 = pb[2 * j + 1] + by;
                    let x4 = pb[2 * j + 2] + bx;
                    let y4 = pb[2 * j + 3] + by;
                    let ulo = if x3 < x4 { x3 } else { x4 };
                    let uhi = if x3 < x4 { x4 } else { x3 };
                    if uhi < slo || shi < ulo {
                        continue;
                    }
                    let vlo = if y3 < y4 { y3 } else { y4 };
                    let vhi = if y3 < y4 { y4 } else { y3 };
                    if vhi < tlo || thi < vlo {
                        continue;
                    }
                    let o1 = osign_overlap(x1, y1, x2, y2, x3, y3);
                    let o2 = osign_overlap(x1, y1, x2, y2, x4, y4);
                    let o3 = osign_overlap(x3, y3, x4, y4, x1, y1);
                    let o4 = osign_overlap(x3, y3, x4, y4, x2, y2);
                    if o1 != 0 && o2 != 0 && o1 != o2
                        && o3 != 0 && o4 != 0 && o3 != o4 {
                        return true; // proper crossing
                    }
                    if o1 == 0 && o2 == 0 {
                        let xdom = (shi - slo) >= (thi - tlo);
                        let lo = if xdom {
                            if slo > ulo { slo } else { ulo }
                        } else if tlo > vlo { tlo } else { vlo };
                        let hi = if xdom {
                            if shi < uhi { shi } else { uhi }
                        } else if thi < vhi { thi } else { vhi };
                        if hi > lo && ra.ccw != 0 && rb.ccw != 0 {
                            let dot = (x2 - x1) * (x4 - x3)
                                + (y2 - y1) * (y4 - y3);
                            let s = (dot > 0.0) as i32 - (dot < 0.0) as i32;
                            if s != 0 && ra.ccw == rb.ccw * s {
                                return true; // interiors same side
                            }
                        }
                    }
                }
            }
        }
    }
    for ra in &a.rings {
        let pa = &a.xy[2 * ra.start..];
        let na = ra.npts - 1;
        for i in 0..na {
            let x1 = pa[2 * i] + ax;
            let y1 = pa[2 * i + 1] + ay;
            if point_in_strict(b, bx, by, x1, y1) > 0 {
                return true;
            }
            let x2 = pa[2 * i + 2] + ax;
            let y2 = pa[2 * i + 3] + ay;
            if point_in_strict(b, bx, by, 0.5 * (x1 + x2), 0.5 * (y1 + y2)) > 0 {
                return true;
            }
        }
    }
    for rb in &b.rings {
        let pb = &b.xy[2 * rb.start..];
        let nb = rb.npts - 1;
        for j in 0..nb {
            let x3 = pb[2 * j] + bx;
            let y3 = pb[2 * j + 1] + by;
            if point_in_strict(a, ax, ay, x3, y3) > 0 {
                return true;
            }
            let x4 = pb[2 * j + 2] + bx;
            let y4 = pb[2 * j + 3] + by;
            if point_in_strict(a, ax, ay, 0.5 * (x3 + x4), 0.5 * (y3 + y4)) > 0 {
                return true;
            }
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Rasterization (stage C): mask / amask cell verdicts on the half-unit
// lattice.  The union geometry itself is still CONSTRUCTED by shapely
// (seam vertices are GEOS products); this code only replaces the per-cell
// predicates -- exact touch-inclusive intersects for the conservative
// mask, exact contains_properly (cell strictly inside the interior) for
// the aggressive mask -- so verdicts are bit-identical to
// shapely.intersects / shapely.contains_properly on the same geometry.
// ---------------------------------------------------------------------------

// Closed segment-set of the geometry boundary meets the closed axis box?
pub(crate) fn box_contact(g: &Geom, x0: f64, y0: f64, x1: f64, y1: f64) -> bool {
    for r in &g.rings {
        if r.bb[0] > x1 || r.bb[2] < x0 || r.bb[1] > y1 || r.bb[3] < y0 {
            continue; // strictly separated (touch survives the filter)
        }
        let p = &g.xy[2 * r.start..];
        let n = r.npts - 1;
        for i in 0..n {
            let ax = p[2 * i];
            let ay = p[2 * i + 1];
            let bx = p[2 * i + 2];
            let by = p[2 * i + 3];
            let slo = if ax < bx { ax } else { bx };
            let shi = if ax < bx { bx } else { ax };
            if slo > x1 || shi < x0 {
                continue;
            }
            let tlo = if ay < by { ay } else { by };
            let thi = if ay < by { by } else { ay };
            if tlo > y1 || thi < y0 {
                continue;
            }
            if (ax >= x0 && ax <= x1 && ay >= y0 && ay <= y1)
                || (bx >= x0 && bx <= x1 && by >= y0 && by <= y1) {
                return true; // an endpoint lies in the closed box
            }
            // both endpoints outside: contact requires crossing an edge
            if seg_intersects(ax, ay, bx, by, x0, y0, x1, y0)
                || seg_intersects(ax, ay, bx, by, x1, y0, x1, y1)
                || seg_intersects(ax, ay, bx, by, x1, y1, x0, y1)
                || seg_intersects(ax, ay, bx, by, x0, y1, x0, y0) {
                return true;
            }
        }
    }
    false
}

/// Fill mask/amask (row-major h x w, cell (r, c) covers
/// [(ox+c)/res, (ox+c+1)/res) x [(oy+r)/res, ...)) for the registered
/// geometry `gid` in its LOCAL frame -- mirrors constructor._rasterize.
/// # Safety: h from rc_new; mask/amask point to h*w writable bytes.
#[no_mangle]
pub unsafe extern "C" fn rc_rasterize(h: *mut Handle, gid: i32,
                                      ox: i64, oy: i64, w: i32, hgt: i32,
                                      res: i32, mask: *mut u8,
                                      amask: *mut u8) {
    let hd = &*h;
    let g = &hd.geoms[gid as usize];
    let cs = 1.0 / res as f64;
    let w = w as usize;
    let hgt = hgt as usize;
    let mask = std::slice::from_raw_parts_mut(mask, hgt * w);
    let amask = std::slice::from_raw_parts_mut(amask, hgt * w);
    for r in 0..hgt {
        let y0 = (oy + r as i64) as f64 * cs;
        let y1 = y0 + cs;
        for c in 0..w {
            let x0 = (ox + c as i64) as f64 * cs;
            let x1 = x0 + cs;
            let idx = r * w + c;
            if g.bb[0] > x1 || g.bb[2] < x0 || g.bb[1] > y1 || g.bb[3] < y0 {
                mask[idx] = 0;
                amask[idx] = 0;
                continue;
            }
            if box_contact(g, x0, y0, x1, y1) {
                // boundary meets the closed cell: intersects, but the
                // cell cannot lie strictly inside the interior
                mask[idx] = 1;
                amask[idx] = 0;
            } else {
                // no boundary contact: the whole cell sits in one
                // region -- one corner decides (it cannot be on the
                // boundary here, so even-odd membership is exact)
                let inside = point_in(g, 0.0, 0.0, x0, y0);
                mask[idx] = inside as u8;
                amask[idx] = inside as u8;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Phase-A kernels (stage D): the mask-tier cascade and the position
// scorer, ports of constructor._k_tiers / _k_score (numba).  The numba
// originals emit a canonical CSR (ascending position, ascending active
// index) and document exact float evaluation order -- this serial port
// reproduces both bit for bit.
// ---------------------------------------------------------------------------

/// Mask-tier cascade over one candidate grid (positions y-major).
/// counters receives [n_mask, ncf, nrs].
/// # Safety: array pointers must match the stated dimensions; mask cell
/// indices are in-range by construction (same contract as the numba
/// kernel, which runs without bounds checks).
#[no_mangle]
pub unsafe extern "C" fn rc_k_tiers(
    occ: *const u8, agg: *const u8, a_dim: i64, mh: i64, mw: i64,
    bbox: *const f64, xs: *const i64, nx: i64, ys: *const i64, ny: i64,
    res: i64, gox: i64, goy: i64,
    ci: *const i64, cj: *const i64, nc: i64,
    ai: *const i64, aj: *const i64, na: i64,
    lx0: f64, ly0: f64, lx1: f64, ly1: f64,
    conf_ptr: *mut i32, conf_flat: *mut i32,
    res_ptr: *mut i32, res_flat: *mut i32,
    counters: *mut i64,
) {
    let a_dim = a_dim as usize;
    let mh = mh as usize;
    let mw = mw as usize;
    let nx = nx as usize;
    let ny = ny as usize;
    let p_total = nx * ny;
    let occ = std::slice::from_raw_parts(occ, a_dim * mh * mw);
    let agg = std::slice::from_raw_parts(agg, a_dim * mh * mw);
    let bbox = std::slice::from_raw_parts(bbox, a_dim * 4);
    let xs = std::slice::from_raw_parts(xs, nx);
    let ys = std::slice::from_raw_parts(ys, ny);
    let ci = std::slice::from_raw_parts(ci, nc as usize);
    let cj = std::slice::from_raw_parts(cj, nc as usize);
    let ai = std::slice::from_raw_parts(ai, na as usize);
    let aj = std::slice::from_raw_parts(aj, na as usize);
    let conf_ptr = std::slice::from_raw_parts_mut(conf_ptr, p_total + 1);
    let res_ptr = std::slice::from_raw_parts_mut(res_ptr, p_total + 1);

    let mut n_mask: i64 = 0;
    let mut ncf: usize = 0;
    let mut nrs: usize = 0;
    conf_ptr[0] = 0;
    res_ptr[0] = 0;
    for p in 0..p_total {
        let y = ys[p / nx];
        let x = xs[p % nx];
        let r0 = y * res + goy;
        let c0 = x * res + gox;
        let wy0 = y as f64 + ly0;
        let wy1 = y as f64 + ly1;
        let wx0 = x as f64 + lx0;
        let wx1 = x as f64 + lx1;
        let mut seen = false;
        for a in 0..a_dim {
            if !(wx0 < bbox[4 * a + 2] && bbox[4 * a] < wx1
                 && wy0 < bbox[4 * a + 3] && bbox[4 * a + 1] < wy1) {
                continue;
            }
            seen = true;
            let base = a * mh * mw;
            let mut hit = false;
            for t in 0..nc as usize {
                let idx = base + (r0 + ci[t]) as usize * mw
                    + (c0 + cj[t]) as usize;
                if *occ.get_unchecked(idx) != 0 {
                    hit = true;
                    break;
                }
            }
            if !hit {
                continue;
            }
            let mut ahit = false;
            for t in 0..na as usize {
                let idx = base + (r0 + ai[t]) as usize * mw
                    + (c0 + aj[t]) as usize;
                if *agg.get_unchecked(idx) != 0 {
                    ahit = true;
                    break;
                }
            }
            if ahit {
                *conf_flat.add(ncf) = a as i32;
                ncf += 1;
            } else {
                *res_flat.add(nrs) = a as i32;
                nrs += 1;
            }
        }
        conf_ptr[p + 1] = ncf as i32;
        res_ptr[p + 1] = nrs as i32;
        if seen {
            n_mask += 1;
        }
    }
    *counters = n_mask;
    *counters.add(1) = ncf as i64;
    *counters.add(2) = nrs as i64;
}

/// Conservative entry + objective-shaped score per position; optimistic
/// nest-track score when enabled.  Same float evaluation order as the
/// numba kernel.
/// # Safety: pointer/dimension contract as in rc_k_tiers.
#[no_mangle]
pub unsafe extern "C" fn rc_k_score(
    conf_ptr: *const i32, conf_flat: *const i32,
    res_ptr: *const i32, res_flat: *const i32, res_ok: *const u8,
    act_s: *const i64, act_f: *const i64,
    xs: *const i64, nx: i64, ys: *const i64, ny: i64,
    r_b: i64, p_b: i64, d_b: i64,
    base_score: f64, w1: f64, ly1: f64, nest: i32, tgt: i64, ptw: f64,
    scores: *mut f64, scores_opt: *mut f64,
) {
    let nx = nx as usize;
    let ny = ny as usize;
    let p_total = nx * ny;
    let ys = std::slice::from_raw_parts(ys, ny);
    let _ = std::slice::from_raw_parts(xs, nx);
    let conf_ptr = std::slice::from_raw_parts(conf_ptr, p_total + 1);
    let res_ptr = std::slice::from_raw_parts(res_ptr, p_total + 1);
    let scores = std::slice::from_raw_parts_mut(scores, p_total);
    let scores_opt = std::slice::from_raw_parts_mut(scores_opt, p_total);

    let mut keys: Vec<i64> = Vec::new();
    let mut hard: Vec<i64> = Vec::new();
    for p in 0..p_total {
        let y = ys[p / nx];
        let c0 = conf_ptr[p];
        let c1 = conf_ptr[p + 1];
        let r0 = res_ptr[p];
        let r1 = res_ptr[p + 1];
        keys.clear();
        for i in c0..c1 {
            let a = *conf_flat.add(i as usize) as usize;
            keys.push((*act_s.add(a) << 20) | *act_f.add(a));
        }
        for i in r0..r1 {
            if *res_ok.add(i as usize) != 0 {
                let a = *res_flat.add(i as usize) as usize;
                keys.push((*act_s.add(a) << 20) | *act_f.add(a));
            }
        }
        let n_iv = keys.len();
        // insertion sort == sorted() on (S, F) pairs (F < 2^20)
        for i in 1..n_iv {
            let k = keys[i];
            let mut j = i;
            while j > 0 && keys[j - 1] > k {
                keys[j] = keys[j - 1];
                j -= 1;
            }
            keys[j] = k;
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
        let mut tard = entry + p_b - d_b;
        if tard < 0 {
            tard = 0;
        }
        let mut dtg = entry - tgt;
        if dtg < 0 {
            dtg = -dtg;
        }
        let sc = base_score + w1 * tard as f64
            + 1e-4 * (y as f64 + ly1) + 1e-6 * entry as f64
            + ptw * dtg as f64;
        let mut sco = sc;
        if nest != 0 && n_iv > 0 {
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
                let mut t_opt = t + p_b - d_b;
                if t_opt < 0 {
                    t_opt = 0;
                }
                let mut dtg = t - tgt;
                if dtg < 0 {
                    dtg = -dtg;
                }
                sco = base_score + w1 * t_opt as f64
                    + 1e-4 * (y as f64 + ly1) + 1e-6 * t as f64
                    + ptw * dtg as f64;
            }
        }
        scores[p] = sc;
        scores_opt[p] = sco;
    }
}

// ---------------------------------------------------------------------------
// C ABI
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn rc_new() -> *mut Handle {
    Box::into_raw(Box::new(Handle { geoms: Vec::new() }))
}

/// # Safety: h must be a pointer returned by rc_new.
#[no_mangle]
pub unsafe extern "C" fn rc_del(h: *mut Handle) {
    drop(Box::from_raw(h));
}

/// Shared geometry builder (rc_add and the engine FFI): xy = npts (x, y)
/// pairs, ring_ptr = nrings+1 point offsets, test_idx = one exterior-ring
/// vertex index per part.
/// # Safety: pointers must reference arrays of the stated lengths.
pub(crate) unsafe fn build_geom(xy: *const f64, npts: i32,
                                ring_ptr: *const i32, nrings: i32,
                                test_idx: *const i32, ntest: i32) -> Geom {
    let xy = std::slice::from_raw_parts(xy, 2 * npts as usize);
    let ring_ptr = std::slice::from_raw_parts(ring_ptr, nrings as usize + 1);
    let test_idx = std::slice::from_raw_parts(test_idx, ntest as usize);

    let mut g = Geom {
        xy: xy.to_vec(),
        rings: Vec::with_capacity(nrings as usize),
        testvtx: test_idx.iter().map(|&t| t as usize).collect(),
        bb: [1e300, 1e300, -1e300, -1e300],
    };
    for r in 0..nrings as usize {
        let start = ring_ptr[r] as usize;
        let end = ring_ptr[r + 1] as usize;
        let mut bb = [1e300, 1e300, -1e300, -1e300];
        for i in start..end {
            let x = g.xy[2 * i];
            let y = g.xy[2 * i + 1];
            if x < bb[0] {
                bb[0] = x;
            }
            if y < bb[1] {
                bb[1] = y;
            }
            if x > bb[2] {
                bb[2] = x;
            }
            if y > bb[3] {
                bb[3] = y;
            }
        }
        for k in 0..2 {
            if bb[k] < g.bb[k] {
                g.bb[k] = bb[k];
            }
        }
        for k in 2..4 {
            if bb[k] > g.bb[k] {
                g.bb[k] = bb[k];
            }
        }
        let mut a2 = 0.0f64; // shoelace, orientation only
        for i in start..end.saturating_sub(1) {
            a2 += g.xy[2 * i] * g.xy[2 * i + 3]
                - g.xy[2 * i + 2] * g.xy[2 * i + 1];
        }
        let ccw = (a2 > 0.0) as i32 - (a2 < 0.0) as i32;
        g.rings.push(Ring { start, npts: end - start, ccw, bb });
    }
    g
}

/// Register a geometry into the handle.
/// # Safety: as build_geom.
#[no_mangle]
pub unsafe extern "C" fn rc_add(h: *mut Handle, xy: *const f64, npts: i32,
                                ring_ptr: *const i32, nrings: i32,
                                test_idx: *const i32, ntest: i32) -> i32 {
    let h = &mut *h;
    h.geoms.push(build_geom(xy, npts, ring_ptr, nrings, test_idx, ntest));
    h.geoms.len() as i32 - 1
}

/// # Safety: h from rc_new; ga/gb are gids returned by rc_add.
#[no_mangle]
pub unsafe extern "C" fn rc_isect_raw(h: *mut Handle, ga: i32, ax: f64, ay: f64,
                                      gb: i32, bx: f64, by: f64) -> i32 {
    let h = &*h;
    geom_intersects(&h.geoms[ga as usize], ax, ay,
                    &h.geoms[gb as usize], bx, by) as i32
}

/// # Safety: h from rc_new; ga/gb are gids returned by rc_add.
#[no_mangle]
pub unsafe extern "C" fn rc_overlap_raw(h: *mut Handle, ga: i32, ax: f64,
                                        ay: f64, gb: i32, bx: f64, by: f64)
                                        -> i32 {
    let h = &*h;
    geom_overlap(&h.geoms[ga as usize], ax, ay,
                 &h.geoms[gb as usize], bx, by) as i32
}

/// Exposed for direct predicate parity fuzzing.
#[no_mangle]
pub extern "C" fn rc_orient2d(ax: f64, ay: f64, bx: f64, by: f64,
                              cx: f64, cy: f64) -> f64 {
    orient2d(ax, ay, bx, by, cx, cy)
}

// ---------------------------------------------------------------------------
// Engine FFI (stage E): the construct() insert loop
// ---------------------------------------------------------------------------

/// # Safety: bay arrays hold m entries.
#[no_mangle]
pub unsafe extern "C" fn rc_eng_new(res: i64, bay_w: *const i64,
                                    bay_h: *const i64, m: i32)
                                    -> *mut engine::Engine {
    let w = std::slice::from_raw_parts(bay_w, m as usize).to_vec();
    let h = std::slice::from_raw_parts(bay_h, m as usize).to_vec();
    Box::into_raw(Box::new(engine::Engine::new(res, w, h)))
}

/// # Safety: e from rc_eng_new.
#[no_mangle]
pub unsafe extern "C" fn rc_eng_del(e: *mut engine::Engine) {
    drop(Box::from_raw(e));
}

/// Register the (bid, oi) union (shapely-constructed rings) plus the RAW
/// layer-vertex bbox; masks and cells are derived in the engine.
/// # Safety: as build_geom; bbox holds 4 doubles.
#[no_mangle]
pub unsafe extern "C" fn rc_eng_add_geo(e: *mut engine::Engine, bid: i32,
                                        oi: i32, has_union: i32,
                                        xy: *const f64, npts: i32,
                                        ring_ptr: *const i32, nrings: i32,
                                        test_idx: *const i32, ntest: i32,
                                        bbox: *const f64) -> i32 {
    let e = &mut *e;
    let u = if has_union != 0 {
        Some(build_geom(xy, npts, ring_ptr, nrings, test_idx, ntest))
    } else {
        None
    };
    let bb = std::slice::from_raw_parts(bbox, 4);
    e.add_geo(bid, oi, u, [bb[0], bb[1], bb[2], bb[3]]) as i32
}

/// Append one layer (CLOSED ring, npts including the closing duplicate;
/// npts == 0 marks a degenerate layer).  Call in layer order per (bid, oi).
/// # Safety: xy holds 2*npts doubles when npts > 0.
#[no_mangle]
pub unsafe extern "C" fn rc_eng_add_layer(e: *mut engine::Engine, bid: i32,
                                          oi: i32, xy: *const f64,
                                          npts: i32) {
    let e = &mut *e;
    let g = if npts >= 4 {
        let rp = [0i32, npts];
        let tv = [0i32];
        Some(build_geom(xy, npts, rp.as_ptr(), 1, tv.as_ptr(), 1))
    } else {
        None
    };
    e.add_layer(bid, oi, g);
}

/// Run the insert loop; out receives n*6 i64 (bay, x, y, oi, entry, exit).
/// Returns 0, or -(bid+1) when a block admits no placement.
/// # Safety: array pointers match the stated dimensions; oi CSR covers n+1
/// entries in oi_ptr.
#[no_mangle]
pub unsafe extern "C" fn rc_eng_run(
    e: *mut engine::Engine, n: i32,
    r: *const i64, d: *const i64, p: *const i64, wl: *const f64,
    prefs: *const f64,
    bay_prior_active: i32, bay_prior_strength: f64,
    bay_prior: *const f64, n_orient: *const i32,
    oi_ptr: *const i32, oi_flat: *const i32,
    order: *const i32, order_len: i32,
    nf: i32, fixed: *const i64,
    plan_active: i32, plan_bays: i32, plan_target_w: f64,
    plan_start: *const i64, plan_bay: *const i32,
    max_xs: i32, max_ys: i32, top_b: i32, nest_lane: i32, slide: i32,
    w1: f64, w2: f64, w3: f64, timelimit: f64,
    out: *mut i64) -> i32 {
    let e = &mut *e;
    let n = n as usize;
    let m = e.bay_w.len();
    let r = std::slice::from_raw_parts(r, n);
    let d = std::slice::from_raw_parts(d, n);
    let p = std::slice::from_raw_parts(p, n);
    let wl = std::slice::from_raw_parts(wl, n);
    let prefs = std::slice::from_raw_parts(prefs, n * m);
    let bay_prior = if bay_prior_active != 0 {
        std::slice::from_raw_parts(bay_prior, n * m)
    } else {
        &[]
    };
    let n_orient = std::slice::from_raw_parts(n_orient, n);
    let oi_ptr = std::slice::from_raw_parts(oi_ptr, n + 1);
    let oi_flat =
        std::slice::from_raw_parts(oi_flat, oi_ptr[n] as usize);
    let order = std::slice::from_raw_parts(order, order_len as usize);
    let mut blocks = Vec::with_capacity(n);
    for i in 0..n {
        blocks.push(engine::BlockMeta {
            r: r[i],
            d: d[i],
            p: p[i],
            wl: wl[i],
            prefs: prefs[i * m..(i + 1) * m].to_vec(),
            oi_uniq: oi_flat[oi_ptr[i] as usize..oi_ptr[i + 1] as usize]
                .to_vec(),
            n_orient: n_orient[i] as usize,
        });
    }
    let fixed_flat = std::slice::from_raw_parts(fixed, nf as usize * 7);
    let fixed_arr: Vec<[i64; 7]> = (0..nf as usize)
        .map(|i| {
            let s = &fixed_flat[7 * i..7 * i + 7];
            [s[0], s[1], s[2], s[3], s[4], s[5], s[6]]
        })
        .collect();
    let plan_start_s = std::slice::from_raw_parts(plan_start, n);
    let plan_bay_s = std::slice::from_raw_parts(plan_bay, n);
    let plan = if plan_active != 0 {
        Some(engine::PlanIn {
            start: plan_start_s,
            bay: plan_bay_s,
            target_w: plan_target_w,
            bays: plan_bays != 0,
        })
    } else {
        None
    };
    let prm = engine::Params {
        max_xs: max_xs as usize,
        max_ys: max_ys as usize,
        top_b: top_b as usize,
        nest_lane: nest_lane as usize,
        slide: slide != 0,
        w1,
        w2,
        w3,
        bay_prior: bay_prior.to_vec(),
        bay_prior_strength,
        bay_prior_active: bay_prior_active != 0,
        timelimit,
    };
    let out = std::slice::from_raw_parts_mut(out, n * 6);
    engine::run(e, &blocks, order, &fixed_arr, plan, &prm, out)
}

// ---------------------------------------------------------------------------
// seatrep window enumeration (win.rs): candidates + conflicts + pairs
// for the windowed exact re-assignment.  Frozen and window blocks are
// identified by (bid, oi) against the engine registry.
// ---------------------------------------------------------------------------

/// # Safety: arrays match the stated lengths; every (bid, oi) referenced
/// must be registered via rc_eng_add_geo/add_layer.
#[no_mangle]
pub unsafe extern "C" fn rc_win_enum(
    e: *mut engine::Engine,
    nf: i32, fz_bid: *const i32, fz_oi: *const i32,
    fz_x: *const i64, fz_y: *const i64, fz_bay: *const i32,
    fz_s: *const i64, fz_f: *const i64,
    nw: i32, wn_bid: *const i32, wn_r: *const i64, wn_p: *const i64,
    wn_d: *const i64, wn_is_off: *const u8, wn_cur_bay: *const i32,
    wn_cur_x: *const i64, wn_cur_y: *const i64, wn_cur_oi: *const i32,
    wn_allowed_ptr: *const i32, wn_allowed_flat: *const i32,
    grid_nx: i32, grid_ny: i32, cand_m: i32, force_off: i32,
) -> *mut win::WinRes {
    let e = &mut *e;
    let nf = nf as usize;
    let nw = nw as usize;
    let fz_bid = std::slice::from_raw_parts(fz_bid, nf);
    let fz_oi = std::slice::from_raw_parts(fz_oi, nf);
    let fz_gk: Vec<usize> = (0..nf)
        .map(|i| e.gidx[&(fz_bid[i], fz_oi[i])])
        .collect();
    let wn_allowed_ptr = std::slice::from_raw_parts(wn_allowed_ptr, nw + 1);
    let n_allow = wn_allowed_ptr[nw] as usize;
    let res = win::win_enum(
        e,
        &fz_gk,
        std::slice::from_raw_parts(fz_x, nf),
        std::slice::from_raw_parts(fz_y, nf),
        std::slice::from_raw_parts(fz_bay, nf),
        std::slice::from_raw_parts(fz_s, nf),
        std::slice::from_raw_parts(fz_f, nf),
        std::slice::from_raw_parts(wn_bid, nw),
        std::slice::from_raw_parts(wn_r, nw),
        std::slice::from_raw_parts(wn_p, nw),
        std::slice::from_raw_parts(wn_d, nw),
        std::slice::from_raw_parts(wn_is_off, nw),
        std::slice::from_raw_parts(wn_cur_bay, nw),
        std::slice::from_raw_parts(wn_cur_x, nw),
        std::slice::from_raw_parts(wn_cur_y, nw),
        std::slice::from_raw_parts(wn_cur_oi, nw),
        wn_allowed_ptr,
        std::slice::from_raw_parts(wn_allowed_flat, n_allow),
        grid_nx as usize, grid_ny as usize, cand_m as usize,
        force_off != 0,
    );
    Box::into_raw(Box::new(res))
}

/// # Safety: res from rc_win_enum; out holds 3 i64.
#[no_mangle]
pub unsafe extern "C" fn rc_win_counts(res: *mut win::WinRes,
                                       out: *mut i64) {
    let r = &*res;
    let out = std::slice::from_raw_parts_mut(out, 3);
    out[0] = r.cands.len() as i64;
    out[1] = r.confs.len() as i64;
    out[2] = r.pairs.len() as i64;
}

/// # Safety: res from rc_win_enum; buffers hold n_cands*8, n_confs*3,
/// n_pairs*5 i64 respectively (counts from rc_win_counts).
#[no_mangle]
pub unsafe extern "C" fn rc_win_fetch(res: *mut win::WinRes,
                                      cands: *mut i64, confs: *mut i64,
                                      pairs: *mut i64) {
    let r = &*res;
    let c = std::slice::from_raw_parts_mut(cands, r.cands.len() * 8);
    for (i, row) in r.cands.iter().enumerate() {
        c[i * 8..i * 8 + 8].copy_from_slice(row);
    }
    let f = std::slice::from_raw_parts_mut(confs, r.confs.len() * 3);
    for (i, row) in r.confs.iter().enumerate() {
        f[i * 3..i * 3 + 3].copy_from_slice(row);
    }
    let p = std::slice::from_raw_parts_mut(pairs, r.pairs.len() * 5);
    for (i, row) in r.pairs.iter().enumerate() {
        p[i * 5..i * 5 + 5].copy_from_slice(row);
    }
}

/// # Safety: res from rc_win_enum, freed exactly once.
#[no_mangle]
pub unsafe extern "C" fn rc_win_del(res: *mut win::WinRes) {
    drop(Box::from_raw(res));
}
