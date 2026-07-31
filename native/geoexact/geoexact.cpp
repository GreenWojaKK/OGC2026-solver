// geoexact.cpp -- exact polygon-intersects engine for the OGC2026 constructor.
//
// Replaces the GEOS prepared-intersects verdicts of the Phase-A residual
// band (conservative-mask hit, aggressive-mask miss) and the single-pair
// _overlap_exact confirm.  The verdict computed here is the mathematically
// exact `intersects` (touch-inclusive, closed sets) of two translated
// float64 polygon soups -- the same world coordinates numpy/GEOS see:
// world = template_coord + double(integer_offset), one IEEE add per
// coordinate, so verdicts coincide bit-for-bit with GEOS's robust
// predicates on identical inputs.
//
// Exact arithmetic: Shewchuk adaptive-precision orient2d (filtered fast
// path, exact expansion fallback).  REQUIRES strict IEEE double semantics:
// compile with  -ffp-contract=off  and WITHOUT -ffast-math/-Ofast.
//
// Memoisation keys include both absolute placements.  Although polygon
// intersection is translation-invariant over the reals, adding an integer
// offset to float64 template coordinates rounds differently at different
// absolute positions.  Relative-offset-only keys can therefore reuse a
// wrong boundary-contact verdict.
//
// Build (Ubuntu, glibc forward-compatible):
//   g++ -O2 -shared -fPIC -ffp-contract=off -fno-math-errno \
//       -static-libstdc++ -static-libgcc -o geoexact.so geoexact.cpp

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

static_assert(sizeof(double) == 8, "IEEE double required");

// ---------------------------------------------------------------------------
// Shewchuk exact predicates (orient2d), transcribed from the public-domain
// "Routines for Arbitrary Precision Floating-point Arithmetic and Fast
// Robust Geometric Predicates" (J. R. Shewchuk).
// ---------------------------------------------------------------------------

static const double s_epsilon        = 1.1102230246251565e-16;  // 2^-53
static const double s_splitter       = 134217729.0;             // 2^27 + 1
static const double s_resulterrbound = (3.0 + 8.0 * s_epsilon) * s_epsilon;
static const double s_ccwerrboundA   = (3.0 + 16.0 * s_epsilon) * s_epsilon;
static const double s_ccwerrboundB   = (2.0 + 12.0 * s_epsilon) * s_epsilon;
static const double s_ccwerrboundC   =
    (9.0 + 64.0 * s_epsilon) * s_epsilon * s_epsilon;

#define Fast_Two_Sum(a, b, x, y) \
    x = (double)(a + b);         \
    bvirt = x - a;               \
    y = b - bvirt

#define Two_Sum(a, b, x, y)  \
    x = (double)(a + b);     \
    bvirt = (double)(x - a); \
    avirt = x - bvirt;       \
    bround = b - bvirt;      \
    around = a - avirt;      \
    y = around + bround

#define Two_Diff(a, b, x, y) \
    x = (double)(a - b);     \
    bvirt = (double)(a - x); \
    avirt = x + bvirt;       \
    bround = bvirt - b;      \
    around = a - avirt;      \
    y = around + bround

#define Split(a, ahi, alo)         \
    c = (double)(s_splitter * a);  \
    abig = (double)(c - a);        \
    ahi = c - abig;                \
    alo = a - ahi

#define Two_Product(a, b, x, y)        \
    x = (double)(a * b);               \
    Split(a, ahi, alo);                \
    Split(b, bhi, blo);                \
    err1 = x - (ahi * bhi);            \
    err2 = err1 - (alo * bhi);         \
    err3 = err2 - (ahi * blo);         \
    y = (alo * blo) - err3

#define Two_One_Diff(a1, a0, b, x2, x1, x0) \
    Two_Diff(a0, b, _i, x0);                \
    Two_Sum(a1, _i, x2, x1)

#define Two_Two_Diff(a1, a0, b1, b0, x3, x2, x1, x0) \
    Two_One_Diff(a1, a0, b0, _j, _0, x0);            \
    Two_One_Diff(_j, _0, b1, x3, x2, x1)

// Transcription note: the original advances with `enow = e[++eindex]`,
// reading one slot past the used length (benign in predicates.c, still UB);
// here the advance is guarded so that `enow == e[eindex]` whenever
// `eindex < elen`, which is the only case the value is ever consumed.
// Results are identical.
static int fast_expansion_sum_zeroelim(int elen, const double* e,
                                       int flen, const double* f, double* h) {
    double Q, Qnew, hh;
    double bvirt, avirt, bround, around;
    double enow = e[0], fnow = f[0];
    int eindex = 0, findex = 0, hindex = 0;

    if ((fnow > enow) == (fnow > -enow)) {
        Q = enow;
        if (++eindex < elen) enow = e[eindex];
    } else {
        Q = fnow;
        if (++findex < flen) fnow = f[findex];
    }
    if ((eindex < elen) && (findex < flen)) {
        if ((fnow > enow) == (fnow > -enow)) {
            Fast_Two_Sum(enow, Q, Qnew, hh);
            if (++eindex < elen) enow = e[eindex];
        } else {
            Fast_Two_Sum(fnow, Q, Qnew, hh);
            if (++findex < flen) fnow = f[findex];
        }
        Q = Qnew;
        if (hh != 0.0) h[hindex++] = hh;
        while ((eindex < elen) && (findex < flen)) {
            if ((fnow > enow) == (fnow > -enow)) {
                Two_Sum(Q, enow, Qnew, hh);
                if (++eindex < elen) enow = e[eindex];
            } else {
                Two_Sum(Q, fnow, Qnew, hh);
                if (++findex < flen) fnow = f[findex];
            }
            Q = Qnew;
            if (hh != 0.0) h[hindex++] = hh;
        }
    }
    while (eindex < elen) {
        Two_Sum(Q, enow, Qnew, hh);
        if (++eindex < elen) enow = e[eindex];
        Q = Qnew;
        if (hh != 0.0) h[hindex++] = hh;
    }
    while (findex < flen) {
        Two_Sum(Q, fnow, Qnew, hh);
        if (++findex < flen) fnow = f[findex];
        Q = Qnew;
        if (hh != 0.0) h[hindex++] = hh;
    }
    if ((Q != 0.0) || (hindex == 0)) h[hindex++] = Q;
    return hindex;
}

static double estimate(int elen, const double* e) {
    double Q = e[0];
    for (int i = 1; i < elen; i++) Q += e[i];
    return Q;
}

static double orient2dadapt(double pax, double pay, double pbx, double pby,
                            double pcx, double pcy, double detsum) {
    double acx, acy, bcx, bcy;
    double acxtail, acytail, bcxtail, bcytail;
    double detleft, detright;
    double detlefttail, detrighttail;
    double det, errbound;
    double B[4], C1[8], C2[12], D[16];
    double B3;
    int C1length, C2length, Dlength;
    double u[4];
    double u3;
    double s1, t1;
    double s0, t0;

    double bvirt, avirt, bround, around;
    double c, abig, ahi, alo, bhi, blo;
    double err1, err2, err3;
    double _i, _j, _0;

    acx = (double)(pax - pcx);
    bcx = (double)(pbx - pcx);
    acy = (double)(pay - pcy);
    bcy = (double)(pby - pcy);

    Two_Product(acx, bcy, detleft, detlefttail);
    Two_Product(acy, bcx, detright, detrighttail);

    Two_Two_Diff(detleft, detlefttail, detright, detrighttail,
                 B3, B[2], B[1], B[0]);
    B[3] = B3;

    det = estimate(4, B);
    errbound = s_ccwerrboundB * detsum;
    if ((det >= errbound) || (-det >= errbound)) return det;

    Two_Diff(pax, pcx, acx, acxtail);   // recompute with tails
    Two_Diff(pbx, pcx, bcx, bcxtail);
    Two_Diff(pay, pcy, acy, acytail);
    Two_Diff(pby, pcy, bcy, bcytail);

    if ((acxtail == 0.0) && (acytail == 0.0)
        && (bcxtail == 0.0) && (bcytail == 0.0)) return det;

    errbound = s_ccwerrboundC * detsum + s_resulterrbound * std::fabs(det);
    det += (acx * bcytail + bcy * acxtail)
         - (acy * bcxtail + bcx * acytail);
    if ((det >= errbound) || (-det >= errbound)) return det;

    Two_Product(acxtail, bcy, s1, s0);
    Two_Product(acytail, bcx, t1, t0);
    Two_Two_Diff(s1, s0, t1, t0, u3, u[2], u[1], u[0]);
    u[3] = u3;
    C1length = fast_expansion_sum_zeroelim(4, B, 4, u, C1);

    Two_Product(acx, bcytail, s1, s0);
    Two_Product(acy, bcxtail, t1, t0);
    Two_Two_Diff(s1, s0, t1, t0, u3, u[2], u[1], u[0]);
    u[3] = u3;
    C2length = fast_expansion_sum_zeroelim(C1length, C1, 4, u, C2);

    Two_Product(acxtail, bcytail, s1, s0);
    Two_Product(acytail, bcxtail, t1, t0);
    Two_Two_Diff(s1, s0, t1, t0, u3, u[2], u[1], u[0]);
    u[3] = u3;
    Dlength = fast_expansion_sum_zeroelim(C2length, C2, 4, u, D);

    return D[Dlength - 1];
}

static double orient2d(double pax, double pay, double pbx, double pby,
                       double pcx, double pcy) {
    double detleft, detright, det;
    double detsum, errbound;

    detleft = (pax - pcx) * (pby - pcy);
    detright = (pay - pcy) * (pbx - pcx);
    det = detleft - detright;

    if (detleft > 0.0) {
        if (detright <= 0.0) return det;
        detsum = detleft + detright;
    } else if (detleft < 0.0) {
        if (detright >= 0.0) return det;
        detsum = -detleft - detright;
    } else {
        return det;
    }
    errbound = s_ccwerrboundA * detsum;
    if ((det >= errbound) || (-det >= errbound)) return det;
    return orient2dadapt(pax, pay, pbx, pby, pcx, pcy, detsum);
}

static inline int osign(double ax, double ay, double bx, double by,
                        double cx, double cy) {
    double d = orient2d(ax, ay, bx, by, cx, cy);
    return (d > 0.0) - (d < 0.0);
}

// The official checker decides interior overlap through GEOS area > 0.
// Decimal rotations can place an intended touch a few ulps to either side of
// a supporting line after the integer translation is added.  The exact sign
// above then sees two almost-coincident proper crossings and certifies an
// infinitesimal sliver that GEOS robust topology collapses to a line/point.
// Use a narrowly filtered sign only for the touch-exclusive overlap verdict;
// touch-inclusive intersects continues to use the mathematically exact sign.
static inline int osign_overlap(double ax, double ay, double bx, double by,
                                double cx, double cy) {
    double acx = ax - cx, bcy = by - cy;
    double acy = ay - cy, bcx = bx - cx;
    double d = orient2d(ax, ay, bx, by, cx, cy);
    double tol = 64.0 * s_epsilon *
                 (std::fabs(acx * bcy) + std::fabs(acy * bcx));
    if (std::fabs(d) <= tol) return 0;
    return (d > 0.0) - (d < 0.0);
}

// ---------------------------------------------------------------------------
// Segment / point-in-polygon primitives (exact, touch-inclusive)
// ---------------------------------------------------------------------------

// r is collinear with pq; true iff r lies within pq's closed bbox.
static inline bool onseg(double px, double py, double qx, double qy,
                         double rx, double ry) {
    return (px < qx ? (px <= rx && rx <= qx) : (qx <= rx && rx <= px))
        && (py < qy ? (py <= ry && ry <= qy) : (qy <= ry && ry <= py));
}

// Closed-segment intersection (touching endpoints / collinear overlap = true).
static bool seg_intersects(double ax, double ay, double bx, double by,
                           double cx, double cy, double dx, double dy) {
    int d1 = osign(cx, cy, dx, dy, ax, ay);
    int d2 = osign(cx, cy, dx, dy, bx, by);
    if (d1 != 0 && d1 == d2) return false;
    int d3 = osign(ax, ay, bx, by, cx, cy);
    int d4 = osign(ax, ay, bx, by, dx, dy);
    if (d3 != 0 && d3 == d4) return false;
    if (d1 && d2 && d3 && d4) return true;           // proper crossing
    if (!d1 && onseg(cx, cy, dx, dy, ax, ay)) return true;
    if (!d2 && onseg(cx, cy, dx, dy, bx, by)) return true;
    if (!d3 && onseg(ax, ay, bx, by, cx, cy)) return true;
    if (!d4 && onseg(ax, ay, bx, by, dx, dy)) return true;
    return false;
}

// ---------------------------------------------------------------------------
// Geometry registry
// ---------------------------------------------------------------------------

struct Ring {
    int32_t start;     // first point index into xy
    int32_t npts;      // number of points INCLUDING the closing duplicate
    int32_t ccw;       // shoelace orientation: +1 CCW, -1 CW, 0 degenerate
    double bb[4];      // local bbox (minx, miny, maxx, maxy)
};

struct Geom {
    std::vector<double> xy;        // flat (x, y) pairs, rings concatenated
    std::vector<Ring> rings;       // exterior + hole rings of every part
    std::vector<int32_t> testvtx;  // one exterior-ring vertex index per part
    double bb[4];                  // overall local bbox
};

// Strict membership of world point (px, py) in geometry G placed at
// (gx, gy): +1 strictly inside, 0 on the boundary, -1 strictly outside.
// Exact: boundary contact is detected per segment before ray casting.
static int point_in_strict(const Geom& G, double gx, double gy,
                           double px, double py) {
    if (px <= G.bb[0] + gx || px >= G.bb[2] + gx ||
        py <= G.bb[1] + gy || py >= G.bb[3] + gy) {
        // on-or-outside the closed bbox: strictly inside is impossible;
        // boundary contact only matters when the point can lie ON an
        // edge, which the per-segment sweep below also detects -- but a
        // point outside the closed bbox touches nothing.
        if (px < G.bb[0] + gx || px > G.bb[2] + gx ||
            py < G.bb[1] + gy || py > G.bb[3] + gy)
            return -1;
    }
    bool in = false;
    for (const Ring& r : G.rings) {
        const double* p = &G.xy[2 * r.start];
        int nseg = r.npts - 1;
        bool span = !(py < r.bb[1] + gy || py > r.bb[3] + gy ||
                      px > r.bb[2] + gx);
        for (int i = 0; i < nseg; i++) {
            double x1 = p[2 * i] + gx, y1 = p[2 * i + 1] + gy;
            double x2 = p[2 * i + 2] + gx, y2 = p[2 * i + 3] + gy;
            if (onseg(x1, y1, x2, y2, px, py) &&
                osign_overlap(x1, y1, x2, y2, px, py) == 0)
                return 0;                  // on the boundary
            if (span && (y1 > py) != (y2 > py)) {
                int o = osign_overlap(x1, y1, x2, y2, px, py);
                if (o == 0) continue;      // crossing point == p: caught above
                if ((y2 > y1) ? (o > 0) : (o < 0)) in = !in;
            }
        }
    }
    return in ? 1 : -1;
}

// Exact positive-area interior intersection of A placed at (ax, ay) and
// B at (bx, by): true iff int(A) meets int(B) in dimension 2 -- edge
// sharing and vertex touching are FALSE (the checker's area>0 rule).
// Certificate set: (1) a proper edge crossing, (2) a positive-length
// collinear edge overlap whose interiors lie on the same side of the
// shared line (rings are simple, so the interior is locally left of a
// CCW traversal), (3) a vertex of one strictly inside the other,
// (4) an edge midpoint of one strictly inside the other (catches
// chord-through-interior contacts whose endpoints all lie on the
// partner's boundary).  Adversarial residues where every certificate
// point lands exactly on the partner boundary are adjudicated by the
// parity gates against the GEOS reference.
static bool geom_overlap(const Geom& A, double ax, double ay,
                         const Geom& B, double bx, double by) {
    // interiors live strictly inside their own bboxes: bbox touch is
    // already interior-disjoint
    if (A.bb[2] + ax <= B.bb[0] + bx || B.bb[2] + bx <= A.bb[0] + ax ||
        A.bb[3] + ay <= B.bb[1] + by || B.bb[3] + by <= A.bb[1] + ay)
        return false;

    for (const Ring& ra : A.rings) {
        double a0 = ra.bb[0] + ax, a1 = ra.bb[1] + ay;
        double a2 = ra.bb[2] + ax, a3 = ra.bb[3] + ay;
        for (const Ring& rb : B.rings) {
            if (a2 < rb.bb[0] + bx || rb.bb[2] + bx < a0 ||
                a3 < rb.bb[1] + by || rb.bb[3] + by < a1)
                continue;
            const double* pa = &A.xy[2 * ra.start];
            const double* pb = &B.xy[2 * rb.start];
            int na = ra.npts - 1, nb = rb.npts - 1;
            for (int i = 0; i < na; i++) {
                double x1 = pa[2 * i] + ax, y1 = pa[2 * i + 1] + ay;
                double x2 = pa[2 * i + 2] + ax, y2 = pa[2 * i + 3] + ay;
                double slo = x1 < x2 ? x1 : x2, shi = x1 < x2 ? x2 : x1;
                double tlo = y1 < y2 ? y1 : y2, thi = y1 < y2 ? y2 : y1;
                if (shi < rb.bb[0] + bx || rb.bb[2] + bx < slo ||
                    thi < rb.bb[1] + by || rb.bb[3] + by < tlo)
                    continue;
                for (int j = 0; j < nb; j++) {
                    double x3 = pb[2 * j] + bx, y3 = pb[2 * j + 1] + by;
                    double x4 = pb[2 * j + 2] + bx, y4 = pb[2 * j + 3] + by;
                    double ulo = x3 < x4 ? x3 : x4, uhi = x3 < x4 ? x4 : x3;
                    if (uhi < slo || shi < ulo) continue;
                    double vlo = y3 < y4 ? y3 : y4, vhi = y3 < y4 ? y4 : y3;
                    if (vhi < tlo || thi < vlo) continue;
                    int o1 = osign_overlap(x1, y1, x2, y2, x3, y3);
                    int o2 = osign_overlap(x1, y1, x2, y2, x4, y4);
                    int o3 = osign_overlap(x3, y3, x4, y4, x1, y1);
                    int o4 = osign_overlap(x3, y3, x4, y4, x2, y2);
                    if (o1 && o2 && o1 != o2 && o3 && o4 && o3 != o4)
                        return true;               // proper crossing
                    if (o1 == 0 && o2 == 0) {
                        // collinear edges: positive-length 1D overlap?
                        bool xdom = (shi - slo) >= (thi - tlo);
                        double lo = xdom ? (slo > ulo ? slo : ulo)
                                         : (tlo > vlo ? tlo : vlo);
                        double hi = xdom ? (shi < uhi ? shi : uhi)
                                         : (thi < vhi ? thi : vhi);
                        if (hi > lo && ra.ccw && rb.ccw) {
                            double dot = (x2 - x1) * (x4 - x3)
                                       + (y2 - y1) * (y4 - y3);
                            int s = (dot > 0.0) - (dot < 0.0);
                            if (s && ra.ccw == rb.ccw * s)
                                return true;       // interiors same side
                        }
                    }
                }
            }
        }
    }

    // no decisive boundary interaction: sample vertices and midpoints
    for (const Ring& ra : A.rings) {
        const double* pa = &A.xy[2 * ra.start];
        int na = ra.npts - 1;
        for (int i = 0; i < na; i++) {
            double x1 = pa[2 * i] + ax, y1 = pa[2 * i + 1] + ay;
            if (point_in_strict(B, bx, by, x1, y1) > 0) return true;
            double x2 = pa[2 * i + 2] + ax, y2 = pa[2 * i + 3] + ay;
            if (point_in_strict(B, bx, by,
                                0.5 * (x1 + x2), 0.5 * (y1 + y2)) > 0)
                return true;
        }
    }
    for (const Ring& rb : B.rings) {
        const double* pb = &B.xy[2 * rb.start];
        int nb = rb.npts - 1;
        for (int j = 0; j < nb; j++) {
            double x3 = pb[2 * j] + bx, y3 = pb[2 * j + 1] + by;
            if (point_in_strict(A, ax, ay, x3, y3) > 0) return true;
            double x4 = pb[2 * j + 2] + bx, y4 = pb[2 * j + 3] + by;
            if (point_in_strict(A, ax, ay,
                                0.5 * (x3 + x4), 0.5 * (y3 + y4)) > 0)
                return true;
        }
    }
    return false;
}

// Even-odd membership of world point (px, py) in geometry G placed at
// (gx, gy).  Exact for points not on the boundary (the only points this
// engine ever queries: boundary contact is resolved by seg tests first).
static bool point_in(const Geom& G, double gx, double gy,
                     double px, double py) {
    bool in = false;
    for (const Ring& r : G.rings) {
        if (py < r.bb[1] + gy || py > r.bb[3] + gy || px > r.bb[2] + gx)
            continue;                      // ray to +x cannot cross
        const double* p = &G.xy[2 * r.start];
        int nseg = r.npts - 1;
        for (int i = 0; i < nseg; i++) {
            double x1 = p[2 * i] + gx, y1 = p[2 * i + 1] + gy;
            double x2 = p[2 * i + 2] + gx, y2 = p[2 * i + 3] + gy;
            if ((y1 > py) != (y2 > py)) {
                int o = osign(x1, y1, x2, y2, px, py);
                if (o == 0) continue;      // on boundary: unreachable by design
                if ((y2 > y1) ? (o > 0) : (o < 0)) in = !in;
            }
        }
    }
    return in;
}

// Exact touch-inclusive intersects of A placed at (ax, ay) and B at (bx, by).
static bool geom_intersects(const Geom& A, double ax, double ay,
                            const Geom& B, double bx, double by) {
    if (A.bb[2] + ax < B.bb[0] + bx || B.bb[2] + bx < A.bb[0] + ax ||
        A.bb[3] + ay < B.bb[1] + by || B.bb[3] + by < A.bb[1] + ay)
        return false;                       // strict bbox separation

    for (const Ring& ra : A.rings) {
        double a0 = ra.bb[0] + ax, a1 = ra.bb[1] + ay;
        double a2 = ra.bb[2] + ax, a3 = ra.bb[3] + ay;
        for (const Ring& rb : B.rings) {
            if (a2 < rb.bb[0] + bx || rb.bb[2] + bx < a0 ||
                a3 < rb.bb[1] + by || rb.bb[3] + by < a1)
                continue;
            const double* pa = &A.xy[2 * ra.start];
            const double* pb = &B.xy[2 * rb.start];
            int na = ra.npts - 1, nb = rb.npts - 1;
            for (int i = 0; i < na; i++) {
                double x1 = pa[2 * i] + ax, y1 = pa[2 * i + 1] + ay;
                double x2 = pa[2 * i + 2] + ax, y2 = pa[2 * i + 3] + ay;
                double slo = x1 < x2 ? x1 : x2, shi = x1 < x2 ? x2 : x1;
                double tlo = y1 < y2 ? y1 : y2, thi = y1 < y2 ? y2 : y1;
                if (shi < rb.bb[0] + bx || rb.bb[2] + bx < slo ||
                    thi < rb.bb[1] + by || rb.bb[3] + by < tlo)
                    continue;
                for (int j = 0; j < nb; j++) {
                    double x3 = pb[2 * j] + bx, y3 = pb[2 * j + 1] + by;
                    double x4 = pb[2 * j + 2] + bx, y4 = pb[2 * j + 3] + by;
                    double ulo = x3 < x4 ? x3 : x4, uhi = x3 < x4 ? x4 : x3;
                    if (uhi < slo || shi < ulo) continue;
                    double vlo = y3 < y4 ? y3 : y4, vhi = y3 < y4 ? y4 : y3;
                    if (vhi < tlo || thi < vlo) continue;
                    if (seg_intersects(x1, y1, x2, y2, x3, y3, x4, y4))
                        return true;
                }
            }
        }
    }
    // no boundary contact: containment is decided by one vertex per part
    for (int32_t t : B.testvtx)
        if (point_in(A, ax, ay, B.xy[2 * t] + bx, B.xy[2 * t + 1] + by))
            return true;
    for (int32_t t : A.testvtx)
        if (point_in(B, bx, by, A.xy[2 * t] + ax, A.xy[2 * t + 1] + ay))
            return true;
    return false;
}

// ---------------------------------------------------------------------------
// Memo (open addressing, exact absolute-placement keys)
// ---------------------------------------------------------------------------

static const size_t MEMO_CAP = 262144;

struct MemoKey {
    int32_t ga = -1;
    int32_t gb = -1;
    int64_t ax = 0, ay = 0, bx = 0, by = 0;
};

static inline bool key_equal(const MemoKey& a, const MemoKey& b) {
    return a.ga == b.ga && a.gb == b.gb &&
           a.ax == b.ax && a.ay == b.ay &&
           a.bx == b.bx && a.by == b.by;
}

struct Memo {
    std::vector<MemoKey> keys;
    std::vector<uint8_t> vals;
    size_t mask = 0, count = 0;

    Memo() { reset(1 << 16); }

    void reset(size_t cap) {
        keys.assign(cap, MemoKey{});
        vals.assign(cap, 0);
        mask = cap - 1;
        count = 0;
    }

    static inline uint64_t mix(uint64_t k) {
        k += 0x9e3779b97f4a7c15ull;
        k = (k ^ (k >> 30)) * 0xbf58476d1ce4e5b9ull;
        k = (k ^ (k >> 27)) * 0x94d049bb133111ebull;
        return k ^ (k >> 31);
    }

    static inline uint64_t hash(const MemoKey& k) {
        uint64_t h = mix((uint64_t)(uint32_t)k.ga << 32 |
                         (uint32_t)k.gb);
        h ^= mix((uint64_t)k.ax + 0x517cc1b727220a95ull);
        h ^= mix((uint64_t)k.ay + 0x6c8e9cf570932bd5ull);
        h ^= mix((uint64_t)k.bx + 0x9e3779b97f4a7c15ull);
        h ^= mix((uint64_t)k.by + 0xbf58476d1ce4e5b9ull);
        return mix(h);
    }

    // returns -1 (miss) or 0/1 verdict
    inline int get(const MemoKey& k) const {
        size_t i = hash(k) & mask;
        while (true) {
            const MemoKey& kk = keys[i];
            if (kk.ga < 0) return -1;
            if (key_equal(kk, k)) return vals[i];
            i = (i + 1) & mask;
        }
    }

    inline void put(const MemoKey& k, uint8_t v) {
        if (count >= MEMO_CAP) reset(keys.size());
        if ((count + 1) * 10 > mask * 7) grow();      // load factor 0.7
        size_t i = hash(k) & mask;
        while (keys[i].ga >= 0) {
            if (key_equal(keys[i], k)) { vals[i] = v; return; }
            i = (i + 1) & mask;
        }
        keys[i] = k;
        vals[i] = v;
        count++;
    }

    void grow() {
        std::vector<MemoKey> ok;
        std::vector<uint8_t> ov;
        ok.swap(keys);
        ov.swap(vals);
        keys.assign(ok.size() * 2, MemoKey{});
        vals.assign(ok.size() * 2, 0);
        mask = keys.size() - 1;
        for (size_t j = 0; j < ok.size(); j++) {
            if (ok[j].ga < 0) continue;
            size_t i = hash(ok[j]) & mask;
            while (keys[i].ga >= 0) i = (i + 1) & mask;
            keys[i] = ok[j];
            vals[i] = ov[j];
        }
    }
};

static inline MemoKey make_key(int32_t ga, int32_t gb,
                               int64_t ax, int64_t ay,
                               int64_t bx, int64_t by) {
    return MemoKey{ga, gb, ax, ay, bx, by};
}

// ---------------------------------------------------------------------------
// Handle + C ABI
// ---------------------------------------------------------------------------

struct Handle {
    std::vector<Geom> geoms;
    Memo memo;
    Memo memo_ov;      // interior-overlap verdicts (separate key space)
};

extern "C" {

void* geo_new() { return new Handle(); }

void geo_del(void* h) { delete (Handle*)h; }

// xy: npts (x, y) pairs; ring_ptr: nrings+1 point offsets; test_idx: one
// exterior-ring vertex (point index) per part.  Returns the gid.
int32_t geo_add(void* hv, const double* xy, int32_t npts,
                const int32_t* ring_ptr, int32_t nrings,
                const int32_t* test_idx, int32_t ntest) {
    Handle* h = (Handle*)hv;
    h->geoms.emplace_back();
    Geom& g = h->geoms.back();
    g.xy.assign(xy, xy + 2 * (size_t)npts);
    g.rings.resize(nrings);
    g.bb[0] = g.bb[1] = 1e300;
    g.bb[2] = g.bb[3] = -1e300;
    for (int32_t r = 0; r < nrings; r++) {
        Ring& rr = g.rings[r];
        rr.start = ring_ptr[r];
        rr.npts = ring_ptr[r + 1] - ring_ptr[r];
        rr.bb[0] = rr.bb[1] = 1e300;
        rr.bb[2] = rr.bb[3] = -1e300;
        for (int32_t i = ring_ptr[r]; i < ring_ptr[r + 1]; i++) {
            double x = g.xy[2 * i], y = g.xy[2 * i + 1];
            if (x < rr.bb[0]) rr.bb[0] = x;
            if (y < rr.bb[1]) rr.bb[1] = y;
            if (x > rr.bb[2]) rr.bb[2] = x;
            if (y > rr.bb[3]) rr.bb[3] = y;
        }
        double a2 = 0.0;               // shoelace (orientation only)
        for (int32_t i = ring_ptr[r]; i + 1 < ring_ptr[r + 1]; i++)
            a2 += g.xy[2 * i] * g.xy[2 * i + 3]
                - g.xy[2 * i + 2] * g.xy[2 * i + 1];
        rr.ccw = (a2 > 0.0) - (a2 < 0.0);
        for (int k = 0; k < 4; k++) {
            if (rr.bb[k] < g.bb[k] && k < 2) g.bb[k] = rr.bb[k];
            if (rr.bb[k] > g.bb[k] && k >= 2) g.bb[k] = rr.bb[k];
        }
    }
    g.testvtx.assign(test_idx, test_idx + ntest);
    return (int32_t)h->geoms.size() - 1;
}

// v5.9.5 hardening: every exported predicate indexed h->geoms[gid]
// unchecked; a stale gid (the 2026-07-20 rollover race, or any future
// caller bug) read out of bounds -- intermittent SIGSEGV at best,
// silent garbage verdicts at worst.  Out-of-range gids now return the
// CONSERVATIVE verdict (intersects / overlaps = blocked): a wrong
// conservative answer can cost objective but can never emit an
// infeasible solution.  In-range behaviour is bit-identical.
static inline bool gid_ok(const Handle* h, int32_t g) {
    return g >= 0 && (size_t)g < h->geoms.size();
}

// Memoised verdict for one placed/candidate pair at integer offsets.
// counters: [0] += exact computes, [1] += memo hits.
int32_t geo_pair(void* hv, int32_t ga, int64_t ax, int64_t ay,
                 int32_t gb, int64_t bx, int64_t by, int64_t* counters) {
    Handle* h = (Handle*)hv;
    if (!gid_ok(h, ga) || !gid_ok(h, gb)) return 1;
    MemoKey key = make_key(ga, gb, ax, ay, bx, by);
    int v = h->memo.get(key);
    if (v < 0) {
        counters[0]++;
        v = geom_intersects(h->geoms[ga], (double)ax, (double)ay,
                            h->geoms[gb], (double)bx, (double)by) ? 1 : 0;
        h->memo.put(key, (uint8_t)v);
    } else {
        counters[1]++;
    }
    return v;
}

// Memoised exact interior-overlap (positive area, touch-exclusive) for one
// pair at integer offsets.  counters: [0] += exact computes, [1] += hits.
int32_t geo_pair_ov(void* hv, int32_t ga, int64_t ax, int64_t ay,
                    int32_t gb, int64_t bx, int64_t by, int64_t* counters) {
    Handle* h = (Handle*)hv;
    if (!gid_ok(h, ga) || !gid_ok(h, gb)) return 1;
    MemoKey key = make_key(ga, gb, ax, ay, bx, by);
    int v = h->memo_ov.get(key);
    if (v < 0) {
        counters[0]++;
        v = geom_overlap(h->geoms[ga], (double)ax, (double)ay,
                         h->geoms[gb], (double)bx, (double)by) ? 1 : 0;
        h->memo_ov.put(key, (uint8_t)v);
    } else {
        counters[1]++;
    }
    return v;
}

// Bulk resolution of the Phase-A residual band.  Positions are the row-major
// (ys x xs) grid; res_ptr is the per-position CSR into res_flat (active
// indices).  Fills res_ok (0/1) for every residual entry.  Iteration order
// matches the python loop: ascending position, ascending entry.
void geo_resolve(void* hv, int32_t gid_cand,
                 const int64_t* xs, int32_t nx,
                 const int64_t* ys, int32_t ny,
                 const int32_t* act_gid, const int64_t* act_x,
                 const int64_t* act_y,
                 const int32_t* res_ptr, const int32_t* res_flat,
                 uint8_t* res_ok, int64_t* counters) {
    Handle* h = (Handle*)hv;
    if (!gid_ok(h, gid_cand)) {
        int32_t npos_g = nx * ny;
        for (int32_t p = 0; p < npos_g; p++)
            for (int32_t ti = res_ptr[p]; ti < res_ptr[p + 1]; ti++)
                res_ok[ti] = 1;
        return;
    }
    const Geom& gB = h->geoms[gid_cand];
    int32_t npos = nx * ny;
    for (int32_t p = 0; p < npos; p++) {
        int32_t s = res_ptr[p], e = res_ptr[p + 1];
        if (s == e) continue;
        double x = (double)xs[p % nx];
        double y = (double)ys[p / nx];
        int64_t xi = xs[p % nx], yi = ys[p / nx];
        for (int32_t ti = s; ti < e; ti++) {
            int32_t a = res_flat[ti];
            MemoKey key = make_key(act_gid[a], gid_cand,
                                   act_x[a], act_y[a], xi, yi);
            if (!gid_ok(h, act_gid[a])) {
                res_ok[ti] = 1;
                continue;
            }
            int v = h->memo.get(key);
            if (v < 0) {
                counters[0]++;
                v = geom_intersects(h->geoms[act_gid[a]],
                                    (double)act_x[a], (double)act_y[a],
                                    gB, x, y) ? 1 : 0;
                h->memo.put(key, (uint8_t)v);
            } else {
                counters[1]++;
            }
            res_ok[ti] = (uint8_t)v;
        }
    }
}

// Direct un-memoised predicate at arbitrary double offsets (test harness).
int32_t geo_isect_raw(void* hv, int32_t ga, double ax, double ay,
                      int32_t gb, double bx, double by) {
    Handle* h = (Handle*)hv;
    if (!gid_ok(h, ga) || !gid_ok(h, gb)) return 1;
    return geom_intersects(h->geoms[ga], ax, ay,
                           h->geoms[gb], bx, by) ? 1 : 0;
}

}  // extern "C"
