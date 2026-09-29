"""Multi-threaded CPU kernels (numba): the CPU counterpart of flybrain/metal.py.

torch on CPU runs the model as thousands of small single-threaded ops per frame (sparse products, then ~30
element-wise ops per LIF step, then a vectorised ray tracer), which put a CPU frame at ~220 ms for 10 ms of
brain time. These kernels mirror the Metal ones line for line: a row-parallel CSR mat-vec (no atomics, same
summation order as scipy, deterministic for any thread count), the event-driven synaptic scatter, the fused
LIF update, the fused optic substep and one-ray-per-iteration tracing with the same intersection precedence,
patterns and value noise as world.World._trace. Arrays are shared with the torch tensors (no copies).

Disable with FLYBRAIN_NUMBA=0; without numba installed the torch path is used unchanged.
"""
from __future__ import annotations

import math
import os

import numpy as np
import scipy.sparse as sp
import torch

try:
    from numba import njit, prange
    AVAILABLE = True
except ImportError:
    AVAILABLE = False

# Runtime switch (tests that pin the torch reference path set it False before building a brain).
ENABLED = os.environ.get("FLYBRAIN_NUMBA", "1") != "0"


def use(device) -> bool:
    return AVAILABLE and ENABLED and torch.device(device).type == "cpu"


def _np(t: torch.Tensor) -> np.ndarray:
    """Flat numpy view sharing memory with a contiguous CPU tensor."""
    if not t.is_contiguous():
        raise ValueError("numba kernels need contiguous tensors")
    return t.detach().reshape(-1).numpy()


if AVAILABLE:
    @njit(parallel=True, cache=True, nogil=True)
    def _spmv(bounds, indptr, indices, data, x, y):
        # rows split into equal-nnz chunks (one per thread): optic rows range from 1 to 11k entries, so equal-row
        # chunks left most threads idle. Within a row the summation order is scipy's, whatever the chunking.
        B = x.shape[0]
        for c in prange(bounds.shape[0] - 1):
            for r in range(bounds[c], bounds[c + 1]):
                s, e = indptr[r], indptr[r + 1]
                for b in range(B):
                    acc = np.float32(0.0)
                    for k in range(s, e):
                        acc += data[k] * x[b, indices[k]]
                    y[b, r] = acc

    @njit(parallel=True, cache=True, nogil=True)
    def _spmv1(bounds, indptr, indices, data, x, y):
        # the B = 1 case with flat vectors: ~25% faster than indexing a (1, n) array
        for c in prange(bounds.shape[0] - 1):
            for r in range(bounds[c], bounds[c + 1]):
                acc = np.float32(0.0)
                for k in range(indptr[r], indptr[r + 1]):
                    acc += data[k] * x[indices[k]]
                y[r] = acc

    @njit(cache=True, nogil=True)
    def _event_scatter(out_ptr, out_post, out_w, x, g, N):
        for gid in range(x.shape[0]):
            xv = x[gid]
            if xv != 0.0:
                base = (gid // N) * N
                pre = gid - base
                for j in range(out_ptr[pre], out_ptr[pre + 1]):
                    g[base + out_post[j]] += out_w[j] * xv

    @njit(parallel=True, cache=True, nogil=True)
    def _lif(v, g, drive, adapt, refrac, active, poisson_p, rnd, res, std_u, spikes, out_buf, rate, counts, P, N, flags):
        for tid in prange(v.shape[0]):
            i = tid % N
            target = P[0] + g[tid] + drive[tid] - adapt[tid]
            vv = target + (v[tid] - target) * P[3]
            rf = refrac[tid]
            if rf > 0.0:
                vv = P[1]
            rf = max(rf - P[4], np.float32(0.0))
            s = np.float32(1.0 if vv >= P[2] else 0.0) * active[i]
            if flags & 1:
                if rnd[tid] < poisson_p[tid]:
                    s = np.float32(1.0)
            fired = s > 0.0
            if fired:
                vv = P[1]; rf = P[5]
            v[tid] = vv; refrac[tid] = rf
            if flags & 8:
                adapt[tid] = adapt[tid] * P[7] + s * P[6]
            if flags & 2:
                r = res[tid]
                out_buf[tid] = s * r
                if fired:
                    r = r * (np.float32(1.0) - std_u[i])
                res[tid] = np.float32(1.0) - (np.float32(1.0) - r) * P[8]
            else:
                out_buf[tid] = s
            spikes[tid] = s
            if flags & 4:
                counts[tid] += s
            rate[tid] = rate[tid] * P[9] + s * P[10]

    @njit(parallel=True, cache=True, nogil=True)
    def _optic_dr(v, bvec, dr, n):
        for tid in prange(v.shape[0]):
            bb = bvec[tid % n]
            dr[tid] = min(max(v[tid] + bb, np.float32(0.0)), np.float32(1.0)) - bb

    @njit(parallel=True, cache=True, nogil=True)
    def _optic_substep(v, adapt, dr, y, pr_in, spk_in, a, bvec, gain_rr, adapt_gain, a_ad, n):
        for tid in prange(v.shape[0]):
            i = tid % n
            d = dr[tid]
            inp = gain_rr * y[tid] + pr_in[tid] - adapt_gain * adapt[tid]
            inp = inp + spk_in[tid]
            vv = inp + (v[tid] - inp) * a[i]
            v[tid] = vv
            adapt[tid] = d + (adapt[tid] - d) * a_ad
            bb = bvec[i]
            dr[tid] = min(max(vv + bb, np.float32(0.0)), np.float32(1.0)) - bb

    # ------------------------------------------------------------ ray tracer (world.World._trace)
    INF = 1e9

    @njit(cache=True, nogil=True, inline="always")
    def _intersect(ox, oy, oz, dx, dy, dz, sc, sr, blo, bhi, pp, pn, miss):
        eps = 1e-4
        best_t = INF; nx = 0.0; ny = 0.0; nz = 0.0; best = miss
        S = sc.shape[0]; Bx = blo.shape[0]; P = pp.shape[0]
        ti = INF; si = -1
        for k in range(S):                                    # ellipsoids: scale space to unit spheres
            rx, ry, rz = sr[k, 0], sr[k, 1], sr[k, 2]
            ocx = (ox - sc[k, 0]) / rx; ocy = (oy - sc[k, 1]) / ry; ocz = (oz - sc[k, 2]) / rz
            ddx = dx / rx; ddy = dy / ry; ddz = dz / rz
            a = ddx * ddx + ddy * ddy + ddz * ddz
            b = 2.0 * (ocx * ddx + ocy * ddy + ocz * ddz)
            c = ocx * ocx + ocy * ocy + ocz * ocz - 1.0
            disc = b * b - 4.0 * a * c
            sq = math.sqrt(max(disc, 0.0))
            t0 = (-b - sq) / (2.0 * a); t1 = (-b + sq) / (2.0 * a)
            t = t0 if t0 > eps else t1
            if not (disc > 0.0 and t > eps):
                t = INF
            if t < ti:
                ti = t; si = k
        if si >= 0 and ti < best_t:
            px = ox + dx * ti; py = oy + dy * ti; pz = oz + dz * ti
            rx, ry, rz = sr[si, 0], sr[si, 1], sr[si, 2]
            mx = (px - sc[si, 0]) / (rx * rx); my = (py - sc[si, 1]) / (ry * ry); mz = (pz - sc[si, 2]) / (rz * rz)
            nn = max(math.sqrt(mx * mx + my * my + mz * mz), 1e-9)
            best_t = ti; nx = mx / nn; ny = my / nn; nz = mz / nn; best = si
        ti = INF; bi = -1; bax = 0
        ivx = 1.0 / (1e-9 if abs(dx) < 1e-9 else dx)
        ivy = 1.0 / (1e-9 if abs(dy) < 1e-9 else dy)
        ivz = 1.0 / (1e-9 if abs(dz) < 1e-9 else dz)
        for k in range(Bx):                                   # axis-aligned boxes: slabs
            lx = (blo[k, 0] - ox) * ivx; hx = (bhi[k, 0] - ox) * ivx
            ly = (blo[k, 1] - oy) * ivy; hy = (bhi[k, 1] - oy) * ivy
            lz = (blo[k, 2] - oz) * ivz; hz = (bhi[k, 2] - oz) * ivz
            tnx = min(lx, hx); tny = min(ly, hy); tnz = min(lz, hz)
            te = tnx; ax = 0
            if tny > te:
                te = tny; ax = 1
            if tnz > te:
                te = tnz; ax = 2
            tx = min(min(max(lx, hx), max(ly, hy)), max(lz, hz))
            t = te if (tx > te and te > eps) else INF
            if t < ti:
                ti = t; bi = k; bax = ax
        if bi >= 0 and ti < best_t:
            dax = dx if bax == 0 else (dy if bax == 1 else dz)
            sgn = -1.0 if dax > 0 else (1.0 if dax < 0 else 0.0)
            best_t = ti; nx = sgn if bax == 0 else 0.0; ny = sgn if bax == 1 else 0.0; nz = sgn if bax == 2 else 0.0
            best = S + bi
        ti = INF; pi = -1
        for k in range(P):                                    # planes
            denom = dx * pn[k, 0] + dy * pn[k, 1] + dz * pn[k, 2]
            num = (pp[k, 0] - ox) * pn[k, 0] + (pp[k, 1] - oy) * pn[k, 1] + (pp[k, 2] - oz) * pn[k, 2]
            t = num / (1e-9 if abs(denom) < 1e-9 else denom)
            if not (t > eps and abs(denom) > 1e-9):
                t = INF
            if t < ti:
                ti = t; pi = k
        if pi >= 0 and ti < best_t:
            mx, my, mz = pn[pi, 0], pn[pi, 1], pn[pi, 2]
            if mx * dx + my * dy + mz * dz > 0:
                mx = -mx; my = -my; mz = -mz
            best_t = ti; nx = mx; ny = my; nz = mz; best = S + Bx + pi
        return best_t, nx, ny, nz, best

    @njit(cache=True, nogil=True, inline="always")
    def _hash3(i0, i1, i2):
        M = 2147483647
        h = (i0 * 374761393 + i1 * 668265263 + i2 * 2147483647) % M
        h = ((h ^ (h >> 13)) * 1274126177) % M
        return (h % 65536) / 32768.0 - 1.0

    @njit(cache=True, nogil=True)
    def _value_noise(px, py, pz, scale):
        qx = px / scale; qy = py / scale; qz = pz / scale
        flx = math.floor(qx); fly = math.floor(qy); flz = math.floor(qz)
        i0 = np.int64(flx); i1 = np.int64(fly); i2 = np.int64(flz)
        fx = qx - flx; fy = qy - fly; fz = qz - flz
        fx = fx * fx * (3 - 2 * fx); fy = fy * fy * (3 - 2 * fy); fz = fz * fz * (3 - 2 * fz)
        out = 0.0
        for ddx in range(2):
            wx = fx if ddx else 1 - fx
            for ddy in range(2):
                wy = fy if ddy else 1 - fy
                for ddz in range(2):
                    wz = fz if ddz else 1 - fz
                    out += wx * wy * wz * _hash3(i0 + ddx, i1 + ddy, i2 + ddz)
        return out

    @njit(parallel=True, cache=True, nogil=True)
    def _trace(o, d, sc, sr, blo, bhi, pp, pn, refl, refl2, emit, pattern, pscale, light, detail, out):
        miss = refl.shape[0] - 1
        for r in prange(o.shape[0]):
            ox, oy, oz = o[r, 0], o[r, 1], o[r, 2]
            dx, dy, dz = d[r, 0], d[r, 1], d[r, 2]
            t, nx, ny, nz, mid = _intersect(ox, oy, oz, dx, dy, dz, sc, sr, blo, bhi, pp, pn, miss)
            if not (t < INF):
                for k in range(4):
                    out[r, k] = 0.0
                continue
            px = ox + dx * t; py = oy + dy * t; pz = oz + dz * t
            tlx = light[0] - px; tly = light[1] - py; tlz = light[2] - pz
            dist = math.sqrt(tlx * tlx + tly * tly + tlz * tlz)
            inv = 1.0 / max(dist, 1e-9)
            lx = tlx * inv; ly = tly * inv; lz = tlz * inv
            lam = max(nx * lx + ny * ly + nz * lz, 0.0)
            ts, _a, _b, _c, _d = _intersect(px + nx * 1e-3, py + ny * 1e-3, pz + nz * 1e-3, lx, ly, lz,
                                            sc, sr, blo, bhi, pp, pn, miss)
            lit = 1.0 if ts >= dist else 0.0
            falloff = 4.0 / (1.0 + dist * dist)
            pat = pattern[mid]; psc = pscale[mid]
            fx = math.floor(px / psc); fy = math.floor(py / psc)
            checks = (fx + fy) % 2.0 == 1.0
            horiz = fy if abs(nx) > 0.5 else fx
            stripes = horiz % 2.0 == 1.0
            planks = (fy % 2.0 == 1.0) or ((fx + 3.0 * fy) % 7.0 == 0.0)
            second = checks if pat == 1 else (stripes if pat == 2 else (planks if pat == 3 else False))
            m = 1.0
            if detail > 0.0:
                m = 1.0 + detail * (0.5 * _value_noise(px, py, pz, 0.02) + 0.3 * _value_noise(px, py, pz, 0.008)
                                    + 0.2 * _value_noise(px, py, pz, 0.003))
                m = max(m, 0.1)
            shade = lam * lit * falloff
            for k in range(4):
                rf = refl2[mid, k] if second else refl[mid, k]
                if detail > 0.0:
                    rf = rf * m
                out[r, k] = rf * (light[7 + k] + light[3 + k] * shade) + emit[mid, k]


class NumbaCSR:
    """A scipy sparse matrix for batched CPU products y (B, rows) = M @ x (B, cols); same API as metal.MetalCSR."""

    def __init__(self, M: sp.spmatrix, device="cpu"):
        M = sp.csr_matrix(M, dtype=np.float32)
        M.sum_duplicates(); M.sort_indices()
        self.shape = tuple(int(s) for s in M.shape)
        self.nnz = int(M.nnz)
        itype = np.int32 if max(*self.shape, self.nnz) < 2**31 else np.int64
        self.indptr = M.indptr.astype(itype)
        self.indices = M.indices.astype(itype)
        self.values = M.data.astype(np.float32)
        import numba
        chunks = max(1, numba.get_num_threads())
        bounds = np.searchsorted(self.indptr, np.linspace(0, self.nnz, chunks + 1)).astype(np.int64)
        bounds[0], bounds[-1] = 0, self.shape[0]
        self.bounds = np.maximum.accumulate(bounds)

    @classmethod
    def from_torch(cls, M: torch.Tensor) -> "NumbaCSR":
        """From a CPU torch sparse CSR/COO tensor, same values."""
        if M.layout == torch.sparse_csr:
            D = sp.csr_matrix((M.values().numpy(), M.col_indices().numpy(), M.crow_indices().numpy()), shape=tuple(M.shape))
        else:
            M = M.coalesce(); i = M.indices().numpy()
            D = sp.csr_matrix((M.values().numpy(), (i[0], i[1])), shape=tuple(M.shape))
        return cls(D)

    def matvec(self, x: torch.Tensor) -> torch.Tensor:
        squeeze = x.dim() == 1
        xn = (x[None] if squeeze else x).detach().to(torch.float32).contiguous().numpy()
        if xn.shape[1] != self.shape[1]:
            raise ValueError(f"matvec: x has {xn.shape[1]} columns, matrix has {self.shape[1]}")
        y = np.empty((xn.shape[0], self.shape[0]), dtype=np.float32)
        if y.size and xn.shape[0] == 1:
            _spmv1(self.bounds, self.indptr, self.indices, self.values, xn[0], y[0])
        elif y.size:
            _spmv(self.bounds, self.indptr, self.indices, self.values, xn, y)
        y = torch.from_numpy(y)
        return y[0] if squeeze else y

    def _nnz(self) -> int:
        return self.nnz


# ---------------------------------------------------------------- wrappers with the metal.py signatures
def event_scatter(out_ptr, out_post, out_w, x: torch.Tensor, g: torch.Tensor) -> None:
    _event_scatter(_np(out_ptr), _np(out_post), _np(out_w), _np(x.contiguous()), _np(g), x.shape[-1])


def lif_update(v, g, drive, adapt, refrac, active, poisson_p, rnd, res, std_u, spikes, out_buf, rate, counts, P, flags: int) -> None:
    N = v.shape[-1]
    _lif(*[_np(t) for t in (v, g, drive, adapt, refrac, active, poisson_p, rnd, res, std_u, spikes, out_buf, rate, counts, P)],
         N, int(flags))


def optic_dr(v, bvec, dr) -> None:
    _optic_dr(_np(v), _np(bvec), _np(dr), v.shape[-1])


def optic_substep(v, adapt, dr, y, pr_in, spk_in, a, bvec, gain_rr: float, adapt_gain: float, a_ad: float) -> None:
    _optic_substep(*[_np(t.contiguous()) if t is y or t is pr_in or t is spk_in else _np(t)
                     for t in (v, adapt, dr, y, pr_in, spk_in, a, bvec)],
                   np.float32(gain_rr), np.float32(adapt_gain), np.float32(a_ad), v.shape[-1])


def trace_rays(world, o: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """world.World._trace on the packed CPU scene, one ray per loop iteration."""
    o = o.contiguous(); d = d.contiguous()
    light = torch.cat([world._lp, world._lc, world._amb]).numpy().astype(np.float64)
    out = np.empty((o.shape[0], 4), dtype=np.float32)
    f64 = lambda t, cols: t.detach().numpy().astype(np.float64).reshape(-1, cols)
    _trace(f64(o, 3), f64(d, 3), f64(world._sc, 3), f64(world._sr, 3), f64(world._blo, 3), f64(world._bhi, 3),
           f64(world._pp, 3), f64(world._pn, 3), f64(world._refl, 4), f64(world._refl2, 4), f64(world._emit, 4),
           world._pattern.numpy().astype(np.int64), world._pscale.numpy().astype(np.float64), light, float(world.detail), out)
    return torch.from_numpy(out)


def _selfcheck():
    """python -m flybrain.cpu : kernels match their torch counterparts."""
    rng = np.random.default_rng(0)
    M = sp.random(300, 200, density=0.05, format="csr", random_state=0, dtype=np.float32)
    x = rng.standard_normal((3, 200)).astype(np.float32)
    got = NumbaCSR(M).matvec(torch.from_numpy(x)).numpy()
    assert np.allclose(got, (M @ x.T).T, atol=1e-5)
    from .world import make_room
    w, info = make_room(0)
    w.device = torch.device("cpu"); w._pack()
    n = 4000
    o = torch.tensor(np.tile([[-0.5, 0.05, info["table_top_z"] + 0.0012]], (n, 1)), dtype=torch.float32)
    dd = torch.from_numpy(rng.standard_normal((n, 3)).astype(np.float32)); dd = dd / dd.norm(dim=1, keepdim=True)
    a, b = w._trace(o, dd), trace_rays(w, o, dd)
    err = (a - b).abs().max().item()
    assert err < 1e-3, err
    print(f"numba CSR ok; ray tracer max |torch - numba| = {err:.2e}")


if __name__ == "__main__":
    _selfcheck()
