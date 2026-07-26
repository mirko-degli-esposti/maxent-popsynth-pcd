"""
test_F.py — suite di equivalenza fra build_indicator_matrix (riferimento) e
build_F_fast / build_F_fast_sparse.

    python test_F.py                          # solo test sintetici
    python test_F.py cs_K7C.json              # + equivalenza sul CS reale
    python test_F.py cs_K9C.json  --sparse-only   # salta la densa (110 GB a K9C)
    python test_F.py cs_K10C.json --no-ref        # solo invarianti, nessun
                                                  #   riferimento costruito
    python test_F.py cs_K7C.json --min-alpha 2e-4

T1  / T1b : equivalenza esatta su CS random e degeneri
T2        : equivalenza esatta sul CS reale (denso e sparso, incl. dtype/shape)
T3        : invarianza bitwise a valle (F@lam, p@F, F[:,mask].sum)
T4/T5/T6  : invarianti che NON richiedono F_old (usabili a qualunque scala)
"""

import sys
import json
import time
import numpy as np

sys.path.insert(0, ".")
from fast_F import build_F_fast, build_F_fast_sparse, nnz_expected  # noqa: E402

rng = np.random.default_rng(0)


def import_constraint_set():
    import os, glob, importlib
    for base in ["~/progetti/maxent-popsynth-pcd", "/content/maxent-popsynth-pcd", "."]:
        hits = glob.glob(f"{os.path.expanduser(base)}/**/constraint_set.py",
                         recursive=True)
        if hits:
            moddir = os.path.dirname(hits[0])
            sys.path.insert(0, os.path.dirname(moddir) or ".")
            pkg = os.path.basename(moddir)
            mod = (importlib.import_module(f"{pkg}.constraint_set")
                   if pkg not in ("", ".") else
                   importlib.import_module("constraint_set"))
            return mod.ConstraintSet
    from constraint_set import ConstraintSet
    return ConstraintSet


ConstraintSet = import_constraint_set()


def ref_dense(cs, at):
    """Riferimento denso: usa _ref se esiste (dopo lo stadio 2), altrimenti
    il metodo originale (prima dello stadio 2)."""
    f = getattr(cs, "build_indicator_matrix_ref", None)
    return f(at) if f else cs.build_indicator_matrix(at)


def ref_sparse(cs, at):
    f = getattr(cs, "build_indicator_matrix_sparse_ref", None)
    return f(at) if f else cs.build_indicator_matrix_sparse(at)


def all_tuples_of(ds):
    g = np.meshgrid(*[np.arange(d) for d in ds], indexing="ij")
    return np.stack([x.ravel() for x in g], axis=1).astype(np.int32)


def load_cs_min(path, min_alpha=0.0, eps=0.0):
    """Replica la logica di filtro di fit_cs.load_cs (senza il filtro blocchi)."""
    spec = json.load(open(path))
    cs = ConstraintSet(spec["domain_sizes"])
    for c in spec["constraints"]:
        a = c["alpha"]
        if a <= 0:
            if eps > 0:
                a = eps
            else:
                continue
        elif a < min_alpha:
            continue
        cs.add(c["attrs"], c["vals"], a)
    return spec, cs


# ------------------------------------------------------------------ T1
def T1_random(n=200):
    bad = 0
    for s in range(n):
        r = np.random.default_rng(s)
        K = int(r.integers(2, 8))
        ds = r.integers(2, 5, size=K)
        cs = ConstraintSet(ds)
        for _ in range(int(r.integers(1, 40))):
            ar = int(r.integers(1, K + 1))
            attrs = r.choice(K, size=ar, replace=False)
            cs.add(attrs, [int(r.integers(ds[a])) for a in attrs],
                   float(r.uniform(1e-6, 0.5)))
        at = all_tuples_of(cs.domain_sizes)
        F_old, F_new = ref_dense(cs, at), build_F_fast(cs)
        if not (np.array_equal(F_old, F_new)
                and F_old.shape == F_new.shape
                and F_old.dtype == F_new.dtype
                and F_old.flags.c_contiguous == F_new.flags.c_contiguous):
            bad += 1
            print(f"  MISMATCH seed={s} K={cs.K} m={cs.m}")
    print(f"T1  {n} CS random (K 2-7, arita' 1..K, duplicati): "
          f"{'OK' if bad == 0 else f'{bad} FALLITI'}")
    return bad == 0


def T1b_degenerate():
    ok = True
    cs = ConstraintSet([2, 3, 2]); cs.add([0, 1, 2], [1, 2, 0], .1)
    F = build_F_fast(cs)
    ok &= np.array_equal(ref_dense(cs, all_tuples_of([2, 3, 2])), F)
    ok &= int(F.sum()) == 1                                   # arita' = K
    cs = ConstraintSet([3, 2, 4]); cs.add([2, 0], [3, 1], .2); cs.add([0, 2], [1, 3], .2)
    F = build_F_fast(cs)
    ok &= np.array_equal(ref_dense(cs, all_tuples_of([3, 2, 4])), F)
    ok &= np.array_equal(F[:, 0], F[:, 1])                    # attrs non ordinati
    ok &= build_F_fast(ConstraintSet([2, 2])).shape == (4, 0)  # m = 0
    cs = ConstraintSet([1, 5, 2]); cs.add([1], [3], .3)        # d_k = 1
    ok &= np.array_equal(ref_dense(cs, all_tuples_of([1, 5, 2])),
                         build_F_fast(cs))
    print(f"T1b degeneri (arita'=K, attrs non ordinati, m=0, d_k=1): "
          f"{'OK' if ok else 'FALLITO'}")
    return ok


# ------------------------------------------------------------------ T2 / T3
def T2_real(spec, cs, dense=True):
    at = all_tuples_of(spec["domain_sizes"])
    if dense:
        t0 = time.time(); F_old = ref_dense(cs, at); t_old = time.time() - t0
        t0 = time.time(); F_new = build_F_fast(cs);   t_new = time.time() - t0
        eq = (np.array_equal(F_old, F_new) and F_old.dtype == F_new.dtype
              and F_old.shape == F_new.shape)
    else:
        F_old = F_new = None
        t_old = t_new = float("nan")
        eq = True
    t0 = time.time(); S_old = ref_sparse(cs, at);  ts_old = time.time() - t0
    t0 = time.time(); S_new = build_F_fast_sparse(cs, verbose=False)
    ts_new = time.time() - t0
    eq_s = (S_old.format == S_new.format
            and np.array_equal(S_old.indptr, S_new.indptr)
            and np.array_equal(S_old.indices, S_new.indices)
            and np.array_equal(S_old.data, S_new.data))
    print(f"T2  {spec.get('livello','?')}: |X|={len(at):,} m={cs.m} | nnz={S_new.nnz:,}")
    if dense:
        print(f"    denso  equal={eq}   t_ref={t_old:7.2f}s t_new={t_new:6.2f}s "
              f"({t_old/max(t_new,1e-9):5.1f}x)")
    else:
        print(f"    denso  saltato (--sparse-only)")
    print(f"    sparso equal={eq_s}   t_ref={ts_old:7.2f}s t_new={ts_new:6.2f}s "
          f"({ts_old/max(ts_new,1e-9):5.1f}x)")
    return eq and eq_s, F_old, F_new, S_old, S_new


def T3_downstream(cs, F_old, F_new, S_old, S_new):
    lam = rng.normal(size=cs.m)
    p = rng.random(S_old.shape[0]); p /= p.sum()
    mask = rng.random(cs.m) < 0.3
    checks = []
    if F_old is not None:
        checks += [
            np.array_equal(F_old @ lam, F_new @ lam),
            np.array_equal(p @ F_old, p @ F_new),
            np.array_equal(F_old[:, mask].sum(axis=1),
                           F_new[:, mask].sum(axis=1)),
        ]
    checks += [
        np.array_equal(S_old @ lam, S_new @ lam),
        np.array_equal(p @ S_old, p @ S_new),
        np.array_equal(S_old.T @ p, S_new.T @ p),   # cio' che usa _phi_and_grad
    ]
    print(f"T3  bitwise a valle (prodotti densi e sparsi): "
          f"{'OK' if all(checks) else 'FALLITO'} {checks}")
    return all(checks)


# ------------------------------------------------------------------ T4/T5/T6
def T4_nnz(cs, F):
    colsum = np.asarray(F.sum(axis=0)).ravel()
    bad = [j for j in range(cs.m) if int(colsum[j]) != nnz_expected(cs, j)]
    print(f"T4  nnz_j == |X| / prod(d_k, k in S_j): "
          f"{'OK' if not bad else f'FALLITO su {len(bad)} colonne'}")
    return not bad


def T5_partition(cs, F):
    ds = np.asarray(cs.domain_sizes, dtype=np.int64)
    sigs = {}
    for j in range(cs.m):
        sigs.setdefault(tuple(cs.attrs_list[j].tolist()), []).append(j)
    ok, n = True, 0
    for sig, cols in sigs.items():
        if len(cols) != int(np.prod(ds[list(sig)])):
            continue                       # blocco incompleto (filtrato): salta
        n += 1
        s = np.asarray(F[:, cols].sum(axis=1)).ravel()
        if not np.all(s == 1.0):
            ok = False
            print(f"  blocco {sig} non partiziona: min={s.min()} max={s.max()}")
    print(f"T5  partizione di blocco ({n} blocchi completi): "
          f"{'OK' if ok else 'FALLITO'}")
    return ok


def T6_spotcheck(cs, F, n=None):
    ds = np.asarray(cs.domain_sizes, dtype=np.int64)
    X = int(np.prod(ds))
    if n is None:                      # il costo e' O(n * m): tara su m
        n = max(200, min(5000, 2_000_000 // max(cs.m, 1)))
    rows = np.unique(rng.integers(0, X, size=min(n, X)))
    for i in rows:
        x, r = np.zeros(cs.K, dtype=np.int64), int(i)
        for k in range(cs.K - 1, -1, -1):
            r, x[k] = divmod(r, int(ds[k]))
        want = np.array([float(np.all(x[cs.attrs_list[j]] == cs.vals_list[j]))
                         for j in range(cs.m)])
        got = np.asarray(F[i].todense()).ravel() if hasattr(F, "todense") else F[i]
        if not np.array_equal(want, got):
            print(f"  spot mismatch i={i}")
            print("T6  spot check per decodifica: FALLITO")
            return False
    print(f"T6  spot check per decodifica divmod ({len(rows)} righe x {cs.m}): OK")
    return True


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    args = sys.argv[1:]
    path = args[0] if args and not args[0].startswith("--") else None
    no_ref = "--no-ref" in args
    sparse_only = "--sparse-only" in args
    ma = float(args[args.index("--min-alpha") + 1]) if "--min-alpha" in args else 0.0
    n_spot = int(args[args.index("--n-spot") + 1]) if "--n-spot" in args else None
    eps = float(args[args.index("--eps") + 1]) if "--eps" in args else 0.0

    print("=" * 68)
    res = [T1_random(), T1b_degenerate()]

    if path:
        spec, cs = load_cs_min(path, ma, eps)
        X = int(np.prod(spec["domain_sizes"]))
        print(f"    [{path}] |X|={X:,} m={cs.m} min_alpha={ma} eps={eps} "
              f"| F densa = {X*cs.m*8/1e9:.2f} GB")
        if no_ref:
            F = build_F_fast_sparse(cs, verbose=True)
            res += [T4_nnz(cs, F), T5_partition(cs, F), T6_spotcheck(cs, F, n_spot)]
        else:
            ok, F_old, F_new, S_old, S_new = T2_real(spec, cs,
                                                     dense=not sparse_only)
            G = F_new if F_new is not None else S_new
            res += [ok, T3_downstream(cs, F_old, F_new, S_old, S_new),
                    T4_nnz(cs, G), T5_partition(cs, G), T6_spotcheck(cs, G, n_spot)]

    print("=" * 68)
    print("TUTTI I TEST PASSATI" if all(res) else "*** QUALCHE TEST FALLITO ***")
    sys.exit(0 if all(res) else 1)
