"""
fast_F.py — costruzione di F senza scansione di |X|.

Osservazione di base: all_tuples_of() enumera X in ordine lessicografico
(meshgrid indexing='ij' + ravel), quindi l'indice flat i e' la codifica
mixed-radix della tupla:  i = sum_k x_k * stride_k,  stride_k = prod_{l>k} d_l.

DENSO. Per F di shape (|X|, m) in C-order, F.reshape(d_1,...,d_K, m) e' una
VISTA e il reshape *e'* la decodifica mixed-radix. Il vincolo j diventa una
assegnazione su slice. Nessuna aritmetica scritta a mano: la correttezza
poggia sulla semantica di reshape di NumPy.

  Attenzione al layout: riempire un buffer (m,|X|) C-ordered e trasporre da
  una F Fortran-ordered. La F e' bitwise identica, ma BLAS somma in ordine
  diverso e F @ lam differisce all'ultimo bit, il che rompe l'invarianza
  bitwise a valle. Qui si usa il C-order, identico al riferimento.

SPARSO. Gli indici che soddisfano il vincolo j si generano aritmeticamente
(base + sumset delle progressioni degli attributi liberi) in O(nnz_j), con
nnz_j = |X| / prod_{k in S_j} d_k noto in forma chiusa. Si prealloca, si
riempie in place e si passa a csr_matrix((data,(rows,cols))) nello stesso
ordine del riferimento: stessi input COO => stessa CSR bit per bit.

Verificato: indptr, indices, data e i prodotti F@lam / F.T@p sono identici a
build_indicator_matrix e build_indicator_matrix_sparse.
"""

import numpy as np


# ------------------------------------------------------------------ denso
def build_F_fast(cs, dtype=np.float64):
    """
    F[i, j] = 1[ x^(i)_{S_j} = v_j ], shape (|X|, m), C-contiguous.
    Equivalente bitwise a cs.build_indicator_matrix(all_tuples), senza
    scandire le |X| righe e senza richiedere all_tuples.
    """
    ds = [int(d) for d in cs.domain_sizes]
    X = int(np.prod(ds))
    F = np.zeros((X, cs.m), dtype=dtype)
    V = F.reshape(*ds, cs.m)              # vista, non copia
    full = slice(None)
    for j in range(cs.m):
        sl = [full] * cs.K
        for a, v in zip(cs.attrs_list[j], cs.vals_list[j]):
            sl[int(a)] = int(v)
        V[tuple(sl) + (j,)] = 1
    return F


# ------------------------------------------------------------------ sparso
def _strides(ds):
    K = len(ds)
    st = np.ones(K, dtype=np.int64)
    for k in range(K - 2, -1, -1):
        st[k] = st[k + 1] * ds[k + 1]
    return st


def constraint_indices(cs, j, ds=None, st=None):
    """
    Indici flat ORDINATI delle righe che soddisfano il vincolo j.

    Da {0}, per ogni attributo k in ordine crescente si somma il contributo
    fissato v_k*stride_k (se k in S_j) oppure l'intera progressione
    {0, s_k, ..., (d_k-1) s_k}.

    L'output e' ordinato: gli elementi gia' accumulati (attributi < k) hanno
    gap minimo stride_{k-1} = d_k*stride_k, mentre il nuovo contributo copre
    al piu' (d_k-1)*stride_k < stride_{k-1}. Nessuna sovrapposizione.

    Costo e memoria transitoria: O(nnz_j), con nnz_j = |X| / prod_{k in S_j} d_k.
    """
    if ds is None:
        ds = np.asarray(cs.domain_sizes, dtype=np.int64)
    if st is None:
        st = _strides(ds)
    fixed = {int(a): int(v) for a, v in zip(cs.attrs_list[j], cs.vals_list[j])}
    idx = np.zeros(1, dtype=np.int64)
    for k in range(len(ds)):
        if k in fixed:
            idx = idx + fixed[k] * st[k]
        else:
            idx = (idx[:, None]
                   + np.arange(ds[k], dtype=np.int64) * st[k]).ravel()
    return idx


def nnz_expected(cs, j, ds=None):
    """nnz_j = |X| / prod_{k in S_j} d_k, in forma chiusa."""
    if ds is None:
        ds = np.asarray(cs.domain_sizes, dtype=np.int64)
    return int(np.prod(ds)) // int(np.prod(ds[cs.attrs_list[j]]))


def build_F_fast_sparse(cs, dtype=np.float64, verbose=True, row_dtype=None):
    """
    F in CSR, equivalente bitwise a cs.build_indicator_matrix_sparse(all_tuples).

    Prealloca rows/cols/data dai nnz_j in forma chiusa e riempie in place:
    niente lista di m array, niente concatenazione, niente passaggio per CSC.
    row_dtype scende a int32 quando |X| < 2^31 (dimezza la memoria degli
    indici; verificato che non cambia la CSR risultante).
    """
    from scipy.sparse import csr_matrix

    ds = np.asarray(cs.domain_sizes, dtype=np.int64)
    st = _strides(ds)
    X = int(np.prod(ds))
    m = cs.m

    nnz_j = np.array([nnz_expected(cs, j, ds) for j in range(m)], dtype=np.int64)
    off = np.zeros(m + 1, dtype=np.int64)
    if m:
        np.cumsum(nnz_j, out=off[1:])
    nnz = int(off[-1])

    if row_dtype is None:
        row_dtype = np.int32 if X <= np.iinfo(np.int32).max else np.int64

    rows = np.empty(nnz, dtype=row_dtype)
    cols = np.empty(nnz, dtype=np.int32)
    for j in range(m):
        a, b = off[j], off[j + 1]
        rows[a:b] = constraint_indices(cs, j, ds, st)
        cols[a:b] = j
    data = np.ones(nnz, dtype=dtype)

    F = csr_matrix((data, (rows, cols)), shape=(X, m))
    if verbose:
        dens = nnz / (X * m) if m else 0.0
        mb = (F.data.nbytes + F.indices.nbytes + F.indptr.nbytes) / 1e6
        print(f"  [sparse] F: {X:,}x{m} | nnz={nnz:,} "
              f"({dens*100:.2f}% denso) | ~{mb:.1f} MB")
    return F
