"""Flow-Shop Scheduling — Guided Local Search engine (numba-optimized).

Numba (`@nb.njit`) reimplementation of ``gls_python.py`` (the faithful pure-Python
port of EoH's ``fssp_gls/prob.py``), mirroring how ``tsp_gls_2O/gls.py`` JITs its
local search. The per-iteration ``local_search`` is O(n^3 * m); in pure Python one
sweep costs ~11 s at n=100/m=20, so only ~2-5 GLS iterations fit in the EoH 60 s
budget (vs. the paper's ~1000) and large instances barely improve over NEH. JITing
makespan + the swap/insert local searches gives ~100-300x speedup so the full 1000
iterations complete in 60 s at all sizes — matching the EoH Table 3 setting.

Same public API as before: ``gls(tasks_val, tasks, machines_val, time_max,
iter_max, heuristic, seed=None) -> best makespan`` and ``makespan(...)``. The outer
GLS loop stays Python (it calls the LLM-generated ``get_matrix_and_jobs``, which
cannot be JIT'd); only the hot inner kernels are compiled. Behaviour is identical
to ``gls_python.py`` (same moves, same accept rule, same 50-iteration restart);
``gls_python.py`` is kept for reference / cross-checking.
"""

import time
import random
from typing import Optional

import numpy as np
import numba as nb

_CACHE = True


@nb.njit(nb.float64(nb.int64[:], nb.float64[:, :], nb.int64), nogil=True, cache=_CACHE)
def _makespan(order, tasks, m):
    """Permutation-flowshop makespan of ``order`` (job indices) on ``m`` machines."""
    times = np.zeros(m)
    n = order.shape[0]
    for idx in range(n):
        j = order[idx]
        times[0] += tasks[j, 0]
        for k in range(1, m):
            if times[k] < times[k - 1]:
                times[k] = times[k - 1]
            times[k] += tasks[j, k]
    # times is non-decreasing, so the makespan is times[m-1]; keep a max for safety.
    best = times[0]
    for k in range(1, m):
        if times[k] > best:
            best = times[k]
    return best


@nb.njit(nb.int64[:](nb.int64[:], nb.int64, nb.int64), nogil=True, cache=_CACHE)
def _relocate(arr, frm, to):
    """Return a copy of ``arr`` with the element at position ``frm`` removed and
    re-inserted at position ``to`` in the resulting (n-1)-length sequence — the
    array equivalent of ``lst.remove(val); lst.insert(to, val)``."""
    n = arr.shape[0]
    val = arr[frm]
    tmp = np.empty(n - 1, dtype=np.int64)
    k = 0
    for idx in range(n):
        if idx != frm:
            tmp[k] = arr[idx]
            k += 1
    out = np.empty(n, dtype=np.int64)
    for idx in range(to):
        out[idx] = tmp[idx]
    out[to] = val
    for idx in range(to, n - 1):
        out[idx + 1] = tmp[idx]
    return out


@nb.njit(nb.int64[:](nb.float64[:, :], nb.int64, nb.int64), nogil=True, cache=_CACHE)
def _sum_and_order(tasks, n, m):
    """Order jobs by descending total processing time (NEH ordering)."""
    tab = np.zeros(n)
    for j in range(n):
        for k in range(m):
            tab[j] += tasks[j, k]
    order = np.empty(n, dtype=np.int64)
    for it in range(n):
        max_time = 1.0
        place = 0
        for i in range(n):
            if max_time < tab[i]:
                max_time = tab[i]
                place = i
        tab[place] = 1.0
        order[it] = place
    return order


@nb.njit(nb.types.Tuple((nb.int64[:], nb.float64))(nb.float64[:, :], nb.int64, nb.int64),
         nogil=True, cache=_CACHE)
def _neh(tasks, n, m):
    """NEH constructive heuristic; returns (sequence, makespan)."""
    order = _sum_and_order(tasks, n, m)
    cur = np.empty(1, dtype=np.int64)
    cur[0] = order[0]
    for i in range(1, n):
        v = order[i]
        L = cur.shape[0]
        best_seq = np.empty(L + 1, dtype=np.int64)
        min_cmax = 1e18
        for j in range(0, L + 1):
            tmp = np.empty(L + 1, dtype=np.int64)
            for idx in range(j):
                tmp[idx] = cur[idx]
            tmp[j] = v
            for idx in range(j, L):
                tmp[idx + 1] = cur[idx]
            c = _makespan(tmp, tasks, m)
            if c < min_cmax:
                min_cmax = c
                best_seq = tmp.copy()
        cur = best_seq
    return cur, _makespan(cur, tasks, m)


@nb.njit(nb.int64[:](nb.int64[:], nb.float64, nb.float64[:, :], nb.int64),
         nogil=True, cache=_CACHE)
def _local_search(seq, cmax_old, tasks, m):
    """One sweep of swap (all position pairs) + insert (relocate job value v=1..n-1
    to each position), best-accept, on the ORIGINAL times. Matches gls_python."""
    cur = seq.copy()
    n = cur.shape[0]
    # swap phase: exchange positions i, j (explicit temp — numba tuple-swap of
    # array elements is not reliably an atomic swap)
    for i in range(n):
        for j in range(i + 1, n):
            tmp_v = cur[i]; cur[i] = cur[j]; cur[j] = tmp_v
            c = _makespan(cur, tasks, m)
            if c < cmax_old:
                cmax_old = c
            else:
                tmp_v = cur[i]; cur[i] = cur[j]; cur[j] = tmp_v   # revert
    # insert phase: relocate job VALUE v to position p (job 0 / position 0 skipped)
    for v in range(1, n):
        for p in range(1, n):
            pos = 0
            for idx in range(n):
                if cur[idx] == v:
                    pos = idx
                    break
            cand = _relocate(cur, pos, p)
            c = _makespan(cand, tasks, m)
            if c < cmax_old:
                cur = cand
                cmax_old = c
    return cur


@nb.njit(nb.int64[:](nb.int64[:], nb.float64, nb.float64[:, :], nb.int64, nb.int64[:]),
         nogil=True, cache=_CACHE)
def _local_search_perturb(seq, cmax_old, tasks, m, job):
    """Targeted swap + insert restricted to the perturb ``job`` list, on the
    PERTURBED times. Faithful to gls_python: in the swap phase job values are used
    as POSITIONS; in the insert phase as job VALUES (the original EoH semantics)."""
    cur = seq.copy()
    n = cur.shape[0]
    for a in range(job.shape[0]):
        i = job[a]
        for j in range(i + 1, n):
            tmp_v = cur[i]; cur[i] = cur[j]; cur[j] = tmp_v
            c = _makespan(cur, tasks, m)
            if c < cmax_old:
                cmax_old = c
            else:
                tmp_v = cur[i]; cur[i] = cur[j]; cur[j] = tmp_v
    for a in range(job.shape[0]):
        v = job[a]
        for p in range(1, n):
            pos = 0
            for idx in range(n):
                if cur[idx] == v:
                    pos = idx
                    break
            cand = _relocate(cur, pos, p)
            c = _makespan(cand, tasks, m)
            if c < cmax_old:
                cur = cand
                cmax_old = c
    return cur


def makespan(order, tasks, machines_val):
    """Public makespan wrapper (accepts any int sequence / float matrix)."""
    order = np.ascontiguousarray(order, dtype=np.int64)
    tasks = np.ascontiguousarray(tasks, dtype=np.float64)
    return float(_makespan(order, tasks, machines_val))


def gls(tasks_val, tasks, machines_val, time_max, iter_max, heuristic,
        seed: Optional[int] = None):
    """Guided local search for one flow-shop instance; returns the best makespan.

    Drop-in replacement for ``gls_python.gls`` — identical algorithm, numba-JIT'd
    inner kernels. ``heuristic`` is the LLM-designed ``get_matrix_and_jobs(
    current_sequence, time_matrix, m, n) -> (new_matrix, perturb_jobs)``; a
    heuristic returning <= 1 valid job, or any crash, yields the 1e10 penalty.
    ``seed`` seeds np.random/random so a stochastic heuristic is reproducible.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    tasks = np.ascontiguousarray(tasks, dtype=np.float64)
    machines_val = int(machines_val)
    cmax_best = 1e10
    try:
        pi, cmax = _neh(tasks, int(tasks_val), machines_val)
        n = pi.shape[0]
        pi_best = pi.copy()
        cmax_best = cmax
        n_itr = 0
        time_start = time.time()
        while time.time() - time_start < time_max and n_itr < iter_max:
            pi = _local_search(pi, cmax, tasks, machines_val)
            cmax = _makespan(pi, tasks, machines_val)
            if cmax < cmax_best:
                pi_best = pi.copy()
                cmax_best = cmax

            # LLM heuristic (pure Python): pass the sequence as a list (as in EoH).
            tasks_perturb, jobs = heuristic(list(pi), tasks.copy(), machines_val, n)
            jobs = np.asarray(list(jobs), dtype=np.int64)
            # keep only valid job indices (guards numba against a malformed heuristic)
            jobs = jobs[(jobs >= 0) & (jobs < n)]

            if jobs.shape[0] <= 1:
                return 1E10
            if jobs.shape[0] > 5:
                jobs = jobs[:5]

            tasks_perturb = np.ascontiguousarray(tasks_perturb, dtype=np.float64)
            cmax = _makespan(pi, tasks_perturb, machines_val)
            pi = _local_search_perturb(pi, cmax, tasks_perturb, machines_val, jobs)

            n_itr += 1
            if n_itr % 50 == 0:
                pi = pi_best.copy()
                cmax = cmax_best

    except Exception:
        cmax_best = 1E10

    return cmax_best
