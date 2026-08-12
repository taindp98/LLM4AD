"""Flow-Shop Scheduling — Guided Local Search engine.

PORTED VERBATIM from EoH's ``packages/EoH/examples/fssp_gls_numba/prob.py`` so
this engine is the original EoH-0.1 implementation, not a reimplementation. The
numba-jitted kernels (``makespan``, ``local_search``, ``local_search_perturb``)
operate on reflected lists of ints and a float64 n*m matrix, exactly as EoH does;
``sum_and_order`` / ``neh`` and the outer GLS loop stay in Python (the latter has
to call back into the LLM-generated heuristic).

Public API kept for this repo's callers:
  * ``gls(tasks_val, tasks, machines_val, time_max, iter_max, heuristic, seed=None)``
    -> best makespan, or ``INVALID`` (1e10) if the heuristic is unusable.
  * ``makespan(order, tasks, machines_val)``
  * ``_makespan`` / ``_local_search`` / ``_neh`` — thin back-compat aliases (see
    the bottom of this file) used by ``src/analyses/test_{ls,neh}_fssp_taillard.py``.

KNOWN DIVERGENCE FROM THE PREVIOUS VERSION (deliberate, requested): EoH's
``sum_and_order`` uses ``1`` as its "already placed" sentinel instead of ``-1``.
On instances whose per-job total processing time can be <= 1 — i.e. the synthetic
Uniform[0,1] training instances from ``get_instance.GetData(use_pregenerated=
False)``, especially at small m — such a job is never selected, the NEH order
comes out with duplicates, and ``local_search``'s ``temp_seq.remove(i)`` raises,
so ``gls`` returns ``INVALID``. Integer-time instances (the EoH TrainingData
files and the Taillard test sets) are unaffected. Use
``GetData(..., use_pregenerated=True)`` for training if this bites.

``gls_python.py`` is kept alongside for reference / cross-checking.
"""

import time
import random
import warnings
from typing import Optional

import numpy as np

try:
    from numba import jit
    try:                                  # reflected lists warn on every call
        from numba.core.errors import (NumbaDeprecationWarning,
                                       NumbaPendingDeprecationWarning)
        warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
        warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
    except ImportError:
        pass
except ImportError:                       # numba optional: fall back to pure Python
    def jit(*args, **kwargs):
        return lambda f: f


@jit(nopython=True, cache=True)
def makespan(order, tasks, machines_val):
    times = [0.0] * machines_val
    for j in order:
        times[0] += tasks[j][0]
        for k in range(1, machines_val):
            if times[k] < times[k - 1]:
                times[k] = times[k - 1]
            times[k] += tasks[j][k]
    return max(times)


@jit(nopython=True, cache=True)
def local_search(sequence, cmax_old, tasks, machines_val):
    # One sweep of swap + insert moves (best-accept within the sweep).
    new_seq = sequence[:]
    for i in range(len(new_seq)):
        for j in range(i + 1, len(new_seq)):
            temp_seq = new_seq[:]
            temp_seq[i], temp_seq[j] = temp_seq[j], temp_seq[i]
            cmax = makespan(temp_seq, tasks, machines_val)
            if cmax < cmax_old:
                new_seq = temp_seq[:]
                cmax_old = cmax

    for i in range(1, len(new_seq)):
        for j in range(1, len(new_seq)):
            temp_seq = new_seq[:]
            temp_seq.remove(i)
            temp_seq.insert(j, i)
            cmax = makespan(temp_seq, tasks, machines_val)
            if cmax < cmax_old:
                new_seq = temp_seq[:]
                cmax_old = cmax

    return new_seq


@jit(nopython=True, cache=True)
def local_search_perturb(sequence, cmax_old, tasks, machines_val, job):
    # Targeted swap + insert moves restricted to the perturbed jobs.
    new_seq = sequence[:]
    for i in job:
        for j in range(i + 1, len(new_seq)):
            temp_seq = new_seq[:]
            temp_seq[i], temp_seq[j] = temp_seq[j], temp_seq[i]
            cmax = makespan(temp_seq, tasks, machines_val)
            if cmax < cmax_old:
                new_seq = temp_seq[:]
                cmax_old = cmax

    for i in job:
        for j in range(1, len(new_seq)):
            temp_seq = new_seq[:]
            temp_seq.remove(i)
            temp_seq.insert(j, i)
            cmax = makespan(temp_seq, tasks, machines_val)
            if cmax < cmax_old:
                new_seq = temp_seq[:]
                cmax_old = cmax

    return new_seq


def sum_and_order(tasks_val, machines_val, tasks):
    """Order jobs by descending total processing time (NEH ordering)."""
    tab = [0] * tasks_val
    tab1 = [0] * tasks_val
    for j in range(tasks_val):
        for k in range(machines_val):
            tab[j] += tasks[j][k]
    place = 0
    it = 0
    while it != tasks_val:
        max_time = 1
        for i in range(tasks_val):
            if max_time < tab[i]:
                max_time = tab[i]
                place = i
        tab[place] = 1
        tab1[it] = place
        it += 1
    return tab1


def neh(tasks, machines_val, tasks_val):
    """NEH constructive heuristic."""
    order = sum_and_order(tasks_val, machines_val, tasks)
    current_seq = [order[0]]
    for i in range(1, tasks_val):
        min_cmax = float("inf")
        best_seq = None
        for j in range(0, i + 1):
            tmp = current_seq[:]
            tmp.insert(j, order[i])
            cmax_tmp = makespan(tmp, tasks, machines_val)
            if min_cmax > cmax_tmp:
                best_seq = tmp
                min_cmax = cmax_tmp
        current_seq = best_seq
    return current_seq, makespan(current_seq, tasks, machines_val)


INVALID = 1E10


def gls(tasks_val, tasks, machines_val, time_max, iter_max, heuristic,
        seed: Optional[int] = None):
    """Guided local search for one flow-shop instance; returns best makespan.

    Returns INVALID (1E10) if the heuristic is unusable (raises, or returns
    fewer than two jobs to perturb), matching EoH-0.1.

    ``seed``: the GLS engine itself uses no RNG, but the LLM-generated
    ``heuristic`` is plain Python and may call ``np.random`` / ``random``;
    seeding here makes it reproducible per (instance, seed). ``seed=None``
    keeps EoH's hardcoded ``random.seed(2024)``.
    """
    cmax_best = INVALID
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    else:
        random.seed(2024)
    try:
        pi, cmax = neh(tasks, machines_val, tasks_val)
        n = len(pi)

        pi_best = pi
        cmax_best = cmax
        n_itr = 0
        time_start = time.time()
        while time.time() - time_start < time_max and n_itr < iter_max:
            piprim = local_search(pi, cmax, tasks, machines_val)

            pi = piprim
            cmax = makespan(pi, tasks, machines_val)

            if cmax < cmax_best:
                pi_best = pi
                cmax_best = cmax

            tasks_perturb, jobs = heuristic(pi, tasks.copy(), machines_val, n)
            # int() keeps the jitted local search typeable whether the heuristic
            # returns a Python list or a numpy array of indices.
            jobs = [int(j) for j in jobs]

            if len(jobs) <= 1:
                return INVALID
            if len(jobs) > 5:
                jobs = jobs[:5]

            # A single (C-contiguous, float64) layout keeps numba from
            # recompiling the local search for every heuristic output.
            tasks_perturb = np.ascontiguousarray(tasks_perturb, dtype=float)
            cmax = makespan(pi, tasks_perturb, machines_val)

            pi = local_search_perturb(pi, cmax, tasks_perturb, machines_val, jobs)

            n_itr += 1
            if n_itr % 50 == 0:
                pi = pi_best
                cmax = cmax_best

    except Exception as e:
        print(f"Error occurred: {e}")
        cmax_best = INVALID

    return cmax_best


# --------------------------------------------------------------------------- #
# Back-compat aliases for the previous numba-typed private kernels, kept so
# src/analyses/test_ls_fssp_taillard.py and test_neh_fssp_taillard.py keep
# working unchanged. Note the (tasks, n, m) argument order of ``_neh``.
# --------------------------------------------------------------------------- #

def _makespan(order, tasks, m):
    return float(makespan(list(order), tasks, int(m)))


def _local_search(seq, cmax_old, tasks, m):
    return local_search(list(seq), float(cmax_old), tasks, int(m))


def _neh(tasks, n, m):
    seq, cmax = neh(tasks, int(m), int(n))
    return seq, float(cmax)
