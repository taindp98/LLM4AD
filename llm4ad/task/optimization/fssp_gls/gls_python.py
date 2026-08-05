"""Flow-Shop Scheduling — Guided Local Search engine (pure-Python REFERENCE).

REFERENCE ONLY — kept for tracing / cross-checking. The ACTIVE engine imported by
``evaluation.py`` is the numba-optimized ``gls.py`` (identical algorithm, ~100-300x
faster). This pure-Python version is too slow to run the EoH 60s / 1000-iteration
budget on large instances (one local_search sweep ~11s at n=100/m=20).

Pure-Python port of EoH's ``fssp_gls/prob.py`` GLS engine, kept separate from the
``Evaluation`` wrapper the same way ``tsp_gls_2O/gls.py`` is. The LLM designs
``get_matrix_and_jobs(current_sequence, time_matrix, m, n) -> (new_matrix,
perturb_jobs)`` which the GLS loop uses to (a) modify the processing-time matrix
to expose bottlenecks and (b) pick 2-5 jobs to perturb via targeted local search.
Cost = makespan (lower = better).
"""

import time
import random
from typing import Optional


def makespan(order, tasks, machines_val):
    """Permutation-flowshop makespan of a job ``order`` on ``machines_val`` machines."""
    times = [0.0] * machines_val
    for j in order:
        times[0] += tasks[j][0]
        for k in range(1, machines_val):
            if times[k] < times[k - 1]:
                times[k] = times[k - 1]
            times[k] += tasks[j][k]
    return max(times)


def local_search(sequence, cmax_old, tasks, machines_val):
    """One sweep of swap + insert moves (best-accept within the sweep)."""
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


def local_search_perturb(sequence, cmax_old, tasks, machines_val, job):
    """Targeted swap + insert moves restricted to the perturbed ``job`` list."""
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
    tab = [0.0] * tasks_val
    tab1 = [0] * tasks_val
    for j in range(tasks_val):
        for k in range(machines_val):
            tab[j] += tasks[j][k]
    place = 0
    it = 0
    while it != tasks_val:
        max_time = -1.0
        for i in range(tasks_val):
            if max_time < tab[i]:
                max_time = tab[i]
                place = i
        tab[place] = -1.0
        tab1[it] = place
        it += 1
    return tab1


def neh(tasks, machines_val, tasks_val):
    """NEH constructive heuristic; returns (sequence, makespan)."""
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


def gls(tasks_val, tasks, machines_val, time_max, iter_max, heuristic,
        seed: Optional[int] = None):
    """Guided local search for one flow-shop instance; returns the best makespan.

    ``heuristic`` is the LLM-designed ``get_matrix_and_jobs(current_sequence,
    time_matrix, m, n) -> (new_matrix, perturb_jobs)``. A heuristic returning
    <= 1 job, or any crash, yields the 1e10 penalty so the race rejects/penalises
    it (matching EoH's convention).

    ``seed``: the GLS engine itself uses no RNG (NEH + deterministic local
    search), but the LLM-generated ``heuristic`` runs as plain Python and may call
    ``np.random`` / ``random``. Seeding here makes such a heuristic reproducible
    for a given (instance, seed) task (mirrors ``tsp_gls_2O.gls``). ``seed=None``
    keeps the unseeded behavior.
    """
    if seed is not None:
        import numpy as np
        np.random.seed(seed)
        random.seed(seed)

    cmax_best = 1E10
    try:
        pi, cmax = neh(tasks, machines_val, tasks_val)
        n = len(pi)

        pi_best = pi[:]
        cmax_best = cmax
        n_itr = 0
        time_start = time.time()
        while time.time() - time_start < time_max and n_itr < iter_max:
            piprim = local_search(pi, cmax, tasks, machines_val)

            pi = piprim
            cmax = makespan(pi, tasks, machines_val)

            if cmax < cmax_best:
                pi_best = pi[:]
                cmax_best = cmax

            tasks_perturb, jobs = heuristic(pi[:], tasks.copy(), machines_val, n)
            jobs = list(jobs)
            jobs = [j for j in jobs if 0 <= j < n]

            if len(jobs) <= 1:
                return 1E10
            if len(jobs) > 5:
                jobs = jobs[:5]

            cmax = makespan(pi, tasks_perturb, machines_val)

            pi = local_search_perturb(pi, cmax, tasks_perturb, machines_val, jobs)

            n_itr += 1
            if n_itr % 50 == 0:
                pi = pi_best[:]
                cmax = cmax_best

    except Exception:
        cmax_best = 1E10

    return cmax_best
