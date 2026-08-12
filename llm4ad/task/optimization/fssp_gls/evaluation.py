# name: str: FSSP_GLS_Evaluation
# Parameters:
# timeout_seconds: int: 700
# end
from __future__ import annotations

import math
from typing import Any
import numpy as np

from llm4ad.base import Evaluation
from llm4ad.task.optimization.fssp_gls.get_instance import GetData, FSSPInstance
from llm4ad.task.optimization.fssp_gls.template import template_program, task_description
from .gls import gls, makespan

__all__ = ['FSSP_GLS_Evaluation']

# Per-instance GLS budget (mirrors tsp_gls_2O's module-level perturbation_moves /
# iter_limit). NOTE: the standalone evaluate_program runs GLS on all 64 training
# instances, so a full program evaluation costs ~= 64 * min(time_max, iter_max
# runtime) seconds — tune these (or n_instance) if that is too slow. A racing/mab
# driver that scores instances individually uses its own per-instance timeout
# instead and calls solve_without_time directly.
time_max = 10.0
iter_max = 1000


def solve_without_time(inst: FSSPInstance, eva, seed=None) -> float:
    """Run the GLS engine with the LLM heuristic ``eva`` on one instance; return
    the best makespan (RAW cost, lower = better). A crash / non-finite makespan
    returns Inf so the caller can reject the configuration. Mirrors
    ``tsp_gls_2O.evaluation.solve_without_time``."""
    try:
        cost = gls(inst.tasks_val, inst.tasks, inst.machines_val,
                   time_max, iter_max, eva, seed=seed)
        cost = float(cost)
        if cost >= 1e9:
            return float("inf")
        return cost if math.isfinite(cost) else float("inf")
    except Exception:
        return float("inf")


def evaluate_without_time(instance_data, n_ins, eva) -> float:
    """Mean makespan over ``n_ins`` instances, NEGATED so higher = better (the
    llm4ad/method/eoh maximisation convention — same as tsp_gls_2O)."""
    objs = np.zeros(n_ins)
    for i in range(n_ins):
        objs[i] = solve_without_time(instance_data[i], eva)
    return -float(np.mean(objs))


class FSSP_GLS_Evaluation(Evaluation):
    """Evaluator for the permutation Flow-Shop Scheduling Problem via GLS.

    Training set (EoH paper): 64 synthetic instances, 50 jobs each, machines
    varying 2-20 per instance, processing times ~ Uniform[0, 1]. Fitness is the
    NEGATED average makespan across the 64 instances (higher = better)."""

    def __init__(self, **kwargs):
        super().__init__(
            template_program=template_program,
            task_description=task_description,
            use_numba_accelerate=False,
            timeout_seconds=700,   # covers 64 instances * per-instance time_max
        )

        self.n_instance = 64
        self.problem_size = 50   # jobs per instance (machines vary 2-20)
        getData = GetData(self.n_instance, n_jobs=self.problem_size)
        self._datasets = getData.generate_instances()

    def evaluate_program(self, program_str: str, callable_func: callable) -> Any | None:
        return evaluate_without_time(self._datasets, self.n_instance, callable_func)


if __name__ == '__main__':
    import numpy as np

    def get_matrix_and_jobs(current_sequence, time_matrix, m, n):
        """Baseline: penalise the bottleneck (highest-load) machine, perturb its
        longest-processing jobs."""
        machine_loads = time_matrix.sum(axis=0)
        critical_machine = int(np.argmax(machine_loads))
        new_matrix = time_matrix.copy()
        new_matrix[:, critical_machine] *= 1.2
        top_jobs = np.argsort(-time_matrix[:, critical_machine])[:3].tolist()
        return new_matrix, top_jobs

    # Fast smoke: one small instance, tiny GLS budget (not the full 64-instance run).
    from llm4ad.task.optimization.fssp_gls.get_instance import GetData as _GD
    inst = _GD(1, n_jobs=15).generate_instances()[0]
    cost = gls(inst.tasks_val, inst.tasks, inst.machines_val, 2.0, 20, get_matrix_and_jobs)
    print(f"FSSP smoke: jobs={inst.tasks_val} machines={inst.machines_val}  makespan={cost:.4f}")
