import numpy as np
import numpy.typing as npt


class GetData():
    """Generate synthetic Flow-Shop Scheduling instances (EoH paper training set).

    Per the EoH paper's FSSP-GLS training setup: ``n_instance`` random instances,
    each with a fixed number of jobs (``n_jobs`` = 50) and a per-instance number
    of machines drawn uniformly from ``[m_low, m_high]`` = [2, 20]; processing
    times are drawn i.i.d. from Uniform[0, 1]. Mirrors the generation style of
    ``tsp_gls_2O.get_instance`` (fixed seed 2024 for reproducibility)."""

    def __init__(self, n_instance, n_jobs=50, m_low=2, m_high=20):
        self.n_instance = n_instance
        self.n_jobs = n_jobs
        self.m_low = m_low
        self.m_high = m_high

    def generate_instances(self):
        np.random.seed(2024)
        instance_data = []
        for _ in range(self.n_instance):
            # machines vary 2..20 (inclusive) per instance
            n_machines = int(np.random.randint(self.m_low, self.m_high + 1))
            processing_times = np.random.random((self.n_jobs, n_machines))  # U[0,1]
            instance_data.append(FSSPInstance(processing_times))
        return instance_data


class FSSPInstance:
    """One permutation Flow-Shop instance: an (n_jobs x n_machines) matrix of
    processing times. ``tasks_val`` / ``machines_val`` name-match the FSSP-GLS
    engine's variables (n jobs, m machines)."""

    def __init__(self, processing_times: npt.NDArray[np.float64]) -> None:
        self.tasks = np.asarray(processing_times, dtype=float)
        self.tasks_val = self.tasks.shape[0]      # number of jobs (n)
        self.machines_val = self.tasks.shape[1]   # number of machines (m)
