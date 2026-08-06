import os
import pathlib
import numpy as np
import numpy.typing as npt


class GetData():
    """Provides Flow-Shop Scheduling instances.

    Supports two modes:
    1. use_taillard=False (Synthetic): 
       Generates instances with processing times drawn from Uniform[0, 1].
    2. use_taillard=True (Original EoH):
       Loads the 64 text files from `packages/EoH/examples/fssp_gls/TrainingData`.
       These instances have integer processing times typically ~50, resulting
       in much larger makespans (e.g. ~1000s).
    """

    def __init__(self, n_instance, n_jobs=50, m_low=2, m_high=20, use_taillard=False):
        self.n_instance = n_instance
        self.n_jobs = n_jobs
        self.m_low = m_low
        self.m_high = m_high
        self.use_taillard = use_taillard
        
        # Point to the original EoH training data directory
        root = pathlib.Path(__file__).resolve().parent.parent.parent.parent.parent.parent.parent
        self.data_dir = root / 'packages' / 'EoH' / 'examples' / 'fssp_gls' / 'TrainingData'

    def _read_file(self, filename):
        with open(filename, "r") as file:
            tasks_val, machines_val = file.readline().split()
            tasks_val = int(tasks_val)
            machines_val = int(machines_val)

            tasks = np.zeros((tasks_val, machines_val))
            for i in range(tasks_val):
                tmp = file.readline().split()
                for j in range(machines_val):
                    tasks[i][j] = int(float(tmp[j * 2 + 1]))
        return tasks_val, machines_val, tasks

    def generate_instances(self):
        """Returns a list of FSSPInstance objects for the evaluator."""
        instance_data = []

        if self.use_taillard:
            # Load original EoH Taillard files (integer processing times)
            for i in range(1, self.n_instance + 1):
                filename = self.data_dir / f"{i}.txt"
                tasks_val, machines_val, tasks = self._read_file(filename)
                instance_data.append(FSSPInstance(tasks))
        else:
            # Generate synthetic U[0, 1] instances
            np.random.seed(2024)
            for _ in range(self.n_instance):
                n_machines = int(np.random.randint(self.m_low, self.m_high + 1))
                processing_times = np.random.random((self.n_jobs, n_machines))
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
