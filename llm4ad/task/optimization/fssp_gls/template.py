template_program = '''
import numpy as np
def get_matrix_and_jobs(current_sequence: list, time_matrix: np.ndarray, m: int, n: int) -> tuple:
    """Modify the processing-time matrix and select jobs to perturb.

    Args:
        current_sequence: current permutation of job indices (list of n ints)
        time_matrix:      n*m matrix of processing times (numpy array)
        m:                number of machines
        n:                number of jobs
    Return:
        new_matrix:   modified n*m processing-time matrix (numpy array)
        perturb_jobs: list of 2-5 job indices to apply targeted local search on
    """
    return time_matrix.copy(), list(range(min(3, n)))
'''

task_description = (
    "Given a flow-shop scheduling problem with n jobs and m machines, "
    "design a novel guided local search perturbation strategy. "
    "At each iteration the strategy modifies the processing-time matrix "
    "to expose bottleneck jobs and returns a short list of jobs to perturb "
    "via targeted local search. "
    "The goal is to minimise the final makespan."
)
