from itertools import permutations
from typing import List, Tuple
import numpy as np


def get_partitions(size: int) -> List[Tuple[int, ...]]:
    """
    Calculates all possible bipartite partitions as list of binary tuples.
    N.B - shouldn't be used for size >= 10 (becomes slow).

    :param size: number of nodes in graph.
    :return: list of binary tuples.
    """
    all_partitions = []
    for i in range(size + 1):
        current_partition = set(list(permutations([1 for j in range(i)] + [0 for k in range(size - i)])))
        all_partitions.extend(current_partition)
    return all_partitions


def qubo_min_cost_partition(partitions: List[Tuple[int, ...]],
                            Q_mat: np.ndarray,
                            nr_states: int = 3) -> Tuple[float, Tuple[int, ...]]:
    """
    Given a list of partitions and a square matrix 'Q', determines minimal cost and
    corresponding partition, for a QUBO cost function of type x^T*Q*x.

    :param partitions: list of binary tuples.
    :param Q_mat: np.ndarray
    :param nr_states: integer determining number of best states to return
    :return: minimal cost, corresponding partition
    """
    costs = []
    for _partition in partitions:
        x = np.array(_partition)
        costs.append(x.T @ Q_mat @ x)
    _i = np.argmin(costs)
    return costs[_i], partitions[_i]