import numpy as np
from random import shuffle
from itertools import combinations
from tsp_concorde import solve_tsp

def solve_tsp_local_search(distance_matrix, include_sentinel=False):
    nvert = len(distance_matrix)
    min_order = list(range(nvert))
    np.random.shuffle(min_order)
    start_idx, end_idx = 0, nvert
    if include_sentinel:
        start_idx, end_idx = 1, nvert-1
    possible_swaps = list(combinations(range(start_idx, end_idx),2))
    # min_order = [10,11,0,8,9,1,2,3,4,5,6,7]
    max_path = -100000
    while True:
        shuffle(possible_swaps)
        updated = False
        for r1, r2 in possible_swaps:
            min_order[r1], min_order[r2] = min_order[r2], min_order[r1]
            path = 0
            for i in range(1,nvert):
                prefix = min_order[i]
                curr = min_order[i-1]
                path += distance_matrix[curr][prefix]
            if path > max_path:
                max_path = path
                updated = True
            else:
                min_order[r1], min_order[r2] = min_order[r2], min_order[r1]
        if not updated:
            break
    return min_order
