from random import shuffle
from itertools import combinations
import numpy as np

def solve_minfbarcset(distance_matrix):
    nvert = len(distance_matrix)
    min_order = tuple(range(nvert))
    min_fb = 100000

    def minfbarcset_helper(curr_order, used, curr_fb):
        nonlocal min_order, min_fb
        if len(curr_order) == nvert:
            if curr_fb <= min_fb:
                min_fb = curr_fb
                min_order = curr_order
                return
        else:
            for i, _ in enumerate(distance_matrix):
                if used[i]:
                    continue
                curr_order = curr_order + (i,)
                used[i] = True
                new_fb = curr_fb
                for j in range(len(used)):
                    if not used[j]:
                        new_fb += distance_matrix[j][i]
                if new_fb > min_fb:
                    curr_order = curr_order[:-1]
                    used[i] = False
                    continue
                minfbarcset_helper(curr_order, used.copy(), new_fb)
                curr_order = curr_order[:-1]
                used[i] = False
    minfbarcset_helper((), [False]*nvert, 0)
    return min_order

def compute_minfbarcset_score(distance_matrix, order):
    nvert = len(distance_matrix)
    updfb = 0
    for i in range(nvert):
        prefix = order[i]
        for j in range(i+1, nvert):
            curr = order[j]
            updfb += distance_matrix[curr][prefix]
    return updfb

def solve_minfbarcset_local_search(distance_matrix, include_sentinel=False):
    nvert = len(distance_matrix)
    min_order = list(range(nvert))
    np.random.shuffle(min_order)
    start_idx, end_idx = 0, nvert
    if include_sentinel:
        min_order = list(range(nvert-2))
        np.random.shuffle(min_order)
        min_order = [0] + [x+1 for x in min_order] + [nvert-1]
        start_idx, end_idx = 1, nvert-1
    # print(nvert, start_idx, end_idx)
    # print(min_order)
    possible_swaps = list(combinations(range(start_idx, end_idx),2))
    # min_order = [10,11,0,8,9,1,2,3,4,5,6,7]
    min_fb = 100000
    while True:
        shuffle(possible_swaps)
        updated = False
        for r1, r2 in possible_swaps:
            min_order[r1], min_order[r2] = min_order[r2], min_order[r1]
            updfb = compute_minfbarcset_score(distance_matrix, min_order)
            if updfb < min_fb:
                min_fb = updfb
                updated = True
            else:
                min_order[r1], min_order[r2] = min_order[r2], min_order[r1]
        if not updated:
            break
    if include_sentinel:
        return [x-1 for x in min_order[1:-1]]
    return min_order