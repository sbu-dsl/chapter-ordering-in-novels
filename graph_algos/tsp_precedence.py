from random import shuffle
from itertools import combinations
import numpy as np

# Minimize
def compute_minfbarcset_score(distance_matrix, order):
    nvert = len(distance_matrix)
    updfb = 0
    for i in range(nvert):
        prefix = order[i]
        for j in range(i+1, nvert):
            curr = order[j]
            updfb += distance_matrix[curr][prefix]
    return updfb

# Maximize
def compute_tsp_score(distance_matrix, order):
    nvert = len(distance_matrix)
    path = 0
    aug_order = [0] + [x+1 for x in order] + [len(order)+1]
    for i in range(1,nvert):
        prefix = aug_order[i]
        curr = aug_order[i-1]
        path += distance_matrix[curr][prefix]
    return path

# Minimize
def compute_combined_score(order, tsp_matrix, precedence_matrix, tsp_stats, precedence_stats, alpha):
    precedence_score = compute_minfbarcset_score(precedence_matrix, order)
    tsp_score = compute_tsp_score(tsp_matrix, order)
    # print(feedback_score, tsp_score)
    tsp_zscore = (tsp_score - tsp_stats[0]) / tsp_stats[1]
    precedence_zscore = (precedence_score - precedence_stats[0]) / precedence_stats[1]
    return alpha*precedence_zscore - (1-alpha)*tsp_zscore


# tsp_stats: (avg, std) of TSP on random orders of TSP matrix
# precedence_stats: (avg, std) of feedback weight on random orders of precedence matrix
def solve_combined_score(augmented_tsp_matrix, precedence_matrix, tsp_stats, precedence_stats, alpha, initial_order=None):
    # print(alpha, initial_order)
    nvert = len(precedence_matrix)
    min_order = list(range(nvert))
    if initial_order:
        min_order = initial_order
    else:
        np.random.shuffle(min_order)
    start_idx, end_idx = 0, nvert
    possible_swaps = list(combinations(range(start_idx, end_idx),2))
    # min_order = [10,11,0,8,9,1,2,3,4,5,6,7]
    # min_order = [16,  9,  0, 10,  4, 14,  1, 13,  6,  8, 15,  3, 11,  7,  2, 12,  5]
    min_fb = compute_combined_score(min_order, augmented_tsp_matrix, precedence_matrix, tsp_stats, precedence_stats, alpha)
    # min_fb = 100000
    # print(min_fb)
    while True:
        shuffle(possible_swaps)
        updated = False
        for r1, r2 in possible_swaps:
            min_order[r1], min_order[r2] = min_order[r2], min_order[r1]
            updfb = compute_combined_score(min_order, augmented_tsp_matrix, precedence_matrix, tsp_stats, precedence_stats, alpha)
            if updfb < min_fb:
                min_fb = updfb
                updated = True
            else:
                min_order[r1], min_order[r2] = min_order[r2], min_order[r1]
        if not updated:
            break
    # print(min_fb)
    return min_order