import subprocess
import shutil
import numpy as np

def convert_asymmetric_to_symmetric(matrix):
    # Convert asymmetric TSP to symmetric TSP problem
    # Nodes 0 to n-1 are original nodes
    # Nodes n to 2n-1 are dummy cities that are respectives copies (0->n, 1->n+1, ...)
    n = len(matrix)
    new_matrix = np.zeros(shape=(n*2, n*2))

    # City A to its dummy city A' is -infinity for all A, A'
    for i in range(n):
        new_matrix[i][i+n] = new_matrix[i+n][i] = -9999999
    
    # City A to city B is infinity for all A, B
    for i in range(n):
        for j in range(n):
            new_matrix[i][j] = 999

    # Dummy city A' to dummy city B' is infinity for all A', B'
    for i in range(n,2*n):
        for j in range(n, 2*n):
            new_matrix[i][j] = 999

    # City A to dummy city B' is weight from A to B' for all A,B'
    for i in range(n):
        for j in range(n):
            if i != j:
                new_matrix[i][j+n] = new_matrix[j+n][i] = int(matrix[i][j])
    
    return new_matrix

def write_concorde_tspfile(matrix, fname):
    with open(fname, "w") as f:
        f.write("NAME: {}\n".format(fname))
        f.write("TYPE: TSP\n")
        f.write("COMMENT: AUGMENTED WEIGHTS\n")
        f.write("DIMENSION: {}\n".format(len(matrix)))
        f.write("EDGE_WEIGHT_TYPE: EXPLICIT\n")
        f.write("EDGE_WEIGHT_FORMAT: FULL_MATRIX\n")
        f.write("EDGE_WEIGHT_SECTION\n")
        for row in matrix:
            f.write(" ".join(map(str,map(int,row))) + "\n")
        f.write("EOF")

def compute_tsp_score(matrix, final_tour):
    nvert = len(matrix)
    path = 0
    for i in range(1,nvert):
        curr = final_tour[i]
        prev = final_tour[i-1]
        path += matrix[prev][curr]
    return path

def solve_tsp(orig_matrix, fpath):
    scale = 1000
    test_matrix = -orig_matrix*scale
    nvert = len(test_matrix)
    new_matrix = convert_asymmetric_to_symmetric(test_matrix)
    write_concorde_tspfile(new_matrix, fpath + "boundary.tsp")
    # solver = TSPSolver.from_tspfile(fpath)
    # solution = solver.solve()
    subprocess.run(["timeout", "1", "~/concorde/TSP/concorde", fpath + "boundary.tsp"])
    with open("boundary.sol", "r") as f:
        num_nodes = f.readline()
        solution = []
        for line in f:
            solution.extend([int(x) for x in line.split()])
    
    shutil.move("boundary.sol", fpath + "boundary.sol")
    # print(solution.tour)
    # print(solution.optimal_value)
    tour = [x for x in solution if x < nvert]
    # # tour = [2,1,0,4,3]
    s1 = compute_tsp_score(orig_matrix, tour)
    s2 = compute_tsp_score(orig_matrix, list(reversed(tour)))
    if s2 > s1:
        tour = list(reversed(tour))

    min_cut_idx = 0
    min_cut = orig_matrix[tour[-1]][tour[0]]
    # # print(tour)
    # # tour = [0,3,2,1]
    # # print(max_cut_idx, max_cut)
    for i in range(1,nvert):
        curr = tour[i]
        prev = tour[i-1]
        if orig_matrix[prev][curr] < min_cut:
            min_cut_idx = i
            min_cut = orig_matrix[prev][curr]
            # print(max_cut_idx, max_cut)
    final_tour = tour[min_cut_idx:] + tour[:min_cut_idx]
    return final_tour

# test_matrix = np.array([
#     [0.00,1.00,0.12,0.20,0.15],
#     [0.15,0.00,1.00,0.11,0.02],
#     [0.08,0.23,0.00,0.02,0.03],
#     [0.01,0.01,0.13,0.00,1.00],
#     [1.00,0.11,0.10,0.02,0.10],
# ])
# test_matrix = np.array([
#     [0.00,0.01,0.01,0.10,1.00],
#     [1.00,0.00,0.05,0.11,0.02],
#     [0.08,1.00,0.00,0.02,0.03],
#     [0.01,0.01,0.03,0.00,0.01],
#     [0.01,0.03,0.02,1.00,0.00],
# ])
# test_matrix = np.array([
#     [0.00,0.01,0.12,0.32],
#     [1.00,0.00,0.05,0.12],
#     [0.08,1.00,0.00,0.12],
#     [0.01,0.01,1.00,0.00]
# ])
# random.seed(42)
# order = list(range(30))
# random.shuffle(order)
# test_matrix = np.zeros(shape=(30,30))
# for i in range(1,len(order)):
#     prev = order[i-1]
#     curr = order[i]
#     test_matrix[prev][curr] = 1

# tour = solve_tsp(test_matrix, "test.tsp")
# print(order)
# print(tour)