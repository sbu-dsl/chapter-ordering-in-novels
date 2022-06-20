from scipy import stats
import numpy as np

def PMR(order, gold_order):
    if np.array_equal(order, gold_order):
        return 1
    else:
        return 0

def Acc(order, gold_order):
    return sum([x==y for x,y in zip(order, gold_order)]) / len(order)

def Tau(order, gold_order):
    tau, _ = stats.kendalltau(order, gold_order)
    return tau

def Spearman(order, gold_order):
    m = {}
    for i, idx in enumerate(order):
        m[idx] = i
    mapped_order = []
    for idx in gold_order:
        mapped_order.append(m[idx])
    r, _ = stats.spearmanr(list(range(len(order))), mapped_order)
    return r

def RougeS(order, gold_order):
    order_pairs = set()
    gold_order_pairs = set()
    for i in range(len(gold_order)):
        for j in range(i+1, len(gold_order)):
            order_pairs.add((order[i], order[j]))
            gold_order_pairs.add((gold_order[i], gold_order[j]))
    return len(order_pairs.intersection(gold_order_pairs)) / len(order_pairs)

def LCS(order, gold_order):
    def lcs(X, Y):
        m = len(X)
        n = len(Y)
        L = [[None]*(n + 1) for i in range(m + 1)]
    
        for i in range(m + 1):
            for j in range(n + 1):
                if i == 0 or j == 0 :
                    L[i][j] = 0
                elif X[i-1] == Y[j-1]:
                    L[i][j] = L[i-1][j-1]+1
                else:
                    L[i][j] = max(L[i-1][j], L[i][j-1])
    
        return L[m][n]
    
    lcs_score = lcs(order, gold_order) / len(order)
    return lcs_score
