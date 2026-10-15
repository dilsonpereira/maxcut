import os
import numpy as np

# Benchmark instances shipped with the code (../instances relative to this file).
INSTANCES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'instances')

def get_file_graph(inst_name):
    # inst_name is either an existing file path, or a path relative to
    # INSTANCES_DIR (e.g. 'mac_all/rudy/g05_100.0').
    filename = os.path.expanduser(inst_name)
    if not os.path.isfile(filename):
        filename = os.path.join(INSTANCES_DIR, inst_name)
    f = open(filename)
    n, m = map(int, f.readline().split())
    Q = np.zeros((n,n))
    for l in range(m):
        i, j, w = f.readline().split()
        i, j = int(i)-1, int(j)-1
        Q[i,j] = Q[j,i] = int(w)

    return Q

def get_max_cut(inst_name):
    Q = get_file_graph(inst_name)
    np.fill_diagonal(Q, Q.diagonal() + np.sum(-Q, axis=0))
    return Q


def get_bqp_to_maxcut(inst_name):
    Q = get_file_graph(inst_name)

    n = Q.shape[0]
    Q_ = np.zeros((n+1, n+1))
    Q_[1:,1:] = Q
    Q_[0,1:] = -np.sum(Q, axis=0)-np.sum(Q, axis=1)
    np.fill_diagonal(Q_, 0)

    np.fill_diagonal(Q_, Q_.diagonal() + np.sum(-Q_, axis=0))

    return Q_
