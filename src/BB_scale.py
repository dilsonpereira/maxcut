import math
import time

import psutil
import os

import config

def solve(root, ub=None, ubx=None):
    search_type = config.SEARCH_TYPE
    output      = config.OUTPUT
    mem_limit   = config.MEM_LIMIT

    ts = time.time()
    ts_cpu = time.process_time()

    total_memory = psutil.virtual_memory().total
    process = psutil.Process(os.getpid())

    lb = None

    def get_ub():
        return ub, ubx

    root.external_ub = get_ub
    root.GilmoreLawler(Q=root.Q)
    ub, ubx = root.get_ub()

    root.solve_node()
    root_lb = root.lb
    root_fixed = getattr(root, 'n_fixed', 0)
    nodes = [root]

    t_root = time.time()-ts

    if output:
        tri_r = len(root.tri) + len(root.tri2)
        print(f'root: lb={root_lb:.4f}  ub={ub:.4f}  tri={tri_r}  '
              f'fixed={root_fixed}  ({t_root:.1f}s)', flush=True)

    solved_nodes = 1

    def get_node():
        if len(nodes) > 0:
            process_memory = process.memory_info().rss
            usage_percent = (process_memory / total_memory) * 100

            if search_type == 'DFS' or (search_type=='hybrid' and usage_percent > mem_limit):
                return nodes.pop()

            i = min((i for i in range(len(nodes))), key=lambda i: nodes[i].lb)
            return nodes.pop(i)
        return None


    def add_node(n):
        nodes.append(n)

    def print_tree(n):
        id = n.id;
        st = [' ']*n.level*2

        for i in range(n.level):
            id //= 2
            if id % 2 == 0:
                st[2*(n.level-1)-i*2:2*(n.level-1)-i*2+2] = ['|', ' ']

        print(''.join(st), end='')
        print(f'+--n{str(n.id)[-5:]} ', end='')
        print(f'ub: {n.get_ub()[0] if n.get_ub()[0] != None else "-"} lb: {n.lb:.2f} ', end='')
        print()

    try:
        while True:
            if time.time()-ts > config.TIME_LIMIT:
                break

            lb = min(n.lb for n in nodes) if len(nodes) > 0 else ub

            if math.ceil(lb-1e-5) >= round(ub):
                break

            n = get_node()

            if n == None:
                break

            if output:
                if search_type == 'DFS':
                    print_tree(n)
                else:
                    tri_n = len(n.tri) + len(n.tri2)
                    print(f'id: {n.id}, ub:{n.get_ub()[0] if n.get_ub() is not None else "-"}, lb:{lb}, tri:{tri_n}')


            if n.get_ub()[0] != None and (ub == None or n.get_ub()[0] < ub):
                ub, ubx = n.get_ub()

            # Branch unless the node is pruned by bound.
            if ub is None or math.ceil(n.lb-1e-5) < round(ub):
                children = n.branch()
                for c in children:
                    c.lb = n.lb
                    c.solve_node()
                    solved_nodes += 1
                    add_node(c)

    except KeyboardInterrupt:
        lb = min(n.lb, min(n.lb for n in nodes) if len(nodes) > 0 else ub)
               
    te = time.time()
    te_cpu = time.process_time()

    info = dict(ub=ub, lb=lb, root_lb=root_lb, t=te-ts, t_cpu=te_cpu-ts_cpu, nodes=solved_nodes, ubx=ubx, t_root=t_root, root_fixed=root_fixed)

    return info


