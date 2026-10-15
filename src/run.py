"""
run.py
======
Batch runner and quick-start entry point for the Max-Cut B&B solver.

Usage
-----
Solve a single Max-Cut instance:
    python run.py --maxcut mac_all/rudy/g05_100.0

Solve a single QUBO/BQP instance:
    python run.py --qubo biq_all/be/be150.3.1.sparse

Optional flags (must come before the subcommand):
    --config <path>   Load solver parameters from an INI file
    --output          Print B&B progress (overrides the config file)

Instance paths are resolved relative to the ../instances directory
(an existing file path may also be given directly).

Batch sub-commands:
    python run.py be [100 120.3 120.8 150.3 150.8 200.3 200.8 250]
    python run.py beasley [50 100 250 500]
    python run.py gka
    python run.py ising [t2g10 t2g15 t2g20 t3g5 t3g6 t3g7 ising2.5-100 ... ising3.0-300]
    python run.py rudy [g05 pm1d pm1s pw01 pw05 pw09 w01 w05 w09]

The best solution found is printed after the results: for Max-Cut, the side
(+1/-1) of each vertex 1..n, with vertex 1 on side +1; for BQP, the 0/1 value
of each variable x_1..x_n.
"""

import sys
import numpy as np
from threadpoolctl import threadpool_limits

import instance
import node
import BB_scale
import config


# ---------------------------------------------------------------------------
# Instance loaders
# ---------------------------------------------------------------------------

def load_maxcut(path):
    """Load a Max-Cut edge-list instance and return (Q, scale)."""
    Q = instance.get_max_cut(path)
    Q = (1 / 4) * Q
    s = np.max(np.abs(Q + Q.T))
    return Q, s


def load_bqp(path):
    """Load a BQP/QUBO instance, lift to Max-Cut, and return (Q, scale)."""
    Q = instance.get_bqp_to_maxcut(path)
    Q = (1 / 4) * Q
    s = np.max(np.abs(Q + Q.T))
    return Q, s


# ---------------------------------------------------------------------------
# Single-instance solver
# ---------------------------------------------------------------------------

def solve_file(path, load_fn):
    """Solve a single instance and return the info dict."""
    Q, s = load_fn(path)
    with threadpool_limits(limits=1, user_api='blas'):
        info = BB_scale.solve(node.maxcut(Q=Q, scale=s))
    return info


def get_solution(info, qubo):
    """Return the best solution found as a list of ints.

    Max-Cut: side (+1/-1) of each vertex, normalized so vertex 1 is on side +1.
    BQP: 0/1 value of each variable (the auxiliary Max-Cut vertex is dropped).
    """
    x = info['ubx']
    if x is None:
        return None
    x = np.rint(x).astype(int)
    x *= x[0]
    if qubo:
        return ((1 - x[1:]) // 2).tolist()
    return x.tolist()


def print_info(info, qubo):
    """Print the batch-run result dict followed by the best solution."""
    # The solver minimizes the negated objective: flip signs for output.
    print({'value': -info['ub'], 'ub': -info['lb'], 'root_ub': -info['root_lb'],
           **{k: v for k, v in info.items() if k not in ('ub', 'lb', 'root_lb', 'ubx')}})
    sol = get_solution(info, qubo)
    print('solution:', ' '.join(map(str, sol)) if sol is not None else '-', flush=True)


# ---------------------------------------------------------------------------
# Batch runners — QUBO/BQP families
# ---------------------------------------------------------------------------

def be(subfamilies=None):
    """Run be instances.

    Subfamilies (by size[.density]): 100, 120.3, 120.8, 150.3, 150.8,
                                     200.3, 200.8, 250
    Default: all.
    """
    all_subfamilies = {
        '100':   [f'biq_all/be/be100.{i}.sparse'     for i in range(1, 11)],
        '120.3': [f'biq_all/be/be120.3.{i}.sparse'   for i in range(1, 11)],
        '120.8': [f'biq_all/be/be120.8.{i}.sparse'   for i in range(1, 11)],
        '150.3': [f'biq_all/be/be150.3.{i}.sparse'   for i in range(1, 11)],
        '150.8': [f'biq_all/be/be150.8.{i}.sparse'   for i in range(1, 11)],
        '200.3': [f'biq_all/be/be200.3.{i}.sparse'   for i in range(1, 11)],
        '200.8': [f'biq_all/be/be200.8.{i}.sparse'   for i in range(1, 11)],
        '250':   [f'biq_all/be/be250.{i}.sparse'     for i in range(1, 11)],
    }
    keys = subfamilies if subfamilies else list(all_subfamilies)
    F = [f for k in keys for f in all_subfamilies[k]]

    for f in F:
        print(f)
        info = solve_file(f, load_fn=load_bqp)
        print_info(info, qubo=True)



def beasley(sizes=None):
    """Run Beasley BQP instances (bqp50/100/250/500).

    sizes: list of integers, e.g. [50, 100].  Default: all four.
    """
    all_sizes = [50, 100, 250, 500]
    sizes = [int(x) for x in sizes] if sizes else all_sizes
    F = [f'biq_all/beasley/bqp{n}-{i}.sparse' for n in sizes for i in range(1, 11)]

    for f in F:
        print(f)
        info = solve_file(f, load_fn=load_bqp)
        print_info(info, qubo=True)



def gka():
    """Run all GKA instances."""
    names = [
        'gka1a', 'gka1b', 'gka1c', 'gka1d', 'gka1e',
        'gka2a', 'gka2b', 'gka2c', 'gka2d', 'gka2e',
        'gka3a', 'gka3b', 'gka3c', 'gka3d', 'gka3e',
        'gka4a', 'gka4b', 'gka4c', 'gka4d', 'gka4e',
        'gka5a', 'gka5b', 'gka5c', 'gka5d', 'gka5e',
        'gka6a', 'gka6b', 'gka6c', 'gka6d',
        'gka7a', 'gka7b', 'gka7c', 'gka7d',
        'gka8a', 'gka8b', 'gka8d',
        'gka9b', 'gka9d',
        'gka10b', 'gka10d',
    ]
    F = [f'biq_all/gka/{name}.sparse' for name in names]

    for f in F:
        print(f)
        info = solve_file(f, load_fn=load_bqp)
        print_info(info, qubo=True)



# ---------------------------------------------------------------------------
# Batch runners — Max-Cut families
# ---------------------------------------------------------------------------

def ising(families=None):
    """Run Ising / grid-graph Max-Cut instances.

    Families: t2g10, t2g15, t2g20, t3g5, t3g6, t3g7,
              ising2.5-100 … ising2.5-300, ising3.0-100 … ising3.0-300
    Default: all families.
    """
    seeds = [5555, 6666, 7777]
    torus_families  = ['t2g10', 't2g15', 't2g20', 't3g5', 't3g6', 't3g7']
    ising25_sizes   = [100, 150, 200, 250, 300]
    ising30_sizes   = [100, 150, 200, 250, 300]

    all_families = {
        **{f: [f'mac_all/ising/{f}_{s}' for s in seeds] for f in torus_families},
        **{f'ising2.5-{n}': [f'mac_all/ising/ising2.5-{n}_{s}' for s in seeds]
           for n in ising25_sizes},
        **{f'ising3.0-{n}': [f'mac_all/ising/ising3.0-{n}_{s}' for s in seeds]
           for n in ising30_sizes},
    }

    if families is None:
        families = list(all_families)
    F = [f for fam in families for f in all_families[fam]]

    for f in F:
        print(f)
        info = solve_file(f, load_fn=load_maxcut)
        print_info(info, qubo=False)



def rudy(families=None):
    """Run Rudy n=100 instances.  *families* defaults to all nine families."""
    if families is None:
        families = ['g05', 'pm1d', 'pm1s', 'pw01', 'pw05', 'pw09', 'w01', 'w05', 'w09']
    F = [f'mac_all/rudy/{fam}_100.{i}' for fam in families for i in range(10)]

    for f in F:
        print(f)
        info = solve_file(f, load_fn=load_maxcut)
        print_info(info, qubo=False)



# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    args = sys.argv[1:]

    if not args:
        print(__doc__)
        sys.exit(0)

    # Optional flags (must come before the subcommand).
    output = False
    while args:
        if args[0] == '--config':
            if len(args) < 2:
                print('Usage: python run.py --config <path> <subcommand> [args...]')
                sys.exit(1)
            config.load(args[1])
            args = args[2:]
        elif args[0] == '--output':
            output = True
            args = args[1:]
        else:
            break

    # Applied after the loop so --output overrides the config file in any order.
    if output:
        config.OUTPUT = True

    if not args:
        print(__doc__)
        sys.exit(0)

    cmd = args[0]

    if cmd == 'be':
        be(args[1:] or None)

    elif cmd == 'beasley':
        beasley(args[1:] or None)

    elif cmd == 'gka':
        gka()

    elif cmd == 'ising':
        ising(args[1:] or None)

    elif cmd == 'rudy':
        rudy(args[1:] or None)

    elif cmd in ('--maxcut', '--qubo'):
        if len(args) < 2:
            print(f'Usage: python run.py {cmd} <instance_path>')
            sys.exit(1)
        path = args[1]
        load_fn = load_maxcut if cmd == '--maxcut' else load_bqp
        info = solve_file(path, load_fn=load_fn)
        # The solver minimizes the negated objective: flip signs for output.
        print(f'\nbest value    : {-info["ub"]:.1f}')
        print(f'upper bound   : {-info["lb"]:.4f}')
        print(f'root ub       : {-info["root_lb"]:.4f}')
        print(f'nodes         : {info["nodes"]}')
        print(f'fixed at root : {info["root_fixed"]}')
        print(f'time (wall)   : {info["t"]:.2f}s')
        print(f'time (cpu)    : {info["t_cpu"]:.2f}s')
        print(f'time root     : {info["t_root"]:.2f}s')
        sol = get_solution(info, qubo=False)
        if cmd == '--qubo':
            # Max-Cut solution of the lifted graph (vertex 1 is the auxiliary vertex).
            print(f'maxcut sol    : {" ".join(map(str, sol)) if sol is not None else "-"}')
            sol = get_solution(info, qubo=True)
        print(f'solution      : {" ".join(map(str, sol)) if sol is not None else "-"}')

    else:
        print(f'Unknown command: {cmd}')
        print(__doc__)
        sys.exit(1)
