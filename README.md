# Lagrangian SDP Branch-and-Bound for Max-Cut

This is an exact Branch-and-Bound solver for the Max-Cut problem, and for
unconstrained binary quadratic problems (BQP/QUBO), which it converts to Max-Cut.
Bounds come from a Lagrangian relaxation of the SDP constraint. The multipliers are
kept in factored form, and triangle inequalities are added through a
relax-and-cut scheme.

## Directory layout

```
.
|-- README.txt
|-- instances/
|   |-- mac_all/            Max-Cut instances
|   |   |-- rudy/           g05, pm1d, pm1s, pw01/05/09, w01/05/09 (n = 60, 80, 100)
|   |   `-- ising/          toroidal grids (t2g*, t3g*) and ising2.5/3.0 instances
|   `-- biq_all/            BQP/QUBO instances
|       |-- be/             be100 ... be250
|       |-- beasley/        bqp50, bqp100, bqp250, bqp500
|       `-- gka/            gka1a ... gka10d
`-- src/
    |-- run.py              Command-line entry point (single instances and batches)
    |-- node.py             B&B node: Lagrangian relaxation, separation, branching
    |-- BB_scale.py         B&B tree search driver
    |-- config.py           Algorithm parameters and INI loader
    |-- params.ini          Example parameter file
    |-- instance.py         Instance readers and BQP-to-Max-Cut conversion
    `-- stepsize.py         Subgradient step-size rules
```

## Requirements

- Python 3.8 or later (tested with Python 3.10.12)
- [NumPy](https://numpy.org/) (tested with 2.2.6)
- [psutil](https://pypi.org/project/psutil/) (tested with 5.9.6)
- [threadpoolctl](https://pypi.org/project/threadpoolctl/) (tested with 3.2.0)

Install them with:

```bash
pip install numpy psutil threadpoolctl
```

The solver uses no other external packages or commercial solvers. It limits BLAS
to a single thread, so all reported times are sequential.

## Running the code

Run all commands from the `src/` directory:

```bash
cd src
```

### Single instances

Max-Cut instance (edge-list format):

```bash
python run.py --maxcut mac_all/rudy/g05_100.0
```

BQP/QUBO instance, converted internally to Max-Cut with one extra vertex:

```bash
python run.py --qubo biq_all/be/be150.3.1.sparse
```

To display solver progress during the execution:

```bash
python run.py --output --qubo biq_all/be/be150.3.1.sparse
```

Example output of `python run.py --maxcut mac_all/rudy/g05_60.0`:

```
best value    : 536.0
upper bound   : 536.9972
root ub       : 537.2733
nodes         : 29
fixed at root : 0
time (wall)   : 2.28s
time (cpu)    : 2.67s
time root     : 1.66s
solution      : 1 -1 -1 1 -1 -1 1 -1 -1 -1 1 -1 1 1 -1 -1 1 -1 ...
```

The final results are reported in **maximization form**.
Internally the solver minimizes the negated objective. The B&B progress printed
with the output flag turned on is therefore shown with negative values.


| Field                       | Meaning |
|-----------------------------|---------|
| `best value`                | Value of the best solution found (the optimum if the search finished) |
| `upper bound`               | Final global upper bound. With integer weights, the instance is solved when `floor(upper bound) == best value` |
| `root ub`                   | Upper bound at the root node |
| `nodes`                     | Number of B&B nodes solved |
| `fixed at root`             | Number of variables fixed at the root by reduced-cost tests |
| `time (wall)`, `time (cpu)` | Total wall-clock and CPU time |
| `time root`                 | Time spent at the root node |
| `solution`                  | Best solution found |

### Batches of instances

Each batch command solves every instance in a family, one after another. For each
instance it prints three lines: the file name, a dictionary of results, and the
best solution. For example:

```
biq_all/beasley/bqp50-1.sparse
{'value': np.float64(2098.0), 'ub': np.float64(2098.0), 'root_ub': np.float64(2098.99265391435), 't': 2.34, 't_cpu': 2.87, 'nodes': 1, 't_root': 2.34, 'root_fixed': 1275}
solution: 0 1 1 0 0 1 1 0 0 0 0 1 0 0 1 0 0 0 0 1 1 1 1 0 0 0 1 1 1 1 0 0 1 ...
```

| Key            | Field in the table above      |
|----------------|-------------------------------|
| `value`        | `best value`                  |
| `ub`           | `upper bound`                 |
| `root_ub`      | `root ub`                     |
| `nodes`        | `nodes`                       |
| `root_fixed`   | `fixed at root`               |
| `t`, `t_cpu`   | `time (wall)`, `time (cpu)`   |
| `t_root`       | `time root`                   |

Batch commands:

| Command                         | Instances                          | Optional arguments (default: all) |
|---------------------------------|------------------------------------|-----------------------------------|
| `python run.py rudy [fam ...]`  | Rudy, n = 100 (10 per family)      | `g05 pm1d pm1s pw01 pw05 pw09 w01 w05 w09` |
| `python run.py ising [fam ...]` | Ising (3 per family)               | `t2g10 t2g15 t2g20 t3g5 t3g6 t3g7`<br>`ising2.5-100 ... ising2.5-300 ising3.0-100 ... ising3.0-300` |
| `python run.py be [sub ...]`    | be (10 per subfamily)              | `100 120.3 120.8 150.3 150.8 200.3 200.8 250` |
| `python run.py beasley [n ...]` | Beasley bqp (10 per size)          | `50 100 250 500` |
| `python run.py gka`             | GKA (40 instances from sets a-e)   | none |

Examples:

```bash
python run.py rudy                 # all Rudy instances
python run.py rudy g05 pw05        # 20 Rudy instances
python run.py be 100 120.3         # 20 be instances
python run.py beasley 50           # bqp50-1 ... bqp50-10
python run.py ising t2g10 t3g5     # 6 Ising instances
```

## Parameters

**All results reported in the paper use the default parameters**, so the
commands above reproduce them with no extra options.

To experiment with other settings, pass an INI file with `--config` before the
subcommand:

```bash
python run.py --config params.ini --maxcut mac_all/rudy/g05_100.0
python run.py --config params.ini rudy g05
```

`src/params.ini` is an example file that lists the main parameters with their
default values and a short description of each.

## Input formats

**Max-Cut** (`instances/mac_all`): the first line is `n m`, followed by one line
`i j w` per edge (1-indexed vertices, integer weights).

**BQP/QUBO** (`instances/biq_all`, `*.sparse`): the same layout, where `n` is the
number of binary variables and each line `i j q` gives an entry of the objective
matrix. The problem is converted to Max-Cut on `n + 1` vertices.
