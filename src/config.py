"""
config.py
=========
Global algorithm configuration for the Max-Cut B&B solver.

Defaults live in the module-level ``root`` and ``child`` dicts and can be
changed at runtime before creating a root node:

    import config
    config.TIME_LIMIT = 3600
    config.root['max_iter'] = 5000
    config.child['branch'] = 'branch_1'

Or loaded from an INI file:

    config.load('my_run.ini')

Config file format
------------------
    [solver]
    time_limit  = 3600
    search_type = hybrid
    mem_limit   = 50
    output      = true
    fixing      = true

    [root]
    max_iter = 5000
    stepsize = halving
    s0       = 1e-1
    target   = 1e-5
    alpha    = 0.99
    branch   = branch_2
    cuts.add      = true
    cuts.freq     = 199
    cuts.min_sep  = 500
    cuts.max_add  = 1000

    [child]
    max_iter = 200
    s0       = 1e-2
    alpha    = 0.95
    cuts.add = false

Recognised top-level keys for [root]/[child]:
    max_iter, stepsize, s0, target, alpha, frac, projected, branch
    (stepsize-function-specific: tolerance, decay)
Recognised cuts sub-keys (cuts.<key>):
    add, rem, max_tri, max_tri2, freq, rem_freq, max_iter,
    min_sep, max_add
"""

import copy
import inspect
import configparser

from stepsize import stepsize_halving, stepsize_exponential, stepsize_sqrt, stepsize_adaptive

# ---------------------------------------------------------------------------
# Stepsize registry — maps string name → factory function
# ---------------------------------------------------------------------------

STEPSIZE_FUNCTIONS = {
    'halving':     stepsize_halving,
    'exponential': stepsize_exponential,
    'sqrt':        stepsize_sqrt,
    'adaptive':    stepsize_adaptive,
}

# ---------------------------------------------------------------------------
# Solver-level settings
# ---------------------------------------------------------------------------

TIME_LIMIT  = 180000    # B&B wall-clock limit (seconds)
SEARCH_TYPE = 'hybrid' # traversal strategy: 'DFS', 'hybrid'
MEM_LIMIT   = 50       # memory % above which hybrid reverts to DFS
OUTPUT      = False     # print B&B progress
FIXING      = True     # apply the delta-test variable fixing in GilmoreLawler

# ---------------------------------------------------------------------------
# Default node parameters (plain values — no function objects)
# 'stepsize' is a string key resolved by get_conf()
# 'branch'   is a method name resolved at call time via getattr(node, ...)
# ---------------------------------------------------------------------------

root = {
    'max_iter':  10000,
    'stepsize':  'halving',
    's0':        1e-1,
    'target':    1e-5,
    'alpha':     0.99,
    'frac':        True,
    'projected':   False,
    'branch':      'branch_2',
    'sb_iter': 20,
    'sb_k':    5,
    'cuts': {
        'add':       True,
        'rem':       False,
        'max_tri':   100000,
        'max_tri2':  100000,
        'freq':       199,
        'rem_freq':   149,
        'max_iter':   10000,
        'min_sep':    500,
        'max_add':    1000,
    },
}

child = {
    'max_iter':  300,
    'stepsize':  'halving',
    's0':        1e-2,
    'target':    1e-4,
    'alpha':     0.95,
    'frac':        True,
    'projected':   False,
    'branch':      'branch_2',
    'sb_iter': 20,
    'sb_k':    5,
    'cuts': {
        'add':       False,
        'rem':       False,
        'max_tri':   10000,
        'max_tri2':  10000,
        'freq':       101,
        'rem_freq':   149,
        'max_iter':  3000,
        'min_sep':   99,
        'max_add':   100,
    },
}

# ---------------------------------------------------------------------------
# conf builder — called by node.maxcut on construction
# ---------------------------------------------------------------------------

def _build_stepsize(fn, params):
    """Call *fn* with whichever of its parameters appear in *params*."""
    sig = inspect.signature(fn)
    kwargs = {k: params[k] for k in sig.parameters if k in params}
    return fn(**kwargs)


def get_conf(is_root):
    """Return the conf dict for a root or child node.

    Resolves the ``'stepsize'`` string to a callable (``'get_stepsize'``)
    using the parameters present in the dict.  ``'branch'`` is kept
    as a string; ``node.branch()`` resolves it via ``getattr``.
    """
    base = copy.deepcopy(root if is_root else child)

    fn = STEPSIZE_FUNCTIONS[base['stepsize']]
    base['get_stepsize'] = _build_stepsize(fn, base)

    # Remove constructor-only keys that the solver does not need at runtime.
    for k in ('stepsize', 's0', 'target', 'tolerance', 'decay'):
        base.pop(k, None)

    return base


# ---------------------------------------------------------------------------
# Config file loader
# ---------------------------------------------------------------------------

def _parse_value(s):
    """Best-effort type coercion: bool → int → float → str."""
    if s.lower() in ('true', 'yes'):
        return True
    if s.lower() in ('false', 'no'):
        return False
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def load(path):
    """Load configuration from an INI-style text file.

    Updates the module-level ``root``, ``child``, ``TIME_LIMIT``, and
    ``SEARCH_TYPE`` in place.  Unknown keys are silently ignored.
    """
    global TIME_LIMIT, SEARCH_TYPE, MEM_LIMIT, OUTPUT, FIXING

    parser = configparser.ConfigParser(inline_comment_prefixes=('#', ';'))
    parser.read(path)

    if parser.has_section('solver'):
        for key, val in parser.items('solver'):
            v = _parse_value(val)
            if key == 'time_limit':
                TIME_LIMIT = v
            elif key == 'search_type':
                SEARCH_TYPE = v
            elif key == 'mem_limit':
                MEM_LIMIT = v
            elif key == 'output':
                OUTPUT = v
            elif key == 'fixing':
                FIXING = v

    for section, target in (('root', root), ('child', child)):
        if not parser.has_section(section):
            continue
        for key, val in parser.items(section):
            v = _parse_value(val)
            if key.startswith('cuts.'):
                sub = key[len('cuts.'):]
                target['cuts'][sub] = v
            elif key in target:
                target[key] = v
