import math
import numpy as np


def stepsize_adaptive(s0, tolerance, decay):
    """Decay s0 by *decay* whenever the lower bound stalls for *tolerance* iterations."""
    noimp = 0
    lb_ = -np.inf

    def f(ub, lb, norm, *args, **kwargs):
        nonlocal s0, noimp, lb_
        if lb - lb_ > 1e-5:
            noimp = 0
        else:
            noimp += 1
        lb_ = lb
        if noimp > tolerance:
            noimp = 0
            s0 *= decay
        return s0 / np.sqrt(norm)

    return f


def stepsize_sqrt(s0):
    """Classic s0 / sqrt(k) schedule."""
    def f(k, *args, **kwargs):
        return s0 / np.sqrt(k)
    return f


def stepsize_exponential(s0, target, max_iter):
    """Exponential decay from s0 to target over max_iter iterations."""
    def f(k, norm, *args, **kwargs):
        lamb = math.exp(math.log(target / s0) / max_iter)
        step = s0 * (lamb ** min(k, max_iter)) / np.sqrt(norm)
        return max(step, 1e-5)
    return f


def stepsize_halving(s0, target, max_iter):
    """Powers-of-2 halving: divide the iteration budget into equal phases,
    halving s0 at the start of each phase."""
    def f(k, norm, *args, **kwargs):
        p = math.ceil(math.log2(s0 / target))
        iter_per_phase = math.floor(max_iter / p)
        phase = k // iter_per_phase
        return (s0 / (2 ** phase)) / np.sqrt(norm)
    return f
