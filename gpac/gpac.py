"""
GPAC is a Python package for numerically simulating a general-purpose analog computer (GPAC),
defined by Claude Shannon in 1941 as an abstract model of programmable analog computational devices
such as the differential analyzer created by Vannevar Bush and Harold Locke Hazen in the 1920s.

TODO: describe how a GPAC works
"""
from dataclasses import dataclass

from typing import Dict, Iterable, Tuple

from scipy.integrate._ivp.ivp import OdeResult
from sympy import symbols, Eq, Function, lambdify, Symbol, Expr
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

def integrate_odes(
        odes: Dict[Symbol, Expr],
        initial_values: Dict[Symbol, float],
        times: Iterable[float] = np.arange(0, 1, 0.01),
) -> OdeResult:
    symbols = tuple(odes.keys())
    ode_funcs = {symbol: lambdify(symbols, ode) for symbol, ode in odes.items()}
    def ode_func_vector(t, vals):
        return [ode_func(*vals) for ode_func in ode_funcs.values()]
    # sort keys of initial_values according to order of keys in odes,
    # and assume initial value of 0 for any symbol not specified
    initial_values_sorted = [initial_values[symbol] if symbol in initial_values else 0 for symbol in symbols]
    solution = solve_ivp(ode_func_vector, [times[0], times[-1]], y0=initial_values_sorted, t_eval=times)
    return solution

def plot(
        odes: Dict[Symbol, Expr],
        initial_values: Dict[Symbol, float],
        times: Iterable[float] = np.arange(0, 1, 0.01),
        figure_size: Tuple[float, float] = (10, 10),
) -> None:
    """
    Plot the solution to the given ODEs using matplotlib.
    (Assumes it is being run in a Jupyter notebook.)

    :param odes:
        dict mapping sympy symbols to sympy expressions representing the ODEs
    :param initial_values:
    :param times:
    :param figure_size:
    :return:
    """
    sol = integrate_odes(odes, initial_values, times)

    figure(figsize=figure_size)

    for idx, symbol in enumerate(odes.keys()):
        plt.plot(sol.t, sol.y[idx], label=str(symbol))

    plt.legend()
    plt.show()

@dataclass
class GPAC:
    """
    A GPAC class for numerically simulating a general-purpose analog computer (GPAC),
    defined by Claude Shannon in 1941 as an abstract model of programmable analog computational devices
    such as the differential analyzer created by Vannevar Bush and Harold Locke Hazen in the 1920s.
    """

    def __init__(self) -> None:
        """
        TODO:
        """
