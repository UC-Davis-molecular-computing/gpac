"""
GPAC is a Python package for numerically simulating a general-purpose analog computer (GPAC),
defined by Claude Shannon in 1941 as an abstract model of programmable analog computational devices
such as the differential analyzer created by Vannevar Bush and Harold Locke Hazen in the 1920s.

TODO: describe how a GPAC works
"""
from dataclasses import dataclass

from typing import Dict, Iterable, Tuple, Union

from scipy.integrate._ivp.ivp import OdeResult
import sympy
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


def integrate_odes(
        odes: Dict[Union[sympy.Symbol, str], Union[sympy.Expr, str]],
        initial_values: Dict[Union[sympy.Symbol, str], float],
        times: Iterable[float] = tuple(np.arange(0, 1, 0.01)),
) -> OdeResult:
    """
    Integrate the given ODEs using scipy, returning the same object returned by `solve_ivp` in the
    package scipy.integrate:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html

    This is a convienence function that wraps the scipy function `solve_ivp`,
    allowing the user to specify the ODEs using sympy symbols and expressions
    (instead of a Python function on tuples of floats, which is what `solve_ivp` expects).

    The object `solution` returned by `solve_ivp` has field `solution.y` which is a 2D numpy array,
    each row of is the trajectory of a value in the ODEs. The order of the rows is the same as the
    order of the keys in the `odes` dict.

    :param odes:
        dict mapping sympy symbols to sympy expressions representing the ODEs.
        Alternatively, the keys can be strings, and the values can be strings that look like expressions,
        e.g., ``{'a': '-a*b + c*a'}``.
    :param initial_values:
        dict mapping sympy symbols to initial values of each symbol.
        Alternatively, the keys can be strings.
    :param times:
        iterable of times at which to evaluate the ODEs
    :return:
        solution to the ODEs (same as object returned by `solve_ivp` in scipy.integrate)
    """
    times = tuple(times)
    odes_symbols = {}
    for symbol, expr in odes.items():
        if isinstance(symbol, str):
            symbol = sympy.symbols(symbol)
        if isinstance(expr, str):
            expr = sympy.parse_expr(expr)
        odes_symbols[symbol] = expr

    odes = odes_symbols

    all_symbols = tuple(odes.keys())
    ode_funcs = {symbol: sympy.lambdify(all_symbols, ode) for symbol, ode in odes.items()}

    def ode_func_vector(_, vals):
        return tuple(ode_func(*vals) for ode_func in ode_funcs.values())

    # sort keys of initial_values according to order of keys in odes,
    # and assume initial value of 0 for any symbol not specified
    initial_values_sorted = [initial_values[symbol] if symbol in initial_values else 0
                             for symbol in all_symbols]
    solution = solve_ivp(ode_func_vector, [times[0], times[-1]], y0=initial_values_sorted, t_eval=times)

    # mypy complains about solution not being an OdeResult, but it is
    return solution  # type:ignore


def plot(
        odes: Dict[sympy.Symbol, sympy.Expr],
        initial_values: Dict[sympy.Symbol, float],
        times: Iterable[float] = tuple(np.arange(0, 1, 0.01)),
        figure_size: Tuple[float, float] = (10, 10),
) -> None:
    """
    Plot the solution to the given ODEs using matplotlib.
    (Assumes it is being run in a Jupyter notebook.)

    :param odes:
        dict mapping sympy symbols to sympy expressions representing the ODEs
    :param initial_values:
        dict mapping synmpy symbols to initial values of each symbol
    :param times:
        iterable of times at which to evaluate the ODEs
    :param figure_size:
        pair (width, height) of the figure
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
