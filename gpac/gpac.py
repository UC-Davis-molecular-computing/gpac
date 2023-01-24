"""
GPAC is a Python package for numerically simulating a general-purpose analog computer (GPAC),
defined by Claude Shannon in 1941 as an abstract model of programmable analog computational devices
such as the differential analyzer created by Vannevar Bush and Harold Locke Hazen in the 1920s.

TODO: describe how a GPAC works
"""
from dataclasses import dataclass

from typing import Dict, Iterable, Tuple, Union, Optional, Callable, Sequence

import scipy.integrate
from scipy.integrate._ivp.ivp import OdeResult
import sympy
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


# TODO: add optional parameters to integrate_odes and plot to customize the call to solve_ivp
def integrate_odes(
        odes: Dict[Union[sympy.Symbol, str], Union[sympy.Expr, str]],
        initial_values: Dict[Union[sympy.Symbol, str], float],
        t_eval: Optional[Iterable[float]] = None,
        t_span: Optional[Tuple[float, float]] = None,
        method: Union[str, scipy.integrate.OdeSolver] = 'RK45',
        dense_output: bool = False,
        events: Optional[Union[Callable, Iterable[Callable]]] = None,
        vectorized: bool = False,
        args: Optional[Tuple] = None,
        **options,
        ########################################################################################
        # XXX: the following are all the options that can be passed to solve_ivp,
        # but some are only for certain solvers, and we get a warning if we pass
        # some of them in (rather than using them as keyword arguments in **options).
        # So despite the fact that I prefer strongly-typed and explicity named parameters
        # instead of just keyword arguments in **options, leaving these out avoids
        # triggering the warning from solve-ivp.
        ########################################################################################
        # first_step: Optional[float] = None,
        # max_step: float = np.inf,
        # rtol: float = 1e-3,
        # atol: float = 1e-6,
        # jac: Optional[Union[Callable, np.ndarray, Sequence]] = None,
        # jac_sparsity: Optional[np.ndarray] = None,
        # lband: Optional[int] = None,
        # uband: Optional[int] = None,
        # min_step: float = 0.0,
) -> OdeResult:
    """
    Integrate the given ODEs using scipy, returning the same object returned by `solve_ivp` in the
    package scipy.integrate:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html

    This is a convienence function that wraps the scipy function `solve_ivp`,
    allowing the user to specify the ODEs using sympy symbols and expressions
    (instead of a Python function on tuples of floats, which is what `solve_ivp` expects).

    The object `solution` returned by `solve_ivp` has field `solution.y` which is a 2D numpy array,
    each row of which is the trajectory of a value in the ODEs. The order of the rows is the same as the
    iteration order of the keys in the `odes` dict.

    Besides the parameters described below,
    all other parameters are simply passed along to `solve_ivp` in scipy.integrate.
    As with that function, the following are explicitly named parameters:
    `method`, `dense_output`, `events`, `vectorized`, `args`, and
    all other keyword arguments are passed in through `**options`; see the
    documentation for solve_ivp for a description of these parameters:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html


    Args:
        odes:
            dict mapping sympy symbols to sympy expressions representing the ODEs.
            Alternatively, the keys can be strings, and the values can be strings that look like expressions,
            e.g., ``{'a': '-a*b + c*a'}``.
            If a symbol is referenced in an expression but is not a key in `odes`,
            a ValueError is raised.

        initial_values:
            dict mapping sympy symbols to initial values of each symbol.
            Alternatively, the keys can be strings.
            Any symbols in the ODEs that are not in the initial values
            will be assumed to have initial value of 0.

        t_eval:
            iterable of times at which to evaluate the ODEs

        t_span:
            pair of (start_time, end_time) for the integration
            (if not specified, first and last times in `t_eval` are used)

        method:
            See documentation for `solve_ivp` in scipy.integrate:
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html

        dense_output:
            See documentation for `solve_ivp` in scipy.integrate:
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html

        events:
            See documentation for `solve_ivp` in scipy.integrate:
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html

        vectorized:
            See documentation for `solve_ivp` in scipy.integrate:
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html

        args:
            See documentation for `solve_ivp` in scipy.integrate:
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html

        options:
            For solver-specific parameters, see documentation for `solve_ivp` in scipy.integrate:
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html

    Returns:
        solution to the ODEs, same as object returned by `solve_ivp` in scipy.integrate
    """
    if t_eval is not None:
        t_eval = np.array(t_eval)

    if t_eval is None and t_span is None:
        raise ValueError("Must specify either t_eval or t_span")
    elif t_eval is not None and t_span is None:
        t_span = (t_eval[0], t_eval[-1])

    # normalize initial values dict to use symbols as keys
    initial_values = {sympy.Symbol(symbol) if isinstance(symbol, str) else symbol: value
                      for symbol, value in initial_values.items()}

    # normalize odes dict to use symbols as keys
    odes_symbols = {}
    symbols_found_in_expressions = set()
    for symbol, expr in odes.items():
        if isinstance(symbol, str):
            symbol = sympy.symbols(symbol)
        if isinstance(expr, (str, int, float)):
            expr = sympy.sympify(expr)
        symbols_found_in_expressions.update(expr.free_symbols)
        odes_symbols[symbol] = expr

    # ensure all symbols in expressions are keys in the odes dict
    symbols_in_expressions_not_in_keys = symbols_found_in_expressions - set(odes_symbols.keys())
    if len(symbols_in_expressions_not_in_keys) > 0:
        raise ValueError(f"Found symbols in expressions that are not keys in the odes dict: "
                         f"{symbols_in_expressions_not_in_keys}\n"
                         f"The keys in the odes dict are: {odes_symbols.keys()}")

    odes = odes_symbols

    all_symbols = tuple(odes.keys())
    ode_funcs = {symbol: sympy.lambdify(all_symbols, ode) for symbol, ode in odes.items()}

    def ode_func_vector(_, vals):
        return tuple(ode_func(*vals) for ode_func in ode_funcs.values())

    # sort keys of initial_values according to order of keys in odes,
    # and assume initial value of 0 for any symbol not specified
    initial_values_sorted = [initial_values[symbol] if symbol in initial_values else 0
                             for symbol in all_symbols]
    solution = solve_ivp(
        fun=ode_func_vector,
        t_span=t_span,
        y0=initial_values_sorted,
        t_eval=t_eval,
        method=method,
        dense_output=dense_output,
        events=events,
        vectorized=vectorized,
        args=args,
        **options,
        # first_step=first_step,
        # max_step=max_step,
        # rtol=rtol,
        # atol=atol,
        # jac=jac,
        # jac_sparsity=jac_sparsity,
        # lband=lband,
        # uband=uband,
        # min_step=min_step,
    )

    # mypy complains about solution not being an OdeResult, but it is
    return solution  # type:ignore


def plot(
        odes: Dict[sympy.Symbol, sympy.Expr],
        initial_values: Dict[sympy.Symbol, float],
        t_eval: Optional[Iterable[float]] = None,
        t_span: Optional[Tuple[float, float]] = None,
        figure_size: Tuple[float, float] = (10, 10),
        symbols_to_plot: Optional[Iterable[Union[sympy.Symbol, str]]] = None,
) -> None:
    """
    Plot the solution to the given ODEs using matplotlib.
    (Assumes it is being run in a Jupyter notebook.)

    Args:
        odes:
            dict mapping sympy symbols to sympy expressions representing the ODEs

        initial_values:
            dict mapping synmpy symbols to initial values of each symbol

        t_eval:
            iterable of times at which to evaluate the ODEs

        t_span:
            pair of (start_time, end_time) for the integration
            (if not specified, first and last times in `t_eval` are used)

        figure_size:
            pair (width, height) of the figure

        symbols_to_plot:
            symbols to plot; if empty, then all symbols are plotted
    """

    # normalize symbols_to_plot to be a frozenset of strings (names of symbols)
    if symbols_to_plot is None:
        symbols_to_plot = odes.keys()
    symbols_to_plot = frozenset(str(symbol) for symbol in symbols_to_plot)

    sol = integrate_odes(
        odes=odes,
        initial_values=initial_values,
        t_span=t_span,
        t_eval=t_eval,
    )

    figure(figsize=figure_size)

    for idx, symbol in enumerate(odes.keys()):
        symbol_name = str(symbol)
        if symbol_name in symbols_to_plot:
            y = sol.y[idx]
            plt.plot(sol.t, y, label=str(symbol))

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
