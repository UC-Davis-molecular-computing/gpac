"""
GPAC is a Python package for numerically simulating a general-purpose analog computer (GPAC),
defined by Claude Shannon in 1941 as an abstract model of programmable analog computational devices
such as the differential analyzer created by Vannevar Bush and Harold Locke Hazen in the 1920s.
See here for a description of GPACs:

    - https://en.wikipedia.org/wiki/General_purpose_analog_computer
    - https://arxiv.org/abs/1805.05729

GPACs are typically defined by a circuit with gates that can add, multiply, introduce constants, and
integrate an input with respect to time. The most elegant way to specify a GPAC is by defining a set of
ordinary differential equations (ODEs) corresponding to the output wires of integrator gates in the GPAC
circuit.

So really this package makes it easy to write down such ODEs and numerically integrate them and plot them.
"""

from typing import Dict, Iterable, Tuple, Union, Optional, Callable, Any

import scipy.integrate
from scipy.integrate._ivp.ivp import OdeResult  # noqa
import sympy
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


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
        # triggering the warning from solve_ivp.
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

    .. code-block:: python

        import sympy, gpac, numpy as np
        a,b,c = sympy.symbols('a b c')
        odes = {
            a: -a*b + c*a,
            b: -b*c + a*b,
            c: -c*a + b*c,
        }
        initial_values = {
            a: 10,
            b: 1,
            c: 1,
        }
        t_eval = np.linspace(0, 3, 200)
        gpac.integrate_odes(odes, initial_values, t_eval=t_eval)

    This outputs

    .. code-block::

          message: 'The solver successfully reached the end of the integration interval.'
         nfev: 62
         njev: 0
          nlu: 0
          sol: None
       status: 0
      success: True
            t: array([0.  , 0.25, 0.5 , 0.75, 1.  ])
     t_events: None
            y: array([[10.        ,  4.84701622,  0.58753815,  0.38765743,  3.07392998],
           [ 1.        ,  6.84903338,  9.63512628,  3.03634559,  0.38421121],
           [ 1.        ,  0.3039504 ,  1.77733557,  8.57599698,  8.54185881]])
     y_events: None


    Although you cannot reference the time variable directly in the ODEs, this can be simulated
    by introducing a new variable `t` whose derivative is 1 and initial value is the initial time.
    For example, the following code implements ``a(t) = sin(t)`` (with time derivative ``a'(t) = cos(t)``)
    and ``b(t) = cos(t)`` (with time derivative ``b'(t) = -sin(t)``):

    .. code-block:: python

        import sympy, gpac, numpy as np
        from sympy import sin, cos
        from math import pi

        a,b,t = sympy.symbols('a b t')
        odes = {
            a: cos(t),
            b: -sin(t),
            t: 1,
        }
        initial_values = {
            a: 0,
            b: 1,
            t: 0,
        }
        t_eval = np.linspace(0, 3*2*pi, 200)
        gpac.integrate_odes(odes, initial_values, t_eval=t_eval)

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
            Any symbols in the ODEs that are not keys in `initial_values`
            will be assumed to have initial value of 0.

        t_eval:
            iterable of times at which to evaluate the ODEs.
            At least one of `t_eval` or `t_span` must be specified.

        t_span:
            pair (start_time, end_time) for the integration.
            If not specified, first and last times in `t_eval` are used.
            (This is different from solve_ivp, which requires `t_span` to be specified.)
            At least one of `t_eval` or `t_span` must be specified.

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
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
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

    # ensure that all symbols that are keys in `initial_values` are also keys in `odes`
    initial_values_keys = set(initial_values.keys())
    odes_keys = set(odes_symbols.keys())
    diff = initial_values_keys - odes_keys
    if len(diff) > 0:
        raise ValueError(f"\nInitial_values contains symbols that are not in odes: "
                         f"{comma_separated(diff)}"
                         f"\nHere are the symbols of the ODES:                     "
                         f"{comma_separated(odes_keys)}")

    # ensure all symbols in expressions are keys in the odes dict
    symbols_in_expressions_not_in_odes_keys = symbols_found_in_expressions - odes_keys
    if len(symbols_in_expressions_not_in_odes_keys) > 0:
        raise ValueError(f"Found symbols in expressions that are not keys in the odes dict: "
                         f"{symbols_in_expressions_not_in_odes_keys}\n"
                         f"The keys in the odes dict are: {odes_keys}")

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
        figure_size: Tuple[float, float] = (8, 8),
        symbols_to_plot: Optional[Iterable[Union[sympy.Symbol, str]]] = None,
        method: Union[str, scipy.integrate.OdeSolver] = 'RK45',
        dense_output: bool = False,
        events: Optional[Union[Callable, Iterable[Callable]]] = None,
        vectorized: bool = False,
        args: Optional[Tuple] = None,
        **options,
) -> None:
    """
    Numerically integrate the given ODEs using the function :func:`integrate_odes`,
    then plot the trajectories using matplotlib.
    (Assumes it is being run in a Jupyter notebook.)

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
            Any symbols in the ODEs that are not keys in `initial_values`
            will be assumed to have initial value of 0.

        t_eval:
            iterable of times at which to evaluate the ODEs.
            At least one of `t_eval` or `t_span` must be specified.

        t_span:
            pair (start_time, end_time) for the integration.
            If not specified, first and last times in `t_eval` are used.
            (This is different from solve_ivp, which requires `t_span` to be specified.)
            At least one of `t_eval` or `t_span` must be specified.

        figure_size:
            pair (width, height) of the figure

        symbols_to_plot:
            symbols to plot; if empty, then all symbols are plotted

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
            For solver-specific parameters to `solve_ivp`,
            see documentation for `solve_ivp` in scipy.integrate:
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html

            Also used for keyword options to `plot` in matplotlib.pyplot:
            https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html.
            However, note that using such arguments here will cause solve_ivp to print a warning
            that it does not recognize the keyword argument.
    """

    # normalize symbols_to_plot to be a frozenset of strings (names of symbols)
    if symbols_to_plot is None:
        symbols_to_plot = odes.keys()
    symbols_to_plot = frozenset(str(symbol) for symbol in symbols_to_plot)

    # check that symbols all appear as keys in odes
    symbols_of_odes = frozenset(str(symbol) for symbol in odes.keys())
    diff = symbols_to_plot - symbols_of_odes
    if len(diff) > 0:
        raise ValueError(f"\nsymbols_to_plot contains symbols that are not in odes: "
                         f"{comma_separated(diff)}"
                         f"\nSymbols in ODEs:                                       "
                         f"{comma_separated(symbols_of_odes)}")

    sol = integrate_odes(
        odes=odes,
        initial_values=initial_values,
        t_span=t_span,
        t_eval=t_eval,
        method=method,
        dense_output=dense_output,
        events=events,
        vectorized=vectorized,
        args=args,
        **options,
    )

    figure(figsize=figure_size)

    for idx, symbol in enumerate(odes.keys()):
        symbol_name = str(symbol)
        if symbol_name in symbols_to_plot:
            y = sol.y[idx]
            plt.plot(sol.t, y, label=str(symbol), **options)

    plt.xlabel('time')
    plt.legend()
    plt.show()


def comma_separated(elts: Iterable[Any]) -> str:
    return ', '.join(str(elt) for elt in elts)
