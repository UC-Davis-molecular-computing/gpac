"""
gpac is a Python package for numerically simulating a general-purpose analog computer (GPAC),
defined by Claude Shannon in 1941 as an abstract model of programmable analog computational devices
such as the differential analyzer created by Vannevar Bush and Harold Locke Hazen in the 1920s.
See here for a description of GPACs:

    - https://en.wikipedia.org/wiki/General_purpose_analog_computer
    - https://arxiv.org/abs/1805.05729

It also has support for a very common model governed by polynomial ODEs, the of continuous mass-action
chemical reaction networks:

    - https://en.wikipedia.org/wiki/Chemical_reaction_network_theory#Overview

GPACs are typically defined by a circuit with gates that can add, multiply, introduce constants, and
integrate an input with respect to time. The most elegant way to specify a GPAC is by defining a set of
ordinary differential equations (ODEs) corresponding to the output wires of integrator gates in the GPAC
circuit.

So essentially, this package makes it easy to write down such ODEs and numerically integrate and plot them.

Although gpac has two submodules ode and crn, you can import all elements from both directly from gpac,
e.g., ``from gpac import plot, plot_crn``.
"""

from typing import Dict, Iterable, Tuple, Union, Optional, Callable, Any, Literal

from scipy.integrate._ivp.ivp import OdeResult  # noqa
import sympy
from scipy.integrate import solve_ivp, OdeSolver
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


def integrate_odes(
        odes: Dict[Union[sympy.Symbol, str], Union[sympy.Expr, str, float]],
        initial_values: Dict[Union[sympy.Symbol, str], float],
        t_eval: Optional[Iterable[float]] = None,
        t_span: Optional[Tuple[float, float]] = None,
        dependent_symbols: Iterable[Union[sympy.Expr, str]] = (),
        method: Union[str, OdeSolver] = 'RK45',
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
    Integrate the given ODEs using scipy's `solve_ivp` function in the
    package scipy.integrate, returning the same object returned by `solve_ivp`:

        https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html

    This is a convienence function that wraps the scipy function `solve_ivp`,
    allowing the user to specify the ODEs using sympy symbols and expressions
    (instead of a Python function on tuples of floats, which is what `solve_ivp` expects,
    but is more awkward to specify than using sympy expressions).

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
        t_eval = np.linspace(0, 1, 5)
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

    All symbols are interpreted as functions of a single variable called "time", and the derivatives
    are with respective to time.

    Although you cannot reference the time variable directly in the ODEs, this can be simulated
    by introducing a new variable `t` whose derivative is 1 and initial value is the initial time.
    For example, the following code implements ``a(t) = sin(t)`` (with time derivative ``a'(t) = cos(t)``)
    and ``b(t) = -(t/2 - 1)^2 + 2`` (with time derivative ``b'(t) = 1 - t/2``):

    .. code-block:: python

        # trick for referencing time variable directly in ODEs
        from sympy import sin, cos
        from math import pi

        a,b,t = sympy.symbols('a b t')
        odes = {
            a: cos(t),
            b: 1 - t/2, # derivative of -(t/2 - 1)^2 + 2
            t: 1,
        }
        initial_values = {
            a: 0,
            b: 1,
            t: 0,
        }
        t_eval = np.linspace(0, 2*pi, 5) # [0, pi/2, pi, 3*pi/2, 2*pi]
        solution = gpac.integrate_odes(odes, initial_values, t_eval=t_eval)
        print(f'a(pi/2) = {solution.y[0][1]:.2f}')
        print(f'a(pi)   = {solution.y[0][2]:.2f}')
        print(f'b(pi/2) = {solution.y[1][1]:.2f}')
        print(f'b(pi)   = {solution.y[1][2]:.2f}')

    which prints

    .. code-block::

        a(pi/2) = 1.00
        a(pi)   = 0.00
        b(pi/2) = 1.95
        b(pi)   = 1.67

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
            If a symbol appears as a key in `initial_values` but is not a key in `odes`,
            a ValueError is raised.

        t_eval:
            iterable of times at which to evaluate the ODEs.
            At least one of `t_eval` or `t_span` must be specified.

        t_span:
            pair (start_time, end_time) for the integration.
            If not specified, first and last times in `t_eval` are used.
            (This is different from solve_ivp, which requires `t_span` to be specified.)
            At least one of `t_eval` or `t_span` must be specified.

        dependent_symbols:
            iterable of sympy expressions (or strings) representing symbols
            that are functions of the other symbols that are keys in `odes`.
            These values are added to the end of the 2D array field `sol.y` in the object `sol`
            returned by `solve_ivp`, in the order in which they appear in `dependent_variables`.
            For an example, see the example notebook
            https://github.com/UC-Davis-molecular-computing/gpac/blob/main/notebook.ipynb.

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
            This is a catch-all for any additional keyword arguments that are passed to `solve_ivp`,
            for example you could pass `rtol=1e-6` to set the relative tolerance to 1e-6:

            .. code-block:: python

                plot(odes, initial_values, t_eval=t_eval, rtol=1e-6)

            For solver-specific parameters to `solve_ivp`,
            see documentation for `solve_ivp` in scipy.integrate:
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

    independent_symbols = tuple(odes.keys())
    ode_funcs = {symbol: sympy.lambdify(independent_symbols, ode) for symbol, ode in odes.items()}

    def ode_func_vector(_, vals):
        return tuple(ode_func(*vals) for ode_func in ode_funcs.values())

    # sort keys of initial_values according to order of keys in odes,
    # and assume initial value of 0 for any symbol not specified
    initial_values_sorted = [initial_values[symbol] if symbol in initial_values else 0
                             for symbol in independent_symbols]
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

    if dependent_symbols != ():
        dependent_funcs = [sympy.lambdify(independent_symbols, func)
                           for func in dependent_symbols]
        # compute dependent variables and append them to solution.y
        dep_vals = np.zeros(shape=(len(dependent_symbols), len(solution.t)))  # type: ignore
        indp_vals = list(solution.y)  # type: ignore
        for i, func in enumerate(dependent_funcs):
            # convert 2D numpy array to list of 1D arrays so we can use Python's * operator to distribute
            # the vectors as separate arguments to the function func
            dep_vals_row = func(*indp_vals)
            dep_vals[i] = dep_vals_row
        solution.y = np.vstack((solution.y, dep_vals))

    # mypy complains about solution not being an OdeResult, but it is
    return solution  # type:ignore


def plot(
        odes: Dict[Union[sympy.Symbol, str], Union[sympy.Expr, str, float]],
        initial_values: Dict[Union[sympy.Symbol, str], float],
        t_eval: Optional[Iterable[float]] = None,
        t_span: Optional[Tuple[float, float]] = None,
        dependent_symbols: Optional[Dict[Union[sympy.Symbol, str], Union[sympy.Expr, str]]] = None,
        figure_size: Tuple[float, float] = (10, 3),
        symbols_to_plot: Optional[Iterable[Union[sympy.Symbol, str]]] = None,
        show: bool = False,
        method: Union[str, OdeSolver] = 'RK45',
        dense_output: bool = False,
        events: Optional[Union[Callable, Iterable[Callable]]] = None,
        vectorized: bool = False,
        args: Optional[Tuple] = None,
        loc: Union[str, Tuple[float, float]] = 'best',
        **options,
) -> None:
    """
    Numerically integrate the given ODEs using the function :func:`integrate_odes`,
    then plot the trajectories using matplotlib.
    (Assumes it is being run in a Jupyter notebook.)
    See :func:`integrate_odes` for description of parameters below that are not documented.

    Args:
        figure_size:
            pair (width, height) of the figure

        symbols_to_plot:
            symbols to plot; if not specified, then all symbols are plotted

        show:
            whether to call ``matplotlib.pyplot.show()`` after creating the plot;
            If False, this helps the user to call other functions
            such as ``matplotlib.pyplot.legend()`` or ``matplotlib.pyplot.grid()`` after calling this
            function, which will not work if ``matplotlib.pyplot.show()`` has already been called.
            However, if you want to display multiple plots from the same cell in a Jupyter notebook,
            you should either set this to True, or (in case you want to configure each plot by calling
            other matplotlib.pyplot functions, such as yscale), manually call ``matplotlib.pyplot.show()``
            after each call to this function.

        dependent_symbols:
            dict mapping symbols (or strings) to sympy expressions (or strings) representing variables
            that are functions of the other variables that are keys in `odes`.
            For an example, see the example notebook
            https://github.com/UC-Davis-molecular-computing/gpac/blob/main/notebook.ipynb.

        loc:
            location of the legend; see documentation for `matplotlib.pyplot.legend`:
            https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html

        options:
            This is a catch-all for any additional keyword arguments that are passed to `solve_ivp`,
            for example you could pass `rtol=1e-6` to set the relative tolerance to 1e-6:

            .. code-block:: python

                plot(odes, initial_values, t_eval=t_eval, rtol=1e-6)

            For solver-specific parameters to `solve_ivp`,
            see documentation for `solve_ivp` in scipy.integrate:
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html

            Also used for keyword options to `plot` in matplotlib.pyplot:
            https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html.
            However, note that using such arguments here will cause `solve_ivp` to print a warning
            that it does not recognize the keyword argument.
    """
    dependent_symbols_expressions = tuple(dependent_symbols.values()) if dependent_symbols is not None else ()

    sol = integrate_odes(
        odes=odes,
        initial_values=initial_values,
        t_span=t_span,
        t_eval=t_eval,
        dependent_symbols=dependent_symbols_expressions,
        method=method,
        dense_output=dense_output,
        events=events,
        vectorized=vectorized,
        args=args,
        **options,
    )

    symbols = tuple(odes.keys()) + (() if dependent_symbols is None else tuple(dependent_symbols.keys()))
    assert len(symbols) == len(sol.y)
    result = {str(symbol): y for symbol, y in zip(symbols, sol.y)}
    times = sol.t
    plot_given_values(
        times=times,
        result=result,
        source='ode',
        dependent_symbols=dependent_symbols,
        figure_size=figure_size,
        symbols_to_plot=symbols_to_plot,
        show=show,
        loc=loc,
        **options,
    )


# This is used to share plotting code between data returned from scipy.integrate.solve_ivp and that
# returned from gillespy2.Model.run(). This is not intended to be called by the user, but we make it public
# so it's accessible from the crn module.
def plot_given_values(
        times: np.ndarray,
        result: Dict[str, np.ndarray],
        source: Literal['ode', 'ssa'],
        dependent_symbols: Optional[Dict[Union[sympy.Symbol, str], Union[sympy.Expr, str]]] = None,
        figure_size: Tuple[float, float] = (10, 3),
        symbols_to_plot: Optional[Iterable[Union[sympy.Symbol, str]]] = None,
        show: bool = False,
        loc: Union[str, Tuple[float, float]] = 'best',
        **options,
) -> None:
    from matplotlib.pylab import rcParams
    if rcParams['figure.dpi'] != 96:
        print(f"matplotlib.pylab.rcParams['figure.dpi'] was {rcParams['figure.dpi']}; setting it to 96")
        rcParams['figure.dpi'] = 96

    # normalize symbols_to_plot to be a frozenset of strings (names of symbols)
    dependent_symbols_tuple = tuple(dependent_symbols.keys()) if dependent_symbols is not None else ()
    if symbols_to_plot is None:
        symbols_given = tuple(result.keys())
        symbols_to_plot = tuple(symbols_given) + dependent_symbols_tuple
    symbols_to_plot = frozenset(str(symbol) for symbol in symbols_to_plot)
    if len(symbols_to_plot) == 0:
        raise ValueError("symbols_to_plot cannot be empty")

    # check that symbols all appear as keys in result
    symbols_of_results = frozenset(str(symbol) for symbol in result.keys())
    symbols_of_odes_and_dependent_symbols = symbols_of_results | frozenset(
        str(symbol) for symbol in dependent_symbols_tuple)
    diff = symbols_to_plot - symbols_of_odes_and_dependent_symbols
    if len(diff) > 0:
        source = 'ODEs' if source == 'ode' else 'reactions'
        raise ValueError(f"\nsymbols_to_plot contains symbols that are not in odes or dependent symbols: "
                         f"{comma_separated(diff)}"
                         f"\nSymbols in {source}:                                       "
                         f"{comma_separated(symbols_of_results)}"
                         f"\nDependent symbols:                                     "
                         f"{comma_separated(dependent_symbols_tuple)}")

    figure(figsize=figure_size)

    for symbol in symbols_to_plot:
        symbol_name = str(symbol)
        assert symbol_name in result.keys()
        y = result[symbol_name]
        plt.plot(times, y, label=str(symbol), **options)

    plt.xlabel('time')
    plt.legend(loc=loc)
    if show:
        plt.show()


def comma_separated(elts: Iterable[Any]) -> str:
    return ', '.join(str(elt) for elt in elts)
