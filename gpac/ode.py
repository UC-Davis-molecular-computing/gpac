"""
This module has ODE-only functions, mainly [integrate_odes][gpac.ode.integrate_odes] and [plot][gpac.ode.plot].
"""

import re
from typing import Iterable, Callable, Any, Literal, TypeAlias, cast

from scipy.integrate._ivp.ivp import OdeResult  # noqa
import sympy
from scipy.integrate import solve_ivp, OdeSolver
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import xarray

def integrate_odes(
        odes: dict[sympy.Symbol | str, sympy.Expr | str | float | int],
        initial_values: dict[sympy.Symbol | str, float],
        t_eval: Iterable[float] | None = None,
        *,
        t_span: tuple[float, float] | None = None,
        dependent_symbols: Iterable[sympy.Expr | str] = (),
        resets: dict[float, dict[sympy.Symbol | str, float | int]] | None = None,
        method: str | OdeSolver = 'RK45',
        dense_output: bool = False,
        events: Callable | Iterable[Callable] | None = None,
        vectorized: bool = False,
        args: tuple | None = None,
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
    Integrate the given ODEs using scipy's 
    [`solve_ivp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html) 
    function in the package scipy.integrate, returning the same object returned by
    [`solve_ivp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html).

    This is a convienence function that wraps the scipy function
    [`solve_ivp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html),
    allowing the user to specify the ODEs using sympy symbols and expressions
    (instead of a Python function on tuples of floats, which is what 
    [`solve_ivp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html)
    expects, but is more awkward to specify than using sympy expressions).

    The object `solution` returned by `solve_ivp` has field `solution.y` which is a 2D numpy array,
    each row of which is the trajectory of a value in the ODEs. The order of the rows is the same as the
    iteration order of the keys in the `odes` dict.

    Besides the parameters described below,
    all other parameters are simply passed along to `solve_ivp` in scipy.integrate.
    As with that function, the following are explicitly named parameters:
    `method`, `dense_output`, `events`, `vectorized`, `args`, and
    all other keyword arguments are passed in through `**options`; see the
    documentation for solve_ivp for a description of these parameters:
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html>

    ```py title="rock-paper-scissors oscillator example"
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
    print(gpac.integrate_odes(odes, initial_values, t_eval=t_eval))
    ```

    This outputs 
    ```
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
    ```

    All symbols are interpreted as functions of a single variable called "time", and the derivatives
    are with respective to time.

    Although you cannot reference the time variable directly in the ODEs, this can be simulated
    by introducing a new variable `t` whose derivative is 1 and initial value is the initial time.
    For example, the following code implements ``a(t) = sin(t)`` (with time derivative ``a'(t) = cos(t)``)
    and ``b(t) = -(t/2 - 1)^2 + 2`` (with time derivative ``b'(t) = 1 - t/2``). In this case we want
    `t`=0 initially, and any symbol that is not a key in `initial_values` is assumed to have initial value 0.

    ```python
    # trick for referencing time variable directly in ODEs
    from sympy import sin, cos
    from math import pi

    a,b,t = sympy.symbols('a b t')
    odes = {
        a: cos(t),
        b: 1 - t/2, # derivative of -(t/2 - 1)^2 + 2
        t: 1,
    }
    initial_values = { b: 1 }
    t_eval = np.linspace(0, 2*pi, 5) # [0, pi/2, pi, 3*pi/2, 2*pi]
    solution = gpac.integrate_odes(odes, initial_values, t_eval=t_eval)
    print(f'a(pi/2) = {solution.y[0][1]:.2f}')
    print(f'a(pi)   = {solution.y[0][2]:.2f}')
    print(f'b(pi/2) = {solution.y[1][1]:.2f}')
    print(f'b(pi)   = {solution.y[1][2]:.2f}')
    ```

    which prints

    ```
    a(pi/2) = 1.00
    a(pi)   = 0.00
    b(pi/2) = 1.95
    b(pi)   = 1.67
    ```

    Parameters
    ----------
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

    resets:
        If specified, this is a dict mapping times to "configurations" (i.e., dict mapping symbols/str to values).
        The configurations are used to set the values of the symbols manually during the ODE integration
        at specific times.
        Any symbols not appearing as keys in `resets` are left at their current values.
        The OdeResult returned (the one returned by `solve_ivp` in scipy) will have two additional fields:
        `reset_times` and `reset_indices`, which are lists of the times and indices in `sol.t`
        corresponding to the times when the resets were applied.
        Raises a ValueError if any time lies outside the integration interval, or if `resets` is empty.

    method:
        See [`solve_ivp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html)

    dense_output:
        See [`solve_ivp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html)

    events:
        See [`solve_ivp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html)

    vectorized:
        See [`solve_ivp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html)

    args:
        See [`solve_ivp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html)

    options:
        This is a catch-all for any additional keyword arguments that are passed to `solve_ivp`,
        for example you could pass `rtol=1e-6` to set the relative tolerance to 1e-6:

        ```py
        plot(odes, initial_values, t_eval=t_eval, rtol=1e-6)
        ```

        For solver-specific parameters,
        see [`solve_ivp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html)

    Returns
    -------
    :
        solution to the ODEs, same as object returned by
        [`solve_ivp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html)
    """
    if t_eval is not None:
        t_eval = np.array(t_eval)
        if not np.all(np.diff(t_eval) >= 0):
            raise ValueError("t_eval must be sorted in increasing order")

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

    independent_symbols = cast(tuple[sympy.Symbol, ...], tuple(odes.keys()))
    ode_funcs = {symbol: sympy.lambdify(independent_symbols, ode) for symbol, ode in odes.items()}

    def ode_func_vector(_, vals):
        return tuple(ode_func(*vals) for ode_func in ode_funcs.values())

    # sort keys of initial_values according to order of keys in odes,
    # and assume initial value of 0 for any symbol not specified
    initial_values_sorted = [initial_values[symbol] if symbol in initial_values else 0
                             for symbol in independent_symbols]

    if resets is None:
        solution = solve_ivp(
            fun=ode_func_vector,
            t_span=t_span,
            y0=initial_values_sorted,
            t_eval=t_eval,
            method=method,  # type:ignore
            dense_output=dense_output,
            events=events,
            vectorized=vectorized,
            args=args,
            **options,
        )
    else:
        symbol_to_idx = {symbol: i for i, symbol in enumerate(independent_symbols)}
        for reset in resets.values():
            for symbol, value in reset.items():
                if isinstance(symbol, str):
                    del reset[symbol]
                    symbol = sympy.symbols(symbol)
                    reset[symbol] = value
                if symbol not in symbol_to_idx:
                    raise ValueError(f"Symbol {symbol} not found in odes")
        solution = _solve_ivp_with_resets(
            resets=resets,  # type:ignore
            symbol_to_idx=symbol_to_idx,
            fun=ode_func_vector,
            t_span=t_span,  # type:ignore
            y0=initial_values_sorted,  # type:ignore
            t_eval=t_eval,
            method=method,
            dense_output=dense_output,
            events=events,
            vectorized=vectorized,
            args=args,
            **options,
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


import numpy as np
from scipy.integrate import solve_ivp
from typing import Callable, Iterable, Any


def _solve_ivp_with_resets(
        resets: dict[float, dict[sympy.Symbol, float | int]],
        symbol_to_idx: dict[sympy.Symbol, int],
        fun: Callable,
        t_span: tuple[float, float],
        y0: np.ndarray,
        t_eval: Iterable[float] | None = None,
        dense_output: bool = False,
        events: Callable | Iterable[Callable] | None = None,
        vectorized: bool = False,
        args: tuple | None = None,
        **options
) -> Any:
    """
    Solve an initial value problem with parameter resets at specific times.
    Similar to scipy's solve_ivp but allows resetting variable values at specific times.

    Parameters
    ----------
    resets:
        Dictionary mapping time points to dictionaries of variable resets
        Each inner dict maps symbols to new values
    symbol_to_idx:
        Dictionary mapping symbols to their indices in the y0 array
    fun:
        Right-hand side of the system, same as in solve_ivp
    t_span:
        Interval of integration (t0, tf)
    y0: array_like, shape (n,)
        Initial state
    t_eval:
        Times at which to store the computed solution, must be sorted and lie within t_span
    dense_output: bool
        Whether to compute a continuous solution
    events:
        Events to track
    vectorized:
        Whether fun is implemented in a vectorized fashion
    args:
        Additional arguments to pass to fun
    **options:
        Additional options to pass to the solver

    Returns
    -------
    :
    Solution with combined results and additional attributes:
    - sol.reset_times: List of times when resets were applied
    - sol.reset_indices: List of indices in sol.t corresponding to reset points

    Raises
    ------
        ValueError
            If resets is empty or if any reset time is outside the integration interval t_span
    """
    # Check if resets is empty
    if len(resets) == 0:
        raise ValueError("resets dictionary must not be empty")

    for reset_time, reset in resets.items():
        if len(reset) == 0:
            raise ValueError(f"Each reset dict must be nonempty, "
                             f"but reset time {reset_time} has an empty dict.")

    # Validate reset times are within the integration interval
    for reset_time in resets.keys():
        if reset_time <= t_span[0] or reset_time >= t_span[1]:
            raise ValueError(f"Reset time {reset_time} is outside the integration interval {t_span}")

    # Extract reset times and sort them
    reset_times = sorted(resets.keys())

    # Break the interval into segments
    segments = []
    for i in range(len(reset_times) + 1):
        if i == 0:
            segments.append((t_span[0], reset_times[0]))
        elif i == len(reset_times):
            segments.append((reset_times[-1], t_span[1]))
        else:
            segments.append((reset_times[i - 1], reset_times[i]))

    # Initialize results storage
    all_t = []
    all_y = []
    reset_indices = []
    total_nfev = 0
    total_njev = 0
    total_nlu = 0

    # If t_eval is provided, create segments for it
    if t_eval is not None:
        t_eval = np.array(t_eval)
        t_eval_segments = []

        for i, (start, end) in enumerate(segments):
            # Get values in this segment
            mask = (t_eval >= start) & (t_eval <= end)
            segment_t_eval = t_eval[mask]

            # Add boundary points if needed
            if len(segment_t_eval) == 0:
                # If there are no points, include both ends
                segment_t_eval = np.array([start, end])
            elif segment_t_eval[0] != start:
                # If the start point is missing, add it
                segment_t_eval = np.insert(segment_t_eval, 0, start)
            elif segment_t_eval[-1] != end:
                # If the end point is missing, add it
                segment_t_eval = np.append(segment_t_eval, end)

            t_eval_segments.append(segment_t_eval)
    else:
        # If t_eval is not provided, don't use it for segments
        t_eval_segments = [None] * len(segments)

    # Integrate each segment
    current_y = y0.copy()

    for i, ((t_start, t_end), segment_t_eval) in enumerate(zip(segments, t_eval_segments)):
        # Solve for this segment
        segment_sol = solve_ivp(
            fun=fun,
            t_span=(t_start, t_end),
            y0=current_y,
            t_eval=segment_t_eval,
            dense_output=dense_output,
            events=events,
            vectorized=vectorized,
            args=args,
            **options
        )

        if not segment_sol.success:
            # If integration failed, return the failed solution with reset info
            segment_sol.reset_times = reset_times[:i]
            segment_sol.reset_indices = reset_indices
            return segment_sol

        # Add results to our collection
        all_t.append(segment_sol.t)
        all_y.append(segment_sol.y)

        # Update counters
        total_nfev += segment_sol.nfev
        total_njev += getattr(segment_sol, 'njev', 0)
        total_nlu += getattr(segment_sol, 'nlu', 0)

        # If not the last segment, apply resets for the next segment
        if i < len(segments) - 1:
            reset_time = reset_times[i]
            reset_indices.append(len(np.concatenate(all_t)) - 1)

            # Get the final state of this segment
            current_y = segment_sol.y[:, -1].copy()

            # Apply the resets using the symbol_to_idx mapping
            reset_dict = resets[reset_time]
            for symbol, new_value in reset_dict.items():
                assert symbol in symbol_to_idx
                idx = symbol_to_idx[symbol]
                current_y[idx] = new_value

    # Combine all results
    combined_t = np.concatenate(all_t)
    combined_y = np.hstack(all_y)

    # Get the final solution object and update its fields
    final_sol = segment_sol
    final_sol.t = combined_t
    final_sol.y = combined_y
    final_sol.nfev = total_nfev
    if hasattr(final_sol, 'njev'):
        final_sol.njev = total_njev
    if hasattr(final_sol, 'nlu'):
        final_sol.nlu = total_nlu

    # Add reset information
    final_sol.reset_times = reset_times
    final_sol.reset_indices = reset_indices

    # Handle dense output if requested
    if dense_output:
        from scipy.integrate._ivp.common import DenseOutput

        class CombinedDenseOutput(DenseOutput):
            def __init__(self, ts, ys):
                super().__init__(ts[0], ts[-1])
                self.ts = ts
                self.ys = ys
                self.n = ys.shape[0]

            def _call_impl(self, t):
                # Find where t belongs
                if np.isscalar(t):
                    idx = np.searchsorted(self.ts, t) - 1
                    if idx < 0:
                        idx = 0
                    if idx >= len(self.ts) - 1:
                        idx = len(self.ts) - 2

                    t0, t1 = self.ts[idx], self.ts[idx + 1]
                    y0, y1 = self.ys[:, idx], self.ys[:, idx + 1]

                    # Linear interpolation
                    return y0 + (y1 - y0) * (t - t0) / (t1 - t0)
                else:
                    result = np.zeros((self.n, len(t)))
                    for i, ti in enumerate(t):
                        result[:, i] = self._call_impl(ti)
                    return result

        final_sol.sol = CombinedDenseOutput(combined_t, combined_y)

    return final_sol


def plot(
        odes: dict[sympy.Symbol | str, sympy.Expr | str | float | int],
        initial_values: dict[sympy.Symbol | str, float | int],
        t_eval: Iterable[float] | None = None,
        *,
        t_span: tuple[float, float] | None = None,
        resets: dict[float, dict[sympy.Symbol | str, float | int]] | None = None,
        dependent_symbols: dict[sympy.Symbol | str, sympy.Expr | str] | None = None,
        figure_size: tuple[float, float] = (10, 3),
        latex_legend: bool = False,
        symbols_to_plot:
        Iterable[sympy.Symbol | str] |
        Iterable[Iterable[sympy.Symbol | str]] |
        str |
        re.Pattern |
        Iterable[re.Pattern]
        | None = None,
        legend: dict[sympy.Symbol | str, str] | None = None,
        show: bool = False,
        method: str | OdeSolver = 'RK45',
        dense_output: bool = False,
        events: Callable | Iterable[Callable] | None = None,
        vectorized: bool = False,
        return_ode_result: bool = False,
        args: tuple | None = None,
        loc: str | tuple[float, float] = 'best',
        warn_change_dpi: bool = False,
        **options,
) -> OdeResult | None:
    """
    Numerically integrate the given ODEs using the function [integrate_odes][gpac.ode.integrate_odes],
    then plot the trajectories using matplotlib.
    (Assumes it is being run in a Jupyter notebook.)
    See [integrate_odes][gpac.ode.integrate_odes] for description of parameters below that are not documented.

    Parameters
    ----------

    figure_size:
        pair (width, height) of the figure
        
    latex_legend:
        If True, surround each symbol name with dollar signs, unless it is already surrounded with them, 
        so that the legend is interpreted as LaTeX. If this is True, then the symbol name must either start and end
        with `$`, or neither start nor end with `$`.

    symbols_to_plot:
        symbols to plot; if not specified, then all symbols are plotted.
        If it is a 2D list (or other Iterable of Iterables of strings or symbols),
        then each group of symbols is plotted in a separate subplot.
        If a string or re.Pattern, then only symbols whose names match the string or pattern are plotted.
        
    legend:
        If specified, should be a dict mapping symbols (or strings) to strings.
        For each symbol that is plotted, the corresponding string is used as the label in the plot's legend
        instead of the original name of the symbol. This can be useful for example to include LaTeX,
        mapping a symbol with a name like `'xt'` to a string like `r'$x_t$'`.

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
        <https://github.com/UC-Davis-molecular-computing/gpac/blob/main/notebook.ipynb>.

    return_ode_result:
        if True, returns solution to the ODEs, same as object returned by 
        [`solve_ivp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html)
        in scipy.integrate.
        Otherwise (default) None is returned. The reason the solution is not automatically returned is that
        it pollutes the output of a Jupyter notebook, so this avoids needing to type something like
        ``_ = gpac.plot(...)`` to suppress the output.
        But if you want that solution object, you can set this to True.

    loc:
        location of the legend; see documentation for [matplotlib.pyplot.legend](
        https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html)

    warn_change_dpi:
        If True, print a warning if the dpi of the figure gets changed from its default.

    options:
        This is a catch-all for any additional keyword arguments that are passed to `solve_ivp`,
        for example you could pass `rtol=1e-6` to set the relative tolerance to 1e-6:

        ```py
        plot(odes, initial_values, t_eval=t_eval, rtol=1e-6)
        ```

        For solver-specific parameters to `solve_ivp`,
        see documentation for [`solve_ivp`](
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html)

        Also used for keyword options to []`matplotlib.pyplot.plot]()
        https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html).
        However, note that using such arguments here will cause `solve_ivp` to print a warning
        that it does not recognize the keyword argument.

    Returns
    -------
    :
        Typically None, but if `return_ode_result` is True, returns the
        solution to the ODEs, same as object returned by 
        [`solve_ivp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html).
    """
    if legend is None:
        legend = {}

    dependent_symbols_expressions = tuple(dependent_symbols.values()) if dependent_symbols is not None else ()

    sol = integrate_odes(
        odes=odes,
        initial_values=initial_values,
        t_span=t_span,
        t_eval=t_eval,
        dependent_symbols=dependent_symbols_expressions,
        resets=resets,
        method=method,
        dense_output=dense_output,
        events=events,
        vectorized=vectorized,
        args=args,
        **options,
    )

    # TODO: add a check that the symbols in dependent_symbols are in the result

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
        latex_legend=latex_legend,
        symbols_to_plot=symbols_to_plot,
        legend=legend,
        show=show,
        loc=loc,
        warn_change_dpi=warn_change_dpi,
        **options,
    )
    return sol if return_ode_result else None


# This is used to share plotting code between data returned from scipy.integrate.solve_ivp and that
# returned from gillespy2.Model.run(). This is not intended to be called by the user, but we make it public
# so it's accessible from the crn module.
def plot_given_values(
        *,
        times: np.ndarray | xarray.DataArray,
        result: dict[str, np.ndarray | xarray.DataArray],
        source: Literal['ode', 'ssa'],
        dependent_symbols: dict[sympy.Symbol | str, sympy.Expr | str] | None = None,
        figure_size: tuple[float, float] = (10, 3),
        latex_legend: bool = False,
        symbols_to_plot:
        Iterable[sympy.Symbol | str] |
        Iterable[Iterable[sympy.Symbol | str]] |
        str |
        re.Pattern |
        Iterable[re.Pattern] | None
        = None,
        legend: dict[sympy.Symbol | str, str] | None = None,
        show: bool = False,
        loc: str | tuple[float, float] = 'best',
        warn_change_dpi: bool = False,
        **options,
) -> None:
    if legend is None:
        legend = {}
    from matplotlib.pylab import rcParams
    if rcParams['figure.dpi'] != 96:
        if warn_change_dpi:
            print(f"matplotlib.pylab.rcParams['figure.dpi'] was {rcParams['figure.dpi']}; setting it to 96")
        rcParams['figure.dpi'] = 96

    # normalize symbols_to_plot to be a frozenset of strings (names of symbols)
    dependent_symbols_tuple = tuple(dependent_symbols.keys()) if dependent_symbols is not None else ()
    if symbols_to_plot is None:
        symbols_given = tuple(result.keys())
        symbols_to_plot = tuple(symbols_given) + dependent_symbols_tuple
    elif isinstance(symbols_to_plot, (str, re.Pattern)):
        pattern = symbols_to_plot if isinstance(symbols_to_plot, re.Pattern) else re.compile(symbols_to_plot)
        symbols_to_plot = [symbol for symbol in result.keys() if pattern.match(symbol)]

    empty = True
    for _ in symbols_to_plot:
        empty = False
        break
    if empty:
        raise ValueError("symbols_to_plot cannot be empty")

    # if symbols_to_plot is an Iterable of Iterables of strings or symbols, make a separate subplot for each
    multiple_subplots = False
    for symbol in symbols_to_plot:
        from gpac import Specie
        if not isinstance(symbol, (sympy.Symbol, Specie, str)):
            multiple_subplots = True
            break

    if multiple_subplots:
        new_symbols_to_plot = []
        assert not isinstance(symbols_to_plot, (str, re.Pattern))
        for symbol_group in symbols_to_plot:
            if isinstance(symbol_group, re.Pattern):
                symbol_group = [symbol for symbol in result.keys() if symbol_group.match(symbol)]
            else:
                try:
                    _ = iter(symbol_group)
                except TypeError as te:
                    raise ValueError(f"expected elements of symbols_to_plot to be iterable, but got "
                                     f"'{symbol_group}', which is of type {type(symbol_group)}") from te
                symbol_group = [str(symbol) for symbol in symbol_group]
            new_symbols_to_plot.append(symbol_group)
        symbols_to_plot = new_symbols_to_plot
        for symbol_group in symbols_to_plot:
            if len(symbol_group) == 0:
                raise ValueError(f"Each group of symbols to plot must be non-empty, "
                                 f"but symbols_to_plot = {symbols_to_plot}")
    else:
        symbols_to_plot = [[str(symbol) for symbol in symbols_to_plot]]

    # check that symbols all appear as keys in result
    all_symbols_to_plot_set = frozenset(str(symbol) for symbol_group in symbols_to_plot for symbol in symbol_group)
    symbols_of_results_set = frozenset(str(symbol) for symbol in result.keys())
    dependent_symbols_set = frozenset(str(symbol) for symbol in dependent_symbols_tuple)
    symbols_of_odes_and_dependent_symbols = symbols_of_results_set | dependent_symbols_set
    diff = all_symbols_to_plot_set - symbols_of_odes_and_dependent_symbols
    if len(diff) > 0:
        source = 'ODEs' if source == 'ode' else 'reactions'
        raise ValueError(f"\nsymbols_to_plot contains symbols that are not in odes or dependent symbols: "
                         f"{comma_separated(diff)}"
                         f"\nSymbols in {source}:                                       "
                         f"{comma_separated(symbols_of_results_set)}"
                         f"\nDependent symbols:                                     "
                         f"{comma_separated(dependent_symbols_tuple)}")

    figure(figsize=figure_size)

    colors = plt.rcParams["axes.prop_cycle"]()
    num_subplots = len(symbols_to_plot)
    for idx, symbol_group in enumerate(symbols_to_plot):
        if num_subplots > 1:
            plt.subplot(num_subplots, 1, idx + 1)

        for symbol in symbol_group:
            symbol_name = str(symbol)
            assert symbol_name in result.keys()
            y = result[symbol_name]
            assert len(y) == len(times)
            color = next(colors)["color"]
            if symbol in legend:
                symbol_name = legend[symbol]
            if symbol_name in legend:
                symbol_name = legend[symbol_name]
            if isinstance(symbol, str) and  sympy.Symbol(symbol) in legend:
                symbol_name = legend[sympy.Symbol(symbol)]
            if latex_legend:
                if symbol_name[0] == '$' or symbol_name[-1] == '$':
                    if not symbol_name[0] == '$' and symbol_name[-1] == '$':
                        raise ValueError(f'symbol name "{symbol_name}" must either end with $, '
                                         f'or neither start nor end with $')
                else:
                    symbol_name = f'${symbol_name}$'
            plt.plot(times, y, label=symbol_name, color=color, **options)

        plt.legend(loc=loc)

    plt.xlabel('time')

    if show:
        plt.show()


def comma_separated(elts: Iterable[Any]) -> str:
    return ', '.join(str(elt) for elt in elts)


def bubble():
    import numpy as np
    import sympy
    # bubble sort values x1, x2, x3, x4
    x1, x2, x3, x4, y12, y23, y34 = sympy.symbols('x1 x2 x3 x4 y12 y23 y34')

    odes = {
        x1: -y12,
        x2: -y23 + y12,
        x3: -y34 + y23,
        x4: y34,
        y12: (x1 - x2) * y12,
        y23: (x2 - x3) * y23,
        y34: (x3 - x4) * y34,
    }
    eps = 0.001
    initial_values = {
        x1: 3,
        x2: 7,
        x3: 2,
        x4: 1,
        y12: eps,
        y23: eps,
        y34: eps,
    }
    t_eval = np.linspace(0, 30, 500)
    # for clarity, you can pass a 2D list for symbols_to_plot
    # each group of symbols will be shown in separate subplots stacked vertically
    _ = plot(odes, initial_values, t_eval, figure_size=(12, 4),
                  symbols_to_plot=[[x1, x2, x3, x4], [y12, y23, y34]])

def display_odes(odes: dict[sympy.Symbol | str, sympy.Expr | str | float | int]) -> None:
    """
    Display the ODEs in a readable format in a Jupyter notebook.

    Parameters
    ----------
    odes:
        dict mapping sympy symbols (or strings) to sympy expressions (or strings or floats) representing the ODEs.
        Alternatively, the keys can be strings, and the values can be strings that look like expressions,
        e.g., ``{'a': '-a*b + c*a'}``.
    """
    from IPython.display import display, Math
    for symbol, expr in odes.items():
        # normalize so symbol is a sympy Symbol and expr is a sympy Expression
        if isinstance(symbol, str):
            symbol = sympy.symbols(symbol)
        if isinstance(expr, (str, float, int)):
            expr = sympy.sympify(expr)

        symbol_latex = sympy.latex(symbol)
        expr_latex = sympy.latex(expr)
        ode_latex = f"{symbol_latex}' = {expr_latex}"
        display(Math(ode_latex))


if __name__ == '__main__':
    bubble()