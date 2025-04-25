r"""
Module for expressing chemical reaction networks and deriving their ODEs. Ideas and much code taken from
[this repo](https://github.com/enricozb/python-crn).

For example, to specify the "approximate majority" chemical reaction network
(see [DOI: 10.1007/978-3-540-75142-7_5](https://doi.org/10.1007/978-3-540-75142-7_5) or
[DOI: 10.1126/science.aal2052](https://doi.org/10.1126/science.aal2052))

$$
\begin{align*}
    A+B &\to 2U \\
    A+U &\to 2A \\
    B+U &\to 2B
\end{align*}
$$

we can write

```py
a, b, u = species('A B U')
rxns = [
    a+b >> 2*u,
    a+u >> 2*a,
    b+u >> 2*b,
]
initial_values = {a: 0.51, b: 0.49}
t_eval = np.linspace(0, 10, 100)
gpac.plot_crn(rxns, initial_values, t_eval)
```

which will plot the concentrations of A, B, and U over time. One can specify reversible reactions
by using the `|` operator instead of `>>` (e.g., `#!py a+b | 2*u`) and rate constants using the functions
`k` (for forward rate constants) and `r` (for reverse rate constants),
e.g., `#!py (a+b | 2*u).k(1.5).r(0.5)`.

See functions [crn_to_odes][gpac.crn.crn_to_odes] to convert reactions to ODEs (ordinary differential equations),
[integrate_crn_odes][gpac.crn.integrate_crn_odes] to get the trajectories of integrating these ODEs over time, and
[plot_crn][gpac.crn.plot_crn] to plot the trajectories. The documentation for [crn_to_odes][gpac.crn.crn_to_odes]
explains how reactions are converted into ODEs by each of these functions.

Also supported are inhibitors, which can be added to reactions using the method [`Reaction.i`](gpac.crn.Reaction.i):

```py
a, b, u, i = species('A B U I')
rxn = (a+b | 2*u).i(i, 100)
```

which represents the reaction $A+B \to 2U$ with inhibitor $I$ and inhibitor constant 100.
Currently the inhibitor is modeled using a first-order Hill function, i.e., its contribution to the
reaction rate is to divide by $1 + i \cdot I$, where $i$ is the inhibitor constant.
So for the reaction defined above, its rate is $[A] \cdot [B] / (1 + 100 \cdot [I])$.
"""

from __future__ import annotations  # needed for forward references in type hints

from typing import Iterable, Callable, Literal, TypeAlias
from collections import defaultdict
import copy
from dataclasses import dataclass, field
import re

import xarray
from scipy.integrate import OdeSolver
from scipy.integrate._ivp.ivp import OdeResult  # noqa
import sympy
import gillespy2 as gp
import rebop as rb
import xarray as xr

from gpac.ode import integrate_odes, plot, plot_given_values


def species(sp: str | Iterable[str]) -> tuple[Specie, ...] | Specie:
    r"""
    Create a tuple of [`Specie`](gpac.crn.Specie) (Single species [`Expression`](gpac.crn.Expression)'s),
    or a single [`Specie`](gpac.crn.Specie) object.

    Examples
    --------

    ```py
    w, x, y, z = species('W X Y Z')
    rxn = x + y >> z + w
    ```

    ```py
    w, x, y, z = species(['W', 'X', 'Y', 'Z'])
    rxn = x + y >> z + w
    ```

    ```py
    x = species('X')
    rxn = x >> 2*x
    ```

    Parameters
    ----------
    sp:
        A string or Iterable of strings representing the names of the species being created.
        If a single string, species names are interpreted as space-separated.

    Returns
    -------
    :
        tuple of [`Specie`](gpac.crn.Specie) objects, or a single [`Specie`](gpac.crn.Specie) object.
    """
    species_list: list[str]
    if isinstance(sp, str):
        species_list = sp.split()
    else:
        species_list = [specie.strip() for specie in sp]

    # if len(species_list) == 1:
    #     return Specie(species_list[0])
    if len(species_list) != len(set(species_list)):
        raise ValueError(f'species_list {species_list} cannot contain duplicates.')

    if len(species_list) > 1:
        return tuple(Specie(specie) for specie in species_list)
    else:
        return Specie(species_list[0])


def crn_to_odes(rxns: Iterable[Reaction]) -> dict[sympy.Symbol, sympy.Expr]:
    r"""
    Given a set of chemical reactions, return the corresponding ODEs.

    Each reaction contributes one term to the ODEs for each species produced or consumed in it.
    The term from a reaction appearing in the ODE for species `X` is the product of:

    - the rate constant,
    - the reactant concentrations, and
    - the net stoichiometry of species `X` in the reaction
      (i.e., the net amount of `X` produced by the reaction, negative if consumed).

    For example, consider the following two reactions with respective rate constants
    :math:`k_1` and :math:`k_2`:

    $$
    \begin{align*}
        X+X &\xrightarrow{k_1} C
        \\
        C+X &\xrightarrow{k_2} C+Y
    \end{align*}
    $$

    The net stoichiometry of `X` in the first reaction is -2, since two copies of `X` are consumed,
    and the net stoichiometry of `C` in that reaction is 1, since one copy of `C` is produced.
    The net stoichiometry of `C` in the *second* reaction is 0, since it is a catalyst
    (neither produced nor consumed).

    This corresponds to ODEs (following the convention of lowercase letter `x`
    for the concentration of species `X`):

    $$
    \begin{align*}
        x' &= -2 k_1 x^2 - k_2 c x
        \\
        c' &= k_1 x^2
        \\
        y' &= k_2 c x
    \end{align*}
    $$

    In the package, this can be implemented (for example setting :math:`k_1 = 1.5` and :math:`k_2 = 0.2`)
    via:

    ```py
    x, y, c = species('X Y C')
    rxns = [
        (x+x >> c).k(1.5),
        (c+x >> c+y).k(0.2),
    ]
    odes = crn_to_odes(rxns)
    for symbol, ode in odes.items():
        print(f"{symbol}' = {ode}")
    ```

    which prints

    ```
    X' = -0.2*C*X - 3.0*X**2
    C' = 1.5*X**2
    Y' = 0.2*C*X
    ```

    Parameters
    ----------
    rxns: list of [`Reaction`'s](gpac.crn.Reaction) comprising the chemical reaction network.
          See documentation for [`Reaction`'s](gpac.crn.Reaction) for details on how to specify reactions.

    Returns
    -------
    :
        Dictionary mapping each species (represented as a sympy Symbol object, rather than a [`Specie`](gpac.crn.Specie)
        object) to its corresponding ODE (represented as a sympy Expression).
        This object can be given as the parameter `odes` to the functions 
        [`integrate_odes`][gpac.ode.integrate_odes]
        and 
        [`plot`][gpac.ode.plot] to integrate/plot the ODEs.
        (which is essentially all the functions [`integrate_crn_odes`](gpac.crn.integrate_crn_odes) 
        and [`plot_crn`](gpac.crn.plot_crn) do.
    """
    # map each symbol to list of reactions in which it appears
    specie_to_rxn: dict[Specie, list[Reaction]] = defaultdict(list)
    for rxn in rxns:
        for specie in rxn.get_species():
            specie_to_rxn[specie].append(rxn)

    odes = {}
    for specie, rxns in specie_to_rxn.items():
        ode = sympy.sympify(0)
        for rxn in rxns:
            ode += rxn.get_ode(specie)
            if rxn.reversible:
                ode += rxn.get_ode(specie, reverse=True)
        symbol = sympy.Symbol(specie.name)
        odes[symbol] = ode

    return odes


def _normalize_crn_initial_values(initial_values: dict[Specie | sympy.Symbol | str, float]) \
        -> dict[sympy.Symbol, float]:
    normalized_initial_values = {}
    for symbol, conc in initial_values.items():
        if isinstance(symbol, Specie):
            symbol = sympy.Symbol(symbol.name)
        elif isinstance(symbol, str):
            symbol = sympy.Symbol(symbol)
        normalized_initial_values[symbol] = conc
    return normalized_initial_values


def integrate_crn_odes(
        rxns: Iterable[Reaction],
        initial_values: dict[Specie, float],
        t_eval: Iterable[float] | None = None,
        *,
        t_span: tuple[float, float] | None = None,
        method: str | OdeSolver = 'RK45',
        dense_output: bool = False,
        events: Callable | Iterable[Callable] | None = None,
        vectorized: bool = False,
        args: tuple | None = None,
        **options,
) -> OdeResult:
    """
    Integrate the ODEs derived from to the given set of chemical reactions.
    This calls [integrate_odes][gpac.ode.integrate_odes] with the ODEs derived from the given reactions via
    [crn_to_odes][gpac.crn.crn_to_odes].
    See [integrate_odes][gpac.ode.integrate_odes] for description of parameters other than 
    `rxns` and `initial_values`.

    Parameters
    ----------
    rxns:
        list of [`Reaction`](gpac.crn.Reaction)'s comprising the chemical reaction network.
        See documentation for [`Reaction`](gpac.crn.Reaction) for details on how to specify reactions.

    initial_values:
        dict mapping each species to its initial concentration.
        Note that unlike the parameter `initial_values` in [`integrate_odes`](gpac.ode.integrate_odes),
        keys in this dict must be [`Specie`](gpac.crn.Specie) objects, not strings or sympy symbols.

    Returns
    -------
    :
        The result of the integration.
        See [`integrate_odes`](gpac.ode.integrate_odes) for details about this parameter.
    """
    odes = crn_to_odes(rxns)
    initial_values = _normalize_crn_initial_values(initial_values)
    return integrate_odes(
        odes,
        initial_values=initial_values,
        t_eval=t_eval,
        t_span=t_span,
        method=method,
        dense_output=dense_output,
        events=events,
        vectorized=vectorized,
        args=args,
        **options
    )


def plot_crn(
        rxns: Iterable[Reaction],
        initial_values: dict[Specie, float],
        t_eval: Iterable[float] | None = None,
        *,
        t_span: tuple[float, float] | None = None,
        resets: dict[float, dict[sympy.Symbol | str, float]] | None = None,
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
        show: bool = False,
        legend: dict[sympy.Symbol | str, str] | None = None,
        method: str | OdeSolver = 'RK45',
        dense_output: bool = False,
        events: Callable | Iterable[Callable] | None = None,
        vectorized: bool = False,
        return_ode_result: bool = False,
        args: tuple | None = None,
        loc: str | tuple[float, float] = 'best',
        warn_change_dpi: bool = False,
        **options,
) -> OdeResult:
    r"""
    Plot the ODEs derived from to the given set of chemical reactions.
    This calls [`plot`][gpac.ode.plot] with the ODEs derived from the given reactions via
    [`crn_to_odes`][gpac.crn.crn_to_odes].

    See [`integrate_crn_odes`][gpac.crn.integrate_crn_odes], 
    [`integrate_odes`][gpac.ode.integrate_odes], and 
    [`plot`][gpac.ode.plot] for description of parameters shared with those functions.
    As with [`plot`][gpac.ode.plot], the keyword arguments in `options` are passed to
    [`matplotlib.pyplot.plot`](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html),
    as well as to
    [`scipy.integrate.solve_ivp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html),
    and as with [plot][gpac.ode.plot], keyword arguments not recognized by `solve_ivp`
    (such as those intended for matplotlib.pyplot.plot) cause `solve_ivp` to print a warning that it
    does not recognize the argument.

    Note that the parameter `dependent_symbols` should use sympy symbols, not 
    [`Specie`][gpac.crn.Specie] objects.
    Here is an example of how to use this parameter. Each species that a dependent symbol depends on
    should be represented by a sympy symbol with the same name as the corresponding 
    [`Specie`][gpac.crn.Specie] object:

    ```py
    Xp,Xm,Yp,Ym = gpac.species('Xp Xm Yp Ym')
    x,y,xp,xm,yp,ym = sympy.symbols('x y Xp Xm Yp Ym')

    # dual-rail CRN implementation of sine/cosine oscillator
    # x' = -y
    # y' = x
    rxns = [
        Yp >> Yp + Xm,
        Ym >> Ym + Xp,
        Xp >> Xp + Yp,
        Xm >> Xm + Ym,
        Xp+Xm >> gpac.empty,
        Yp+Ym >> gpac.empty,
    ]
    inits = { Xp: 1, Yp: 0 }
    from math import pi
    t_eval = np.linspace(0, 6*pi, 200)

    dependent_symbols = {
        x: xp - xm,
        y: yp - ym,
    }

    gpac.plot_crn(rxns, inits, t_eval, dependent_symbols=dependent_symbols, symbols_to_plot=[x,y])
    ```

    Parameters
    ----------
    dependent_symbols:
        dict mapping each symbol to an expression that defines its value in terms of other symbols.
        Note that these are not [`Specie`](gpac.crn.Specie) objects as in the parameter `rxns`, but sympy symbols.
        Symbols used in the expressions must have the same name as [`Specie`](gpac.crn.Specie) objects in `rxns`.

    Returns
    -------
    :
        None, or the result of the integration, which is the same as the result of [`integrate_odes`](gpac.ode.integrate_odes)
        if `return_ode_result` is True. See [`integrate_odes`](gpac.ode.integrate_odes) for details about this parameter.
    """
    odes = crn_to_odes(rxns)
    initial_values = _normalize_crn_initial_values(initial_values)
    return plot(
        odes,
        initial_values=initial_values,
        t_eval=t_eval,
        resets=resets,
        t_span=t_span,
        dependent_symbols=dependent_symbols,
        figure_size=figure_size,
        latex_legend=latex_legend,
        symbols_to_plot=symbols_to_plot,
        legend=legend,
        show=show,
        method=method,
        dense_output=dense_output,
        events=events,
        vectorized=vectorized,
        return_ode_result=return_ode_result,
        args=args,
        loc=loc,
        warn_change_dpi=warn_change_dpi,
        **options,
    )


def find_all_species(rxns: Iterable[Reaction]) -> tuple[Specie, ...]:
    all_species = []
    all_species_set = set()
    for rxn in rxns:
        for specie in rxn.get_species():
            if specie not in all_species_set:
                all_species.append(specie)
                all_species_set.add(specie)
    return tuple(all_species)


def gillespy2_crn_counts(
        rxns: Iterable[Reaction],
        initial_counts: dict[Specie, int],
        t_eval: Iterable[float],
        *,
        dependent_symbols: dict[sympy.Symbol | str, sympy.Expr | str] | None = None,
        seed: int | None = None,
        solver_class: type = gp.NumPySSASolver,
        **options,
) -> gp.Results:
    r"""
    Run the reactions using the GillesPy2 package for discrete simulation using the Gillespie algorithm.

    Any parameters not described here are passed along to the function [gillespy2.GillesPySolver.run](
    https://gillespy2.readthedocs.io/en/latest/classes/gillespy2.core.html#gillespy2.core.gillespySolver.GillesPySolver.run)


    Parameters
    ----------
    rxns:
        list of [`Reaction`](gpac.crn.Reaction)'s comprising the chemical reaction network.
        See documentation for [`Reaction`](gpac.crn.Reaction) for details on how to specify reactions.

    initial_counts:
        dict mapping each species to its initial integer count.
        Note that unlike the parameter `initial_values` in [`integrate_odes`](gpac.ode.integrate_odes),
        keys in this dict must be [`Specie`](gpac.crn.Specie) objects, not strings or sympy symbols.

    Returns
    -------
    :
        Same Result object returned by [gillespy2.GillesPySolver.run](
        https://gillespy2.readthedocs.io/en/latest/classes/gillespy2.core.html#gillespy2.core.gillespySolver.GillesPySolver.run)
    """
    if 'solver' in options:
        raise ValueError('solver should not be passed in options; instead, pass the solver_class parameter')
    model = gp.Model()

    all_species = find_all_species(rxns)
    for specie in all_species:
        val = initial_counts.get(specie, 0)
        model.add_species(gp.Species(name=specie.name, initial_value=val))
        if specie.name == 'time':
            raise ValueError('species cannot be named "time"')

    for rxn in rxns:
        rxn_name = (''.join([rct.name for rct in rxn.reactants.species]) + 'to'
                    + ''.join([prd.name for prd in rxn.products.species]))
        rate_f = gp.Parameter(name=f'{rxn_name}_k', expression=f'{rxn.rate_constant}')
        model.add_parameter(rate_f)
        reactant_counts = rxn.reactants.species_counts('str')
        product_counts = rxn.products.species_counts('str')
        gp_rxn = gp.Reaction(name=rxn_name, reactants=reactant_counts, products=product_counts,
                             rate=rate_f)  # type: ignore
        model.add_reaction(gp_rxn)
        if rxn.reversible:
            rxn_name_r = rxn_name + '_r'
            rate_r = gp.Parameter(name=f'{rxn_name_r}_f', expression=f'{rxn.rate_constant_reverse}')
            model.add_parameter(rate_r)
            gp_rxn_r = gp.Reaction(name=rxn_name_r, reactants=product_counts, products=reactant_counts,
                                   rate=rate_r)  # type: ignore
            model.add_reaction(gp_rxn_r)

    tspan = gp.TimeSpan(t_eval)
    model.timespan(tspan)
    solver: gp.GillesPySolver = solver_class(model=model)
    if seed is None:
        gp_results = model.run(solver=solver, **options)
    else:
        gp_results = model.run(solver=solver, seed=seed, **options)

    if dependent_symbols is not None:
        independent_symbols = [sympy.Symbol(specie.name) for specie in all_species]
        dependent_funcs = {symbol: sympy.lambdify(independent_symbols, func)
                           for symbol, func in dependent_symbols.items()}

        indp_vals = []
        for specie in all_species:
            indp_vals.append(gp_results[0][specie.name])

        for dependent_symbol, func in dependent_funcs.items():
            # convert 2D numpy array to list of 1D arrays so we can use Python's * operator to distribute
            # the vectors as separate arguments to the function func
            dep_vals_row = func(*indp_vals)
            dependent_symbol_name = dependent_symbol.name if isinstance(dependent_symbol,
                                                                        sympy.Symbol) else dependent_symbol
            gp_results[0][dependent_symbol_name] = dep_vals_row

    return gp_results


def _run_rebop_with_resets(
        resets: dict[float, dict[str, int]],
        crn: rb.Gillespie,
        inits: dict[str, int],
        tmax: float,
        nb_steps: int = 0,
        seed: int | None = None,
) -> xarray.Dataset:
    if len(resets) == 0:
        raise ValueError("resets dictionary must not be empty")

    for reset_time, reset in resets.items():
        if len(reset) == 0:
            raise ValueError(f"Each reset dict must be nonempty, "
                             f"but reset time {reset_time} has an empty dict.")

    for reset_time in resets.keys():
        if reset_time <= 0 or reset_time >= tmax:
            raise ValueError(f"Reset time {reset_time} is outside the simulated time interval [0, {tmax}]")

    reset_times = sorted(resets.keys())

    # Break the interval (0,tmax) into segments based on reset times
    segments_and_resets = []
    for i in range(len(reset_times) + 1):
        reset = resets[reset_times[i - 1]] if i > 0 else inits
        if i == 0:
            segments_and_resets.append(((0, reset_times[0]), reset))
        elif i == len(reset_times):
            segments_and_resets.append(((reset_times[-1], tmax), reset))
        else:
            segments_and_resets.append(((reset_times[i - 1], reset_times[i]), reset))

    total_results = None
    for (t_start, t_end), reset in segments_and_resets:
        tmax = t_end - t_start
        latest_results = crn.run(init=reset, tmax=tmax, nb_steps=nb_steps, seed=seed)
        if total_results is None:
            total_results = latest_results
        else:
            time_offset = total_results.time.values[-1]
            latest_results_adjusted = latest_results.assign_coords(time=latest_results.time + time_offset)
            total_results_trimmed = total_results.isel(time=slice(0, -1))
            print(f"total_results before concat: {total_results}")
            print(f"latest_results before concat: {total_results}")
            total_results = xr.concat([total_results_trimmed, latest_results_adjusted], dim="time")
            print(f"total_results after concat: {total_results}")

    return total_results


def rebop_crn_counts(
        rxns: Iterable[Reaction],
        initial_counts: dict[Specie, int],
        tmax: float,
        *,
        nb_steps: int = 0,
        vol: float | None = None,
        resets: dict[float, dict[sympy.Symbol | str, int]] | None = None,
        dependent_symbols: dict[sympy.Symbol | str, sympy.Expr | str] | None = None,
        seed: int | None = None,
) -> xarray.Dataset:
    r"""
    Run the reactions using the [rebop package](https://pypi.org/project/rebop/)
    for discrete simulation using the Gillespie algorithm.


    Parameters
    ----------
    rxns:
        list of [`Reaction`](gpac.crn.Reaction)'s comprising the chemical reaction network.
        See documentation for [`Reaction`](gpac.crn.Reaction) for details on how to specify reactions.

    initial_counts:
        dict mapping each species to its initial integer count.
        Note that unlike the parameter `initial_values` in [`integrate_odes`](gpac.ode.integrate_odes),
        keys in this dict must be [`Specie`](gpac.crn.Specie) objects, not strings or sympy symbols.

    nb_steps:
        Number of evenly-spaced time points at which to record the counts between 0 and `tmax`.
        If not specified, all reaction events and their exact times are recorded,
        instead of fixed, evenly-spaced time points.

    tmax:
        the maximum time for the simulation.

    vol:
        the volume of the system. If not specified, the volume is assumed to be the sum of the initial counts.
        reactions with k total reactants have their rate divided by vol^(k-1) to account for the volume.

    resets:
        If specified, this is a dict mapping times to "configurations" (i.e., dict mapping symbols/str to values).
        The configurations are used to set the values of the symbols manually during the ODE integration
        at specific times.
        Any symbols not appearing as keys in `resets` are left at their current values.
        The OdeResult returned (the one returned by `solve_ivp` in scipy) will have two additional fields:
        `reset_times` and `reset_indices`, which are lists of the times and indices in `sol.t`
        corresponding to the times when the resets were applied.
        Raises a ValueError if any time lies outside the integration interval, or if `resets` is empty.

    Returns
    -------
    :
        Same Result object returned by rebop.Gillespie.run. (an xarray.Dataset object)
        It can be indexed by species name to get the counts,
        and by the key `"time"` to get the times at which the counts were recorded.
    """
    if vol is None:
        vol = sum(initial_counts.values())

    crn = rb.Gillespie()
    for rxn in rxns:
        reactants = [specie.name for specie in rxn.reactants.species]
        products = [specie.name for specie in rxn.products.species]
        rate = rxn.rate_constant / vol ** (len(reactants) - 1)
        crn.add_reaction(rate, reactants, products)
        if rxn.reversible:
            rate_rev = rxn.rate_constant_reverse / vol ** (len(products) - 1)
            crn.add_reaction(rate_rev, products, reactants)

    initial_counts_str = {specie.name: count for specie, count in initial_counts.items()}
    if resets is None:
        rb_results = crn.run(init=initial_counts_str, tmax=tmax, nb_steps=nb_steps, seed=seed)
    else:
        # normalize resets to have strings as keys
        resets_normalized: dict[float, dict[str, int]] = {
            time: {str(symbol): count for symbol, count in counts.items()}
            for time, counts in resets.items()
        }
        rb_results = _run_rebop_with_resets(resets=resets_normalized, crn=crn, inits=initial_counts_str, tmax=tmax,
                                            nb_steps=nb_steps, seed=seed)

    all_species = find_all_species(rxns)
    if dependent_symbols is not None:
        independent_symbols = [sympy.Symbol(specie.name) for specie in all_species]
        dependent_funcs = {symbol: sympy.lambdify(independent_symbols, func)
                           for symbol, func in dependent_symbols.items()}

        indp_vals = []
        for specie in all_species:
            indp_vals.append(rb_results[specie.name])

        for dependent_symbol, func in dependent_funcs.items():
            # convert 2D numpy array to list of 1D arrays so we can use Python's * operator to distribute
            # the vectors as separate arguments to the function func
            dep_vals_row = func(*indp_vals)
            dependent_symbol_name = dependent_symbol.name if isinstance(dependent_symbol,
                                                                        sympy.Symbol) else dependent_symbol
            rb_results[dependent_symbol_name] = dep_vals_row

    return rb_results


def plot_gillespie(
        rxns: Iterable[Reaction],
        initial_counts: dict[Specie, int],
        tmax: float,
        *,
        nb_steps: int = 0,
        seed: int | None = None,
        resets: dict[float, dict[sympy.Symbol | str, int]] | None = None,
        dependent_symbols: dict[sympy.Symbol | str, sympy.Expr | str] | None = None,
        figure_size: tuple[float, float] = (10, 3),
        latex_legend: bool = False,
        symbols_to_plot: Iterable[sympy.Symbol | str] |
                         Iterable[Iterable[sympy.Symbol | str]] |
                         str |
                         re.Pattern |
                         Iterable[re.Pattern] |
                         None = None,
        legend: dict[sympy.Symbol | str, str] | None = None,
        show: bool = False,
        return_simulation_result: bool = False,
        loc: str | tuple[float, float] = 'best',
        warn_change_dpi: bool = False,
        vol: float | None = None,
        simulation_package: Literal['rebop', 'gillespy2'] = 'rebop',
        **options: dict[str, object],
) -> xarray.Dataset:
    r"""
    Similar to [`plot_crn`](gpac.crn.plot_crn), but uses the [rebop package](https://pypi.org/project/rebop/)
    for discrete simulation using the Gillespie algorithm instead of continuous ODEs.

    Undocumented arguments have the same meaning as with [`plot_crn`](gpac.crn.plot_crn).

    Arguments `tmax`, `nb_steps`, and `seed` are passed to [`rebop_crn_counts`](gpac.crn.rebop_crn_counts).
    Any custom keyword arguments (specified in `**options`) to the function matplotlib.pyplot.plot.

    Parameters
    ----------
    rxns: the reactions of the CRN

    initial_counts: initial (integer) counts of each species

    vol: volume of the system (reactions with k ractants have their rate divided by vol^(k-1))

    tmax: the maximum time for which to run the simulation

    nb_steps: number of evenly-spaced time points at which to record the counts between 0 and `tmax`;
        if not specified (or set to 0, the default value), all reaction events and their exact times are recorded

    return_simulation_result: whether to return the simulation result; if True, the result of the simulation
        is returned, but the default behavior is not to do this, so that if the last line of a notebook cell
        is a call to `plot_gillespie`, the result is not printed to the output, only the plot is shown.

    seed: seed for random number generator used by rebop for stochastic simulation. Note that currently,
        the value `nb_steps` actually changes the stochastic sampling in the simulation, i.e., if you double the
        value of `nb_steps`, but keep the value of `seed` the same, you would think that every other sampled
        configuration would be the same as when `nb_steps` was half as large, but this is not the case.
        See [rebop issue #26](https://github.com/Armavica/rebop/issues/26).

    resets:
        If specified, this is a dict mapping times to "configurations" (i.e., dict mapping symbols/str to values).
        The configurations are used to set the values of the symbols manually during the ODE integration
        at specific times.
        Any symbols not appearing as keys in `resets` are left at their current values.
        The OdeResult returned (the one returned by `solve_ivp` in scipy) will have two additional fields:
        `reset_times` and `reset_indices`, which are lists of the times and indices in `sol.t`
        corresponding to the times when the resets were applied.
        Raises a ValueError if any time lies outside the integration interval, or if `resets` is empty.

    Returns
    -------
    :
        The result of the simulation, which is the same as the result of
        [`rebop_crn_counts`](gpac.crn.rebop_crn_counts).
    """
    if simulation_package == 'rebop':
        rb_result = rebop_crn_counts(
            rxns=rxns,
            initial_counts=initial_counts,
            tmax=tmax,
            nb_steps=nb_steps,
            seed=seed,
            vol=vol,
            resets=resets,
            dependent_symbols=dependent_symbols,
        )
        times = rb_result['time']
        result = {str(name): rb_result[name] for name in rb_result if name != 'time'}
    elif simulation_package == 'gillespy2':
        raise NotImplementedError('gillespy2 is not yet supported')
        # gp_result = gillespie_crn_counts(
        #     rxns=rxns,
        #     initial_counts=initial_counts,
        #     t_eval=t_eval,
        #     seed=seed,
        #     dependent_symbols=dependent_symbols,
        #     **options,
        # )
        # symbols = tuple(name for name in gp_result[0].keys() if name != 'time')
        # assert len(symbols) == len(gp_result[0]) - 1  # -1 for 'time'
        # times = np.array(t_eval)
        # # convert gp_result to Dict[str, np.ndarray] for _plot_given_values
        # result = {symbol: gp_result[0][symbol] for symbol in symbols}
    else:
        raise ValueError(f'Unknown simulation_package {simulation_package}')

    plot_given_values(
        times=times,
        result=result,
        source='ssa',
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
    return rb_result if return_simulation_result else None


SpeciePair: TypeAlias = tuple['Specie', 'Specie']  # forward annotations don't seem to work here
Output: TypeAlias = SpeciePair | dict[SpeciePair, float]


def replace_reversible_rxns(rxns: Iterable[Reaction]) -> list[Reaction]:
    r"""
    Parameters
    ----------
    rxns:
        list of [`Reaction`](gpac.crn.Reaction)'s

    Returns
    -------
    :
        list of [`Reaction`](gpac.crn.Reaction)'s, where every reversible reaction in `rxns` has been replaced by
        two irreversible reactions, and all others have been left as they are
    """
    new_rxns: list[Reaction] = []
    for rxn in rxns:
        if not rxn.reversible:
            new_rxn = copy.deepcopy(rxn)
            new_rxns.append(new_rxn)
        else:
            forward_rxn = Reaction(reactants=rxn.reactants, products=rxn.products,
                                   k=rxn.rate_constant, reversible=False)
            reverse_rxn = Reaction(reactants=rxn.products, products=rxn.reactants,
                                   k=rxn.rate_constant_reverse, reversible=False)
            new_rxns.extend([forward_rxn, reverse_rxn])
    return new_rxns


@dataclass(frozen=True)
class Specie:
    """
    Represents species in a chemical reaction network. In general these are not created directly,
    but rather via the [`species`](gpac.crn.species) function, 
    which creates a tuple of [`Specie`](gpac.crn.Specie) objects.
    """

    name: str
    """
    Name of the species. This is used in two ways: when plotting, this name is used 
    in the legend, and when using the `dependent_symbols` parameter in the functions
    [`plot_crn`](gpac.crn.plot_crn) and [`plot_gillespie`](gpac.crn.plot_gillespie), 
    this name is used to identify the species, since dependent symbols need to be specified
    as sympy.Symbol objects defined as functions of other sympy.Symbol objects.
    The way to connect that to the species is to make two objects, one Specie and one Symbol,
    with the same name. See the 
    [example notebook](https://github.com/UC-Davis-molecular-computing/gpac/blob/main/notebook.ipynb)
    for an example of how this is done.
    """

    def __add__(self, other: Specie | Expression) -> Expression:
        if isinstance(other, Expression):
            return other + Expression([self])
        elif isinstance(other, Specie):
            return Expression([self]) + Expression([other])

        raise NotImplementedError()

    __radd__ = __add__

    def __rshift__(self, other: Specie | Expression) -> Reaction:
        return Reaction(self, other)

    def __rrshift__(self, other: Specie | Expression) -> Reaction:
        return Reaction(other, self)

    def __or__(self, other: Specie | Expression) -> Reaction:
        return Reaction(self, other, reversible=True)

    def __mul__(self, other: int) -> Expression:
        if isinstance(other, int):
            return other * Expression([self])
        else:
            raise NotImplementedError()

    def __rmul__(self, other: int) -> Expression:
        if isinstance(other, int):
            return other * Expression([self])
        else:
            raise NotImplementedError()

    def __str__(self) -> str:
        return self.name

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Specie):
            return NotImplemented
        return self.name < other.name

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Specie):
            return NotImplemented
        return self.name == other.name

    __req__ = __eq__


@dataclass(frozen=True)
class Expression:
    """
    Class used for very basic symbolic manipulation of left/right hand
    side of stoichiometric equations. Not very user friendly; users should
    just use the [`species`](gpac.crn.species) function and manipulate [`Specie`](gpac.crn.Specie) objects
    with operators `>>`, `|`, `+`, and `*` to create reactions
    (see [`Reaction`](gpac.crn.Reaction) for examples).
    """

    species: list[Specie]
    """
    ordered list of species in expression, e.g, A+A+B is [A,A,B]
    """

    def __getitem__(self, idx: int) -> Specie:
        """
        Parameters
        ----------
        idx:
            index of species to return

        Returns
        -------
        :
            [`Specie`](gpac.crn.Specie) at index `idx` in this [`Expression`](gpac.crn.Expression)
        """
        return self.species[idx]

    def __add__(self, other: Expression | Specie) -> Expression:
        """
        Parameters
        ----------
        other:
            [`Expression`](gpac.crn.Expression) or [`Specie`](gpac.crn.Specie) to add to this one

        Returns
        -------
        :
            [`Expression`](gpac.crn.Expression) representing the union of this [`Expression`](gpac.crn.Expression) and `other`
        """
        if isinstance(other, Expression):
            species_copy = list(self.species)
            species_copy.extend(other.species)
            return Expression(species_copy)
        elif isinstance(other, Specie):
            species_copy = list(self.species)
            species_copy.append(other)
            return Expression(species_copy)
        else:
            raise NotImplementedError()

    def __rmul__(self, coeff: int) -> Expression:
        """
        Parameters
        ----------
        coeff:
            coefficient to multiply this [`Expression`](gpac.crn.Expression) by

        Returns
        -------
        :
            [`Expression`](gpac.crn.Expression) representing this [`Expression`](gpac.crn.Expression)
            multiplied by `coeff`
        """
        if isinstance(coeff, int):
            species_copy = []
            for _ in range(coeff):
                species_copy.extend(self.species)
            return Expression(species_copy)
        else:
            raise NotImplementedError()

    __mul__ = __rmul__

    def __rshift__(self, expr: Specie | Expression) -> Reaction:
        return Reaction(self, expr)

    def __or__(self, other: Specie | Expression) -> Reaction:
        return Reaction(self, other, reversible=True)

    def __str__(self) -> str:
        if len(self.species) == 0:
            return '∅'
        return '+'.join(s.name for s in self.species)

    def __len__(self) -> int:
        return len(self.species)

    def get_species(self) -> set[Specie]:
        """
        Returns the set of species in this expression, not their
        coefficients.
        """
        return set(self.species)

    def species_counts(self, key_type: Literal['str', 'Specie'] = 'Specie') -> dict[Specie, int]:
        """
        Returns a dictionary mapping each species in this expression to its
        coefficient.
        """
        species_counts = {}
        for specie in self.species:
            key = specie.name if key_type == 'str' else specie
            species_counts[key] = species_counts.get(key, 0) + 1
        return species_counts


empty = Expression([])
r"""
Used for chemical reactions with empty reactant or product lists, e.g., to implement the exponential
decay reaction $X \to \emptyset$:
```py
x = species('X')
rxn = x >> empty
```
"""

avogadro = 6.02214076e23


def concentration_to_count(concentration: float, volume: float) -> int:
    """
    Parameters
    ----------
    concentration:
        units of M (molar) = moles / liter

    volume:
        units of liter

    Returns
    -------
    :
        count of molecule with `concentration` in `volume`
    """
    return round(avogadro * concentration * volume)


@dataclass
class Reaction:
    r"""
    Representation of a stoichiometric reaction using a pair of [`Expression`](gpac.crn.Expression)'s,
    one for the reactants and one for the products.

    Reactions are constructed by creating objects of type [`Specie`](gpac.crn.Specie) and using the operators
    `>>` (for irreversible reactions) and `|` (for reversible reactions), as well as the `+` and
    `*` operators to specify the stoichiometric coefficients of the reactants and products,
    and optionally the methods [`Reaction.k`](gpac.crn.Reaction.k) and [`Reaction.r`](gpac.crn.Reaction.r) 
    to specify forward and reverse rate constants.

    For example, the following code creates a reaction that represents the irreversible reaction
    $A + B \rightarrow C$ (with implicit rate constant 1.0):

    ```py
    a,b,c = species('A B C')
    rxn = a+b >> c
    ```

    To create reactions

    $$
    \begin{align*}
        A+B &\underset{4.1}{\stackrel{0.6}{\rightleftharpoons}} 2C
    \\
        C   &\xrightarrow{5.2} D
    \end{align*}
    $$

    use the following code:

    ```py
    a,b,c,d = gpac.species('A B C D')
    rxns = [
        (a+b | 2*c).k(0.6).r(4.1),
        (c >> d).k(5.2),
    ]
    ```

    Also supported are inhibitors, which can be added to reactions using the method [`Reaction.i`](gpac.crn.Reaction.i):

    ```py
    a, b, u, i = species('A B U I')
    rxn = (a+b | 2*u).i(i, 100)
    ```

    which represents the reaction $A+B \to 2U$ with inhibitor $I$ and inhibitor constant 100.
    Currently the inhibitor is modeled using a first-order Hill function, i.e., its contribution to the
    reaction rate is to divide by $1 + i \cdot I$, where $i$ is the inhibitor constant.
    So for the reaction defined above, its rate is $[A] \cdot [B] / (1 + 100 \cdot [I])$.
    """

    reactants: Expression
    """The left side of species in the reaction."""

    products: Expression
    """The right side of species in the reaction."""

    rate_constant: float = 1.0
    """Rate constant of forward reaction."""

    rate_constant_reverse: float = 1.0
    """Rate constant of reverse reaction 
    (only used if [`Reaction.reversible`](gpac.crn.Reaction.reversible) is true)."""

    reversible: bool = False
    """Whether reaction is reversible, i.e. `products` :math:`\\to` `reactants` is a reaction also."""

    inhibitors: list[Specie] = field(default_factory=list)
    """Inhibitors of the reaction."""

    inhibitor_constants: list[float] = field(default_factory=list)

    def __init__(self, reactants: Specie | Expression, products: Specie | Expression,
                 k: float = 1, r: float = 1,
                 reversible: bool = False) -> None:
        """
        In general this constructor should not be used directly; instead, use the operators ``>>``,
        ``|``, ``+``, and ``*`` to construct reactions. (See description of [`Reaction`](gpac.crn.Reaction) for
        examples.)


        Parameters
        ----------
        reactants:
            left side of species in the reaction
        products:
            right side of species in the reaction
        k:
            Rate constant of forward reaction
        r:
            Rate constant of reverse reaction (only used if [`Reaction.reversible`](gpac.crn.Reaction.reversible) is true
        reversible:
            Whether reaction is reversible
        """
        if not (isinstance(reactants, Specie) or isinstance(reactants, Expression)):
            raise ValueError(
                "Attempted construction of reaction with type of reactants "
                f"as {type(reactants)}. Type of reactants must be Species "
                "or Expression")
        if not (isinstance(products, Specie) or isinstance(products, Expression)):
            raise ValueError(
                "Attempted construction of products with type of products "
                f"as {type(products)}. Type of products must be Species "
                "or Expression")

        if isinstance(reactants, Specie):
            reactants = Expression([reactants])
        if isinstance(products, Specie):
            products = Expression([products])
        self.reactants = reactants
        self.products = products
        self.rate_constant = k
        self.rate_constant_reverse = r
        self.reversible = reversible
        self.inhibitors = []
        self.inhibitor_constants = []

    def with_inhibitor(self, inhibitor: Specie, constant: float = 1.0) -> Reaction:
        """
        Parameters
        ----------
        inhibitor:
            The inhibitor species
        constant:
            The inhibitor constant
        """
        self.inhibitors.append(inhibitor)
        self.inhibitor_constants.append(constant)
        return self

    def i(self, inhibitor: Specie, constant: float = 1.0) -> Reaction:
        """
        alias for [`Reaction.with_inhibitor`](gpac.crn.Reaction.with_inhibitor)
        """
        return self.with_inhibitor(inhibitor, constant)

    def get_ode(self, specie: Specie, reverse: bool = False) -> sympy.Expr:
        r"""
        Parameters
        ----------
        specie:
            A [`Specie`](gpac.crn.Specie) that may or may not appear in this [`Reaction`](gpac.crn.Reaction).

        reverse:
            Whether to interpret this reaction in reverse, i.e., treat products as reactants
            and vice versa. Raises exception if the reaction is not reversible.

        Returns
        -------
        :
            sympy expression for the ODE term for the given [`Specie`](gpac.crn.Specie).
            For example, if the reaction is $A+B \to 2C$,
            then the ODE for $A$ is $-k \cdot A \cdot B$,
            the ODE for B is $-k \cdot A \cdot B$,
            and the ODE for C is $2 \cdot k \cdot A \cdot B$.
        """
        if reverse and not self.reversible:
            raise ValueError(f'reaction {self} is not reversible, so `reverse` parameter must be False')

        if specie not in self.get_species():
            return sympy.Integer(0)

        reactants = self.reactants
        products = self.products
        inhibitors = self.inhibitors
        rate_constant = self.rate_constant
        if reverse:
            reactants = self.products
            products = self.reactants
            rate_constant = self.rate_constant_reverse

        reactant_coeff = reactants.species.count(specie)
        product_coeff = products.species.count(specie)
        net_produced = product_coeff - reactant_coeff
        reactants_ode = sympy.Integer(1)
        for reactant in reactants.get_species():
            reactant_term = sympy.Symbol(reactant.name) ** reactants.species.count(reactant)
            reactants_ode *= reactant_term

        inhibitors_ode = sympy.Integer(1)
        for inhibitor, inhibitor_constant in zip(inhibitors, self.inhibitor_constants):
            inh = sympy.Symbol(inhibitor.name)

            den = 1 + inhibitor_constant * inh if inhibitor_constant != 1.0 else 1 + inh
            inhibitor_term = 1 / den

            # inhibitor_term = inhibitor_constant * sympy.exp(-inh) \
            #     if inhibitor_constant != 1.0 else \
            #     sympy.exp(-inh)

            # # if inhibitor close to 0, take down to 0
            # term = sympy.Piecewise(
            #     (inh, inh > 0.000000000001),
            #     (0, True),
            # )
            # den = 1 + inhibitor_constant * term
            # inhibitor_term = 1 / den

            # inhibitor_term = sympy.Piecewise(
            #     (0, inh > 10**(-3)),
            #     (1, True),
            # )

            inhibitors_ode *= inhibitor_term

        # if rate constant is 1.0, avoid the ugly "1.0*" factor in the output
        if len(inhibitors) == 0:
            ode = net_produced * reactants_ode if rate_constant == 1.0 \
                else net_produced * rate_constant * reactants_ode
        else:
            ode = net_produced * reactants_ode * inhibitors_ode if rate_constant == 1.0 \
                else net_produced * rate_constant * reactants_ode * inhibitors_ode

        return ode

    def is_unimolecular(self) -> bool:
        """
        Returns
        -------
        :
            true if there is one reactant
        """
        return self.num_reactants() == 1

    def is_bimolecular(self) -> bool:
        """
        Returns
        -------
        :
            true if there are two reactants
        """
        return self.num_reactants() == 2

    def symmetric(self) -> bool:
        """
        Returns
        -------
        :
            true if there are two reactants that are the same species
        """
        return self.num_reactants() == 2 and self.reactants.species[0] == self.reactants.species[1]

    def symmetric_products(self) -> bool:
        """
        Returns
        -------
        :
            true if there are two products that are the same species
        """
        return self.num_products() == 2 and self.products.species[0] == self.products.species[1]

    def num_reactants(self) -> int:
        """
        Returns
        -------
        :
            number of reactants
        """
        return len(self.reactants)

    def num_products(self) -> int:
        """
        Returns
        -------
        :
            number of products
        """
        return len(self.products)

    def num_inhibitors(self) -> int:
        """
        Returns
        -------
        :
            number of inhibitors
        """
        return len(self.inhibitors)

    def is_conservative(self) -> bool:
        """
        Returns
        -------
        :
            true if number of reactants equals number of products
        """
        return self.num_reactants() == self.num_products()

    def reactant_if_unimolecular(self) -> Specie:
        """
        Returns
        -------
        :
            unique reactant if there is only one

        Raises
        ------
        ValueError
            if there are multiple reactants
        """
        if self.is_unimolecular():
            return self.reactants.species[0]
        else:
            raise ValueError(f'reaction {self} is not unimolecular')

    def product_if_unique(self) -> Specie:
        """
        Returns
        -------
        :
            unique product if there is only one

        Raises
        ------
        ValueError
            if there are multiple products
        """
        if self.num_products() == 1:
            return self.products.species[0]
        else:
            raise ValueError(f'reaction {self} does not have exactly one product')

    def reactants_if_bimolecular(self) -> tuple[Specie, Specie]:
        """
        Returns
        -------
        :
            pair of reactants if there are exactly two

        Raises
        ------
        ValueError
            if there are not exactly two reactants
        """
        if self.is_bimolecular():
            return self.reactants.species[0], self.reactants.species[1]
        else:
            raise ValueError(f'reaction {self} is not bimolecular')

    def reactant_names_if_bimolecular(self) -> tuple[str, str]:
        """
        Returns
        -------
        :
            pair of reactant names if there are exactly two

        Raises
        ------
        ValueError
            if there are not exactly two reactants
        """
        r1, r2 = self.reactants_if_bimolecular()
        return r1.name, r2.name

    def products_if_exactly_two(self) -> tuple[Specie, Specie]:
        """
        Returns
        -------
        :
            pair of products if there are exactly two

        Raises
        ------
        ValueError
            if there are not exactly two products
        """
        if self.num_products() == 2:
            return self.products.species[0], self.products.species[1]
        else:
            raise ValueError(f'reaction {self} does not have exactly two products')

    def product_names_if_exactly_two(self) -> tuple[str, str]:
        """
        Returns
        -------
        :
            pair of product names if there are exactly two

        Raises
        ------
        ValueError
            if there are not exactly two products
        """
        p1, p2 = self.products_if_exactly_two()
        return p1.name, p2.name

    def __str__(self) -> str:
        for_rate_str = '' if self.rate_constant == 1 else f'({self.rate_constant})'
        if not self.reversible:
            rev_rate_str = ''
        elif self.rate_constant_reverse == 1:
            rev_rate_str = '<'
        else:
            rev_rate_str = f'({self.rate_constant_reverse})<'
        if len(self.inhibitors) > 0:
            def constant_str(constant: float) -> str:
                return '' if constant == 1.0 else f'[{constant}]'

            inhibitor_str = '--' + ','.join(f'{inhibitor.name}{constant_str(constant)}'
                                            for inhibitor, constant in zip(self.inhibitors, self.inhibitor_constants))
        else:
            inhibitor_str = ''
        return f"{self.reactants} {rev_rate_str}{inhibitor_str}-->{for_rate_str} {self.products}"

    def __repr__(self) -> str:
        return (f"Reaction({repr(self.reactants)}, {repr(self.products)}, "
                f"rate_constant={self.rate_constant}, "
                f"rate_constant_reverse={self.rate_constant_reverse}, "
                f"reversible={self.reversible})")

    def k(self, coeff: float) -> Reaction:
        """
        Same as [`Reaction.f`](gpac.crn.Reaction.f).

        Parameters
        ----------
        coeff: float
            The new reaction coefficient
        """
        self.rate_constant = coeff
        return self

    def f(self, coeff: float) -> Reaction:
        """
        Changes the reaction coefficient to `coeff` and returns `self`.

        This is useful for including the rate constant during the construction
        of a reaction. For example

        ```py
        x, y, z = species("X Y Z")
        rxns = [
            (x + y >> z).f(2.5),
            (z >> x).f(1.5),
            (z >> y).f(0.5)),
        ]
        ```

        Note that if this is a reversible reaction, this specifies the *forward* rate constant.

        Parameters
        ----------
        coeff: float
            The new (forward) reaction coefficient
        """
        self.rate_constant = coeff
        return self

    def r(self, coeff: float) -> Reaction:
        r"""
        Changes the reverse reaction reaction rate constant to `coeff` and returns `self`.

        This is useful for including the rate constant during the construction
        of a reaction. For example, the following defines a reversible reaction
        $X + Y \rightleftharpoons Z$ with forward rate constant 2.5 and reverse rate constant 1.5.

        ```py
        x, y, z = species("X Y Z")
        rxn = (x + y | z).k(2.5).r(1.5)
        ```

        Parameters
        ----------
        coeff: float
            The new reverse reaction rate constant
        """
        if not self.reversible:
            raise ValueError('cannot set r on an irreversible reaction')
        self.rate_constant_reverse = coeff
        return self

    def get_species(self) -> tuple[Specie, ...]:
        """
        Returns
        -------
        :
            a tuple with the species present in the reactants, products, and inhibitors, in that order.
        """
        all_species = []
        all_species_set = set()
        for s in self.reactants.species + self.products.species + self.inhibitors:
            if s not in all_species_set:
                all_species.append(s)
                all_species_set.add(s)
        return tuple(all_species)


if __name__ == '__main__':
    main()
