"""
Module for expressing chemical reaction networks and deriving their ODEs. Ideas and much code taken from
https://github.com/enricozb/python-crn.

For example, to specify the "approximate majority" chemical reaction network
(see https://doi.org/10.1007/978-3-540-75142-7_5 or https://doi.org/10.1126/science.aal2052)

.. math::

    A+B \\to 2U

    A+U \\to 2A

    B+U \\to 2B

we can write

.. code-block:: python

    a, b, u = species('A B U')
    rxns = [
        a+b >> 2*u,
        a+u >> 2*a,
        b+u >> 2*b,
    ]
    initial_values = {a: 0.51, b: 0.49}
    t_eval = np.linspace(0, 10, 100)
    gpac.plot_crn(rxns, initial_values, t_eval)

which will plot the concentrations of A, B, and U over time. One can specify reversible reactions
by using the ``|`` operator instead of ``>>`` (e.g., ``a+b | 2*u``) and rate constants using the functions
``k`` (for forward rate constants) and ``r`` (for reverse rate constants),
e.g., ``(a+b | 2*u).k(1.5).r(0.5)``.

See functions :func:`crn_to_odes` to convert reactions to ODEs (ordinary differential equations),
:func:`integrate_crn_odes` to get the trajectories of integrating these ODEs over time, and
:func:`plot_crn` to plot the trajectories. The documentation for :func:`crn_to_odes` explains
how reactions are converted into ODEs by each of these functions.

Also supported are inhibitors, which can be added to reactions using the method :meth:`Reaction.i`:

.. code-block:: python

    a, b, u, i = species('A B U I')
    rxn = (a+b | 2*u).i(i, 100)

which represents the reaction :math:`A+B \\to 2U` with inhibitor :math:`I` and inhibitor constant 100.
Currently the inhibitor is modeled using a first-order Hill function, i.e., its contribution to the
reaction rate is to divide by :math:`1 + k \\cdot I`, where :math:`k` is the inhibitor constant.
So for the reaction defined above, its rate is :math:`k \\cdot [A] \\cdot [B] / (1 + 100 \\cdot [I])`.
"""

from __future__ import annotations  # needed for forward references in type hints

from typing import Dict, Iterable, Tuple, Set, Union, Optional, Callable, List, Literal
from collections import defaultdict
import copy
from dataclasses import dataclass, field

from scipy.integrate import OdeSolver
from scipy.integrate._ivp.ivp import OdeResult  # noqa
import sympy
import gillespy2 as gp
import numpy as np

from gpac import integrate_odes, plot, plot_given_values


def crn_to_odes(rxns: Iterable[Reaction]) -> Dict[sympy.Symbol, sympy.Expr]:
    """
    Given a set of chemical reactions, return the corresponding ODEs.

    Each reaction contributes one term to the ODEs for each species produced or consumed in it.
    The term from a reaction appearing in the ODE for species `X` is the product of:

    - the rate constant,
    - the reactant concentrations, and
    - the net stoichiometry of species `X` in the reaction
      (i.e., the net amount of `X` produced by the reaction, negative if consumed).

    For example, consider the following two reactions with respective rate constants
    :math:`k_1` and :math:`k_2`:

    .. math::

        X+X &\\xrightarrow{k_1} C

        C+X &\\xrightarrow{k_2} C+Y

    The net stoichiometry of `X` in the first reaction is -2, since two copies of `X` are consumed,
    and the net stoichiometry of `C` in that reaction is 1, since one copy of `C` is produced.
    The net stoichiometry of `C` in the *second* reaction is 0, since it is a catalyst
    (neither produced nor consumed).

    This corresponds to ODEs (following the convention of lowercase letter `x`
    for the concentration of species `X`):

    .. math::

        x' &= -2 k_1 x^2 - k_2 c x

        c' &= k_1 x^2

        y' &= k_2 c x

    In the package, this can be implemented (for example setting :math:`k_1 = 1.5` and :math:`k_2 = 0.2`)
    via:

    .. code-block:: python

        x, y, c = species('X Y C')
        rxns = [
            (x+x >> c).k(1.5),
            (c+x >> c+y).k(0.2),
        ]
        odes = crn_to_odes(rxns)
        for symbol, ode in odes.items():
            print(f"{symbol}' = {ode}")

    which prints

    .. code-block:: none

        X' = -0.2*C*X - 3.0*X**2
        C' = 1.5*X**2
        Y' = 0.2*C*X

    Args:
        rxns: list of :any:`Reaction`'s comprising the chemical reaction network.
              See documentation for :any:`Reaction` for details on how to specify reactions.

    Returns:
        Dictionary mapping each species (represented as a sympy Symbol object, rather than a :any:`Specie`
        object) to its corresponding ODE (represented as a sympy Expression).
        This object can be given as the parameter `odes` to the functions :func:`ode.integrate_odes`
        and :func:`ode.plot` to integrate/plot the ODEs.
        (which is essentially all the functions :func:`integrate_crn_odes` and :func:`plot_crn` do).
    """
    # map each symbol to list of reactions in which it appears
    specie_to_rxn: Dict[Specie, List] = defaultdict(list)
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


def _normalize_crn_initial_values(initial_values: Dict[Union[Specie, sympy.Symbol, str], float]) \
        -> Dict[sympy.Symbol, float]:
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
        initial_values: Dict[Specie, float],
        t_eval: Optional[Iterable[float]] = None,
        t_span: Optional[Tuple[float, float]] = None,
        method: Union[str, OdeSolver] = 'RK45',
        dense_output: bool = False,
        events: Optional[Union[Callable, Iterable[Callable]]] = None,
        vectorized: bool = False,
        args: Optional[Tuple] = None,
        **options,
) -> OdeResult:
    """
    Integrate the ODEs derived from to the given set of chemical reactions.
    This calls :func:`ode.integrate_odes` with the ODEs derived from the given reactions via
    :func:`crn_to_odes`.
    See :func:`ode.integrate_odes` for description of parameters other than `rxns` and `initial_values`.

    Args:
        rxns:
            list of :any:`Reaction`'s comprising the chemical reaction network.
            See documentation for :any:`Reaction` for details on how to specify reactions.

        initial_values:
            dict mapping each species to its initial concentration.
            Note that unlike the parameter `initial_values` in :func:`ode.integrate_odes`,
            keys in this dict must be :any:`Specie` objects, not strings or sympy symbols.

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
        initial_values: Dict[Specie, float],
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
    Plot the ODEs derived from to the given set of chemical reactions.
    This calls :func:`ode.plot` with the ODEs derived from the given reactions via
    :func:`crn_to_odes`.

    See :func:`crn.integrate_crn_odes`, :func:`ode.integrate_odes`, and :func:`ode.plot`
    for description of parameters.
    As with :func:`ode.plot`, the keyword arguments in `options` are passed to
    matplotlib.pyplot.plot
    (https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html),
    as well as to
    scipy.integrate.solve_ivp
    (https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html),
    and as with :func:`ode.plot`, keyword arguments not recognized by scipy.integrate.solve_ivp
    (such as those intended for matplotlib.pyplot.plot) cause `solve_ivp` to print a warning that it
    does not recognize the argument.

    Note that the parameter `dependent_symbols` should use sympy symbols, not :any:`Specie` objects.
    Here is an example of how to use this parameter. Each species that a dependent symbol depends on
    should be represented by a sympy symbol with the same name as the corresponding :any:`Specie` object:

    .. code-block:: python

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

    Args:
        dependent_symbols:
            dict mapping each symbol to an expression that defines its value in terms of other symbols.
            Note that these are not :any:`Specie` objects as in the parameter `rxns`, but sympy symbols.
            Symbols used in the expressions must have the same name as :any:`Specie` objects in `rxns`.

    """
    odes = crn_to_odes(rxns)
    initial_values = _normalize_crn_initial_values(initial_values)
    plot(
        odes,
        initial_values=initial_values,
        t_eval=t_eval,
        t_span=t_span,
        dependent_symbols=dependent_symbols,
        figure_size=figure_size,
        symbols_to_plot=symbols_to_plot,
        show=show,
        method=method,
        dense_output=dense_output,
        events=events,
        vectorized=vectorized,
        args=args,
        loc=loc,
        **options,
    )


def find_all_species(rxns: Iterable[Reaction]) -> Tuple[Specie, ...]:
    all_species = []
    all_species_set = set()
    for rxn in rxns:
        for specie in rxn.get_species():
            if specie not in all_species_set:
                all_species.append(specie)
                all_species_set.add(specie)
    return tuple(all_species)


def gillespie_crn_counts(
        rxns: Iterable[Reaction],
        initial_counts: Dict[Specie, int],
        t_eval: Iterable[float],
        dependent_symbols: Optional[Dict[Union[sympy.Symbol, str], Union[sympy.Expr, str]]] = None,
        seed: Optional[int] = None,
        solver_class: type = gp.NumPySSASolver,
        **options,
) -> gp.Results:
    """
    Run the reactions using the GillesPy2 package for discrete simulation using the Gillespie algorithm.

    Any parameters not described here are passed along to the function gillespy2.GillesPySolver.run:
    https://gillespy2.readthedocs.io/en/latest/classes/gillespy2.core.html#gillespy2.core.gillespySolver.GillesPySolver.run


    Args:
        rxns:
            list of :any:`Reaction`'s comprising the chemical reaction network.
            See documentation for :any:`Reaction` for details on how to specify reactions.

        initial_counts:
            dict mapping each species to its initial integer count.
            Note that unlike the parameter `initial_values` in :func:`ode.integrate_odes`,
            keys in this dict must be :any:`Specie` objects, not strings or sympy symbols.

    Returns:
        Same Result object returned by gillespy2.GillesPySolver.run.
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


def plot_gillespie(
        rxns: Iterable[Reaction],
        initial_counts: Dict[Specie, int],
        t_eval: Iterable[float],
        seed: Optional[int] = None,
        dependent_symbols: Optional[Dict[Union[sympy.Symbol, str], Union[sympy.Expr, str]]] = None,
        figure_size: Tuple[float, float] = (10, 3),
        symbols_to_plot: Optional[Iterable[Union[sympy.Symbol, str]]] = None,
        show: bool = False,
        loc: Union[str, Tuple[float, float]] = 'best',
        **options,
) -> None:
    gp_result = gillespie_crn_counts(
        rxns=rxns,
        initial_counts=initial_counts,
        t_eval=t_eval,
        seed=seed,
        dependent_symbols=dependent_symbols,
        **options,
    )

    symbols = tuple(name for name in gp_result[0].keys() if name != 'time')
    assert len(symbols) == len(gp_result[0]) - 1  # -1 for 'time'
    times = np.array(t_eval)
    # convert gp_result to Dict[str, np.ndarray] for _plot_given_values
    result = {symbol: gp_result[0][symbol] for symbol in symbols}
    plot_given_values(
        times=times,
        result=result,
        source='ssa',
        dependent_symbols=dependent_symbols,
        figure_size=figure_size,
        symbols_to_plot=symbols_to_plot,
        show=show,
        loc=loc,
        **options,
    )


def species(sp: Union[str, Iterable[str]]) -> Tuple[Specie, ...]:
    """
    Create a list of :any:`Specie` (Single species :any:`Expression`'s),
    or a single one.

    args:
        sp:
            A string or Iterable of strings representing the names of the species being created.
            If a single string, species names are interpreted as space-separated.

    Examples:

    .. code-block:: python

        w, x, y, z = species('W X Y Z')
        rxn = x + y >> z + w


    .. code-block:: python

        w, x, y, z = species(['W', 'X', 'Y', 'Z'])
        rxn = x + y >> z + w

    """
    species_list: List[str]
    if isinstance(sp, str):
        species_list = sp.split()
    else:
        species_list = [specie.strip() for specie in sp]

    # if len(species_list) == 1:
    #     return Specie(species_list[0])
    if len(species_list) != len(set(species_list)):
        raise ValueError(f'species_list {species_list} cannot contain duplicates.')

    return tuple(Specie(specie) for specie in species_list)


SpeciePair = Tuple['Specie', 'Specie']  # forward annotations don't seem to work here
Output = Union[SpeciePair, Dict[SpeciePair, float]]


def replace_reversible_rxns(rxns: Iterable[Reaction]) -> List[Reaction]:
    """
    Args:
        rxns: list of :any:`Reaction`'s

    Returns:
        list of :any:`Reaction`'s, where every reversible reaction in `rxns` has been replaced by
        two irreversible reactions, and all others have been left as they are
    """
    new_rxns: List[Reaction] = []
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
    name: str

    def __add__(self, other: Union[Specie, Expression]) -> Expression:
        if isinstance(other, Expression):
            return other + Expression([self])
        elif isinstance(other, Specie):
            return Expression([self]) + Expression([other])

        raise NotImplementedError()

    __radd__ = __add__

    def __rshift__(self, other: Union[Specie, Expression]) -> Reaction:
        return Reaction(self, other)

    def __rrshift__(self, other: Union[Specie, Expression]) -> Reaction:
        return Reaction(other, self)

    def __or__(self, other: Union[Specie, Expression]) -> Reaction:
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
    just use the :func:`species` function and manipulate :any:`Specie` objects
    with operators ``>>``, ``|``, ``+``, and ``*`` to create reactions
    (see :any:`Reaction` for examples).
    """

    species: List[Specie]
    """
    ordered list of species in expression, e.g, A+A+B is [A,A,B]
    """

    def __getitem__(self, idx: int) -> Specie:
        """
        Args:
            idx: index of species to return

        Returns:
            :any:`Specie` at index `idx` in this :any:`Expression`
        """
        return self.species[idx]

    def __add__(self, other: Union[Expression, Specie]) -> Expression:
        """
        Args:
            other: :any:`Expression` or :any:`Specie` to add to this one

        Returns:
            :any:`Expression` representing the union of this :any:`Expression` and `other`
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
        Args:
            coeff: coefficient to multiply this :any:`Expression` by

        Returns:
            :any:`Expression` representing this :any:`Expression` multiplied by `coeff`
        """
        if isinstance(coeff, int):
            species_copy = []
            for _ in range(coeff):
                species_copy.extend(self.species)
            return Expression(species_copy)
        else:
            raise NotImplementedError()

    __mul__ = __rmul__

    def __rshift__(self, expr: Union[Specie, Expression]) -> Reaction:
        return Reaction(self, expr)

    def __or__(self, other: Union[Specie, Expression]) -> Reaction:
        return Reaction(self, other, reversible=True)

    def __str__(self) -> str:
        if len(self.species) == 0:
            return '∅'
        return '+'.join(s.name for s in self.species)

    def __len__(self) -> int:
        return len(self.species)

    def get_species(self) -> Set[Specie]:
        """
        Returns the set of species in this expression, not their
        coefficients.
        """
        return set(self.species)

    def species_counts(self, key_type: Literal['str', 'Specie'] = 'Specie') -> Dict[Specie, int]:
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
"""
Used for chemical reactions with empty reactant or product lists, e.g., to implement the exponential
decay reaction :math:`X \\to \\emptyset`:

.. code-block:: python
    
    x = species('X')
    rxn = x >> empty

"""

avogadro = 6.02214076e23


def concentration_to_count(concentration: float, volume: float) -> int:
    """

    Args:
        concentration: units of M (molar) = moles / liter
        volume: units of liter

    Returns:
        count of molecule with `concentration` in `volume`
    """
    return round(avogadro * concentration * volume)


@dataclass
class Reaction:
    """
    Representation of a stoichiometric reaction using a pair of :any:`Expression`'s,
    one for the reactants and one for the products.

    Reactions are constructed by creating objects of type :any:`Specie` and using the operators
    ``>>`` (for irreversible reactions) and ``|`` (for reversible reactions), as well as the ``+`` and
    ``*`` operators to specify the stoichiometric coefficients of the reactants and products,
    and optionally the methods :meth:`Reaction.k` and :meth:`Reaction.r` to specify forward and reverse
    rate constants.

    For example, the following code creates a reaction that represents the irreversible reaction
    :math:`A + B \\rightarrow C` (with implicit rate constant 1.0):

    .. code-block:: python

        a,b,c = species('A B C')
        rxn = a+b >> c

    To create reactions

    .. math::

        A+B &\\underset{4.1}{\\stackrel{0.6}{\\rightleftharpoons}} 2C

        C   &\\xrightarrow{5.2} D

    use the following code:

    .. code-block:: python

        a,b,c,d = gpac.species('A B C D')
        rxns = [
            (a+b | 2*c).k(0.6).r(4.1),
            (c >> d).k(5.2),
        ]

    Also supported are inhibitors, which can be added to reactions using the method :meth:`Reaction.i`:

    .. code-block:: python

        a, b, u, i = species('A B U I')
        rxn = (a+b | 2*u).i(i, 100)

    which represents the reaction :math:`A+B \\to 2U` with inhibitor :math:`I` and inhibitor constant 100.
    Currently the inhibitor is modeled using a first-order Hill function, i.e., its contribution to the
    reaction rate is to divide by :math:`1 + k \\cdot I`, where :math:`k` is the inhibitor constant.
    So for the reaction defined above, its rate is :math:`k \\cdot [A] \\cdot [B] / (1 + 100 \\cdot [I])`.
    """

    reactants: Expression
    """The left side of species in the reaction."""

    products: Expression
    """The right side of species in the reaction."""

    rate_constant: float = 1.0
    """Rate constant of forward reaction."""

    rate_constant_reverse: float = 1.0
    """Rate constant of reverse reaction (only used if :py:data:`Reaction.reversible` is true)."""

    reversible: bool = False
    """Whether reaction is reversible, i.e. `products` :math:`\\to` `reactants` is a reaction also."""

    inhibitors: List[Specie] = field(default_factory=list)
    """Inhibitors of the reaction."""

    inhibitor_constants: List[float] = field(default_factory=list)

    def __init__(self, reactants: Union[Specie, Expression], products: Union[Specie, Expression],
                 k: float = 1, r: float = 1,
                 reversible: bool = False) -> None:
        """
        In general this constructor should not be used directly; instead, use the operators ``>>``,
        ``|``, ``+``, and ``*`` to construct reactions. (See description of :any:`Reaction` for
        examples.)

        Args:
            reactants: left side of species in the reaction
            products: right side of species in the reaction
            k: Rate constant of forward reaction
            r: Rate constant of reverse reaction (only used if :py:data:`Reaction.reversible` is true
            reversible: Whether reaction is reversible
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
        Args:
            inhibitor: The inhibitor species
            constant: The inhibitor constant
        """
        self.inhibitors.append(inhibitor)
        self.inhibitor_constants.append(constant)
        return self

    def i(self, inhibitor: Specie, constant: float = 1.0) -> Reaction:
        """
        alias for :meth:`Reaction.with_inhibitor`
        """
        return self.with_inhibitor(inhibitor, constant)

    def get_ode(self, specie: Specie, reverse: bool = False) -> sympy.Expr:
        """

        Args:
            specie:
                A :any:`Specie` that may or may not appear in this :any:`Reaction`.

            reverse:
                Whether to interpret this reaction in reverse, i.e., treat products as reactants
                and vice versa. Raises exception if the reaction is not reversible.

        Returns:
            sympy expression for the ODE term for the given :any:`Specie`.
            For example, if the reaction is :math:`A+B \\to 2C`,
            then the ODE for :math:`A` is :math:`-k \\cdot A \\cdot B`,
            the ODE for B is :math:`-k \\cdot A \\cdot B`,
            and the ODE for C is :math:`2 \\cdot k \\cdot A \\cdot B`.
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
        Returns: true if there is one reactant
        """
        return self.num_reactants() == 1

    def is_bimolecular(self) -> bool:
        """
        Returns: true if there are two reactants
        """
        return self.num_reactants() == 2

    def symmetric(self) -> bool:
        """
        Returns: true if there are two reactants that are the same species
        """
        return self.num_reactants() == 2 and self.reactants.species[0] == self.reactants.species[1]

    def symmetric_products(self) -> bool:
        """
        Returns: true if there are two products that are the same species
        """
        return self.num_products() == 2 and self.products.species[0] == self.products.species[1]

    def num_reactants(self) -> int:
        """
        Returns: number of reactants
        """
        return len(self.reactants)

    def num_products(self) -> int:
        """
        Returns: number of products
        """
        return len(self.products)

    def num_inhibitors(self) -> int:
        """
        Returns: number of inhibitors
        """
        return len(self.inhibitors)

    def is_conservative(self) -> bool:
        """
        Returns: true if number of reactants equals number of products
        """
        return self.num_reactants() == self.num_products()

    def reactant_if_unimolecular(self) -> Specie:
        """
        Returns: unique reactant if there is only one
        Raises: ValueError if there are multiple reactants
        """
        if self.is_unimolecular():
            return self.reactants.species[0]
        else:
            raise ValueError(f'reaction {self} is not unimolecular')

    def product_if_unique(self) -> Specie:
        """
        Returns: unique product if there is only one
        Raises: ValueError if there are multiple products
        """
        if self.num_products() == 1:
            return self.products.species[0]
        else:
            raise ValueError(f'reaction {self} does not have exactly one product')

    def reactants_if_bimolecular(self) -> Tuple[Specie, Specie]:
        """
        Returns: pair of reactants if there are exactly two
        Raises: ValueError if there are not exactly two reactants
        """
        if self.is_bimolecular():
            return self.reactants.species[0], self.reactants.species[1]
        else:
            raise ValueError(f'reaction {self} is not bimolecular')

    def reactant_names_if_bimolecular(self) -> Tuple[str, str]:
        """
        Returns: pair of reactant names if there are exactly two
        Raises: ValueError if there are not exactly two reactants
        """
        r1, r2 = self.reactants_if_bimolecular()
        return r1.name, r2.name

    def products_if_exactly_two(self) -> Tuple[Specie, Specie]:
        """
        Returns: pair of products if there are exactly two
        Raises: ValueError if there are not exactly two products
        """
        if self.num_products() == 2:
            return self.products.species[0], self.products.species[1]
        else:
            raise ValueError(f'reaction {self} does not have exactly two products')

    def product_names_if_exactly_two(self) -> Tuple[str, str]:
        """
        Returns: pair of product names if there are exactly two
        Raises: ValueError if there are not exactly two products
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
        Same as :meth:`Reaction.f`.

        args:
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

        .. code-block:: python

            x, y, z = species("X Y Z")
            rxns = [
                (x + y >> z).f(2.5),
                (z >> x).f(1.5),
                (z >> y).f(0.5)),
            ]

        Note that if this is a reversible reaction, this specifies the *forward* rate constant.

        args:
            coeff: float
                The new (forward) reaction coefficient
        """
        self.rate_constant = coeff
        return self

    def r(self, coeff: float) -> Reaction:
        """
        Changes the reverse reactionn reaction rate constant to `coeff` and returns `self`.

        This is useful for including the rate constant during the construction
        of a reaction. For example, the following defines a reversible reaction
        :math:`X + Y \\rightleftharpoons Z` with forward rate constant 2.5 and reverse rate constant 1.5.

        .. code-block:: python

            x, y, z = species("X Y Z")
            rxn = (x + y | z).k(2.5).r(1.5)

        args:
            coeff: float
                The new reverse reaction rate constant
        """
        if not self.reversible:
            raise ValueError('cannot set r on an irreversible reaction')
        self.rate_constant_reverse = coeff
        return self

    def get_species(self) -> Tuple[Specie, ...]:
        """
        Return: the set of species present in the reactants, products, and inhibitors, in the order.
        """
        all_species = []
        all_species_set = set()
        for s in self.reactants.species + self.products.species + self.inhibitors:
            if s not in all_species_set:
                all_species.append(s)
                all_species_set.add(s)
        return tuple(all_species)
