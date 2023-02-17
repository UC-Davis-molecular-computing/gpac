"""
Module for expressing chemical reaction networks and deriving their ODEs. Ideas and much code taken from
https://github.com/enricozb/python-crn.

For example, to specify the "approximate majority" reactions
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
"""

from __future__ import annotations  # needed for forward references in type hints

from typing import Dict, Iterable, Tuple, Set, Union, Optional, Callable, List
from collections import defaultdict
import copy
from dataclasses import dataclass

from scipy.integrate import OdeSolver
from scipy.integrate._ivp.ivp import OdeResult
import sympy

from gpac import integrate_odes, plot


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
    specie_to_rxn = defaultdict(list)
    for rxn in rxns:
        for specie in rxn.get_species():
            specie_to_rxn[specie].append(rxn)

    odes = {}
    for specie, rxns in specie_to_rxn.items():
        ode = sympy.sympify(0)
        for rxn in rxns:
            ode += rxn.get_ode(specie)
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
        show: bool = True,
        method: Union[str, OdeSolver] = 'RK45',
        dense_output: bool = False,
        events: Optional[Union[Callable, Iterable[Callable]]] = None,
        vectorized: bool = False,
        args: Optional[Tuple] = None,
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

    Args:
        dependent_symbols:
            dict mapping each symbol to an expression that defines its value in terms of other symbols.
            Note that these are not :any:`Specie` objects as in the parameter `rxns`, but sympy symbols.
            Symbols used in the expressions must have the same name as :any:`Specie` objects in `rxns`.
            For an example, see the example notebook
            https://github.com/UC-Davis-molecular-computing/gpac/blob/main/notebook.ipynb.

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

    return tuple(map(Specie, species_list))


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

    def get_ode(self, specie: Specie) -> sympy.Expr:
        """

        Args:
            specie:
                A :any:`Specie` that may or may not appear in this :any:`Reaction`.

        Returns:
            sympy expression for the ODE term for the given :any:`Specie`.
            For example, if the reaction is :math:`A+B \\to 2C`,
            then the ODE for :math:`A` is :math:`-k \\cdot A \\cdot B`,
            the ODE for B is :math:`-k \\cdot A \\cdot B`,
            and the ODE for C is :math:`2 \\cdot k \\cdot A \\cdot B`.
        """
        if specie not in self.reactants.get_species() and specie not in self.products.get_species():
            return sympy.Integer(0)

        reactant_coeff = self.reactants.species.count(specie)
        product_coeff = self.products.species.count(specie)
        net_produced = product_coeff - reactant_coeff
        reactants_ode = sympy.Integer(1)
        for reactant in self.reactants.get_species():
            reactant_term = sympy.Symbol(reactant.name) ** self.reactants.species.count(reactant)
            reactants_ode *= reactant_term

        # if rate constant is 1.0, avoid the ugly "1.0*" factor in the output
        ode = net_produced * reactants_ode if self.rate_constant == 1.0 \
            else net_produced * self.rate_constant * reactants_ode
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
        rev_rate_str = '' if not self.reversible or self.rate_constant_reverse == 1 \
            else f'({self.rate_constant_reverse})<'
        return f"{self.reactants} {rev_rate_str}-->{for_rate_str} {self.products}"

    def __repr__(self) -> str:
        return (f"Reaction({repr(self.reactants)}, {repr(self.products)}, "
                f"rate_constant={self.rate_constant}, "
                f"rate_constant_reverse={self.rate_constant_reverse}, "
                f"reversible={self.reversible})")

    def k(self, coeff: float) -> Reaction:
        """
        Changes the reaction coefficient to `coeff` and returns `self`.

        This is useful for including the rate constant during the construction
        of a reaction. For example

        .. code-block:: python

            x, y, z = species("X Y Z")
            rxns = [
                (x + y >> z).k(2.5),
                (z >> x).k(1.5),
                (z >> y).k(0.5)),
            ]

        args:
            coeff: float
                The new reaction coefficient
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

    def get_species(self) -> Tuple[Specie]:
        """
        Return: the set of species present in the products and reactants, in the order.
        """
        all_species = []
        all_species_set = set()
        for s in self.reactants.species + self.products.species:
            if s not in all_species_set:
                all_species.append(s)
                all_species_set.add(s)
        return tuple(all_species)


def species_in_rxns(rxns: Iterable[Reaction]) -> List[Specie]:
    """
    Args:
        rxns: iterable of :any:`Reaction`'s

    Returns:
        list of species (without repetitions) in :any:`Reaction`'s in `rxns`
    """
    species_set: Set[Specie] = set()
    species_list: List[Specie] = []
    for rxn in rxns:
        for sp in rxn.reactants.species + rxn.products.species:
            if sp not in species_set:
                species_set.add(sp)
                species_list.append(sp)
    return species_list
