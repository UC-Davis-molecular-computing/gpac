# gpac Python package

This document is intended to be read on [github.com](https://github.com/UC-Davis-molecular-computing/gpac?tab=readme-ov-file#readme); 
some relative links below will not work on other sites such as PyPI.

## Table of contents

* [Overview](#overview)
* [API](#api)
* [Installation](#installation)
* [Examples](#examples)
  - [Plotting ODEs](#plotting-odes)
  - [Getting trajectory data of ODEs](#getting-trajectory-data-of-odes)
  - [Chemical reaction networks](#chemical-reaction-networks)


## Overview
This is a Python package for simulating General-Purpose Analog Computers as defined and studied by Claude Shannon. It's primarily a front-end to [scipy](https://scipy.org/) and [sympy](https://www.sympy.org/) making it easier to specify systems of ODEs, numerically integrate them, and plot their solutions. It also has support for a very common model governed by polynomial ODEs: continuous mass-action [chemical reaction networks](https://en.wikipedia.org/wiki/Chemical_reaction_network_theory#Overview). (And despite having nothing to do with GPAC or ODEs, it also can simulate discrete CRNs; see [Chemical reaction networks](#chemical-reaction-networks) section below.)

This is ostensibly what [pyodesys](https://github.com/bjodah/pyodesys) does as well, and that package is much more powerful and configurable than gpac. The purpose of gpac is primarily to be simpler to use for common cases of ODEs, at the cost of being less expressive. For example, gpac has some functions ([`plot`](https://gpac.readthedocs.io/en/latest/#gpac.ode.plot) and [`plot_crn`](https://gpac.readthedocs.io/en/latest/#gpac.crn.plot_crn)) to do plotting in matplotlib, which is easier than manually getting the ODE data through [`integrate_odes`](https://gpac.readthedocs.io/en/latest/#gpac.ode.integrate_odes) and passing it along to the matplotlib plot function. This is possible if you want to have more control over how things are plotted than is possible with the gpac plotting functions; however in most cases you can configure what you need in `plot` and `plot_crn` either by passing keyword arguments (which are passed along to the matplotlib plot function), or by calling functions in matplotlib.pyplot (e.g., [`yscale`](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.yscale.html)) after calling gpac's `plot` or `pyplot`.

## API
The API for the package is here: https://gpac.readthedocs.io/

## Installation
Python 3.7 or above is required. There are two ways you can install the `gpac` package, pip or git:

A. **pip:** The easiest option is to install via `pip` by typing the following at the command line:
   ```
   pip install gpac
   ```

B. **git:** The other option is to clone the git repo. You may need to install `git` first: https://git-scm.com/book/en/v2/Getting-Started-Installing-Git

   1. Clone this repo by typing the following at the command line:
      ```
      git clone https://github.com/UC-Davis-molecular-computing/gpac.git
      ```

   2. Install the Python package by changing to the directory where the gpac repository is stored localled and type `pip install -e .` This should install the needed dependencies. After doing this you should be able to import the gpac package in your Python scripts/Jupyter notebooks with `import gpac`. Try testing this out in the Python interpreter:
       ```python
       $ python
       Python 3.9.12 (main, Apr  4 2022, 05:22:27) [MSC v.1916 64 bit (AMD64)] :: Anaconda, Inc. on win32
       Type "help", "copyright", "credits" or "license" for more information.
       >>> import gpac
       >>>
       ```

## Examples
See more examples in the Jupyter notebook [notebook.ipynb](notebook.ipynb).

### Plotting ODEs
ODEs are specified by creating [sympy](https://www.sympy.org/) symbols and expressions (or if you like, Python strings), represented as a Python dict `odes` mapping each variable---a single sympy symbol or Python string---to an expression representing its time derivative, represented as a sympy expression composed of sympy symbols (or if the derivative is constant, a Python `int` or `float`).

Every symbol that appears in any of the expressions *must* also be a key in this dict.

The initial values are specified as a Python dict `inits` mapping variables (again, sympy symbols or strings) to their initial values (floats). If you leave out a symbol as a key to `inits`, it is assumed to have initial value 0.

Finally, you can specify the times at which to solve for the ODEs as an iterable of floats `t_eval`. (This is optional; if not specified it uses the time values 0.0, 0.01, 0.02, 0.03, ..., 0.98, 0.99, 1.0)

Remaining parameters are optional (see below for examples of them). See API documentation for [`integrate_odes`](https://gpac.readthedocs.io/en/latest/#gpac.ode.integrate_odes) and [`plot`](https://gpac.readthedocs.io/en/latest/#gpac.ode.plot) for more details.

```python
import sympy
import gpac
import numpy as np

a,b,c = sympy.symbols('a b c')

# ODEs specified as dict mapping each variable to expression describing its derivative.
# key representing variable can be a sympy Symbol or string.
# value representing derivative can be a sympy Expr, string, or (if constant) int or float.
odes = {            # represents ODEs:
    a: -a*b + c*a,  # d/dt a(t) = -a(t)*b(t) + c(t)*a(t)
    b: -b*c + a*b,  # d/dt b(t) = -b(t)*c(t) + a(t)*b(t)
    c: -c*a + b*c,  # d/dt c(t) = -c(t)*a(t) + b(t)*c(t)
}
inits = {
    a: 10,
    b: 1,
    c: 1,
}
t_eval = np.linspace(0, 5, 200)

gpac.plot(odes, inits, t_eval=t_eval, symbols_to_plot=[a,c])
```

![](images/rps-a-c.png)


### Getting trajectory data of ODEs
If you want the data itself from the ODE numerical integration (without plotting it), you can call [`integrate_odes`](https://gpac.readthedocs.io/en/latest/#gpac.ode.integrate_odes) (replace the call to [`plot`](https://gpac.readthedocs.io/en/latest/#gpac.ode.plot) above with the following code).

```python
t_eval = np.linspace(0, 1, 5)

solution = gpac.integrate_odes(odes, initial_values, t_eval=t_eval)
print(f'times = {solution.t}')
print(f'a = {solution.y[0]}')
print(f'b = {solution.y[1]}')
print(f'c = {solution.y[2]}')
```
which prints
```
times = [0.   0.25 0.5  0.75 1.  ]
a = [10.          4.84701622  0.58753815  0.38765743  3.07392998]
b = [1.         6.84903338 9.63512628 3.03634559 0.38421121]
c = [1.         0.3039504  1.77733557 8.57599698 8.54185881]
```
The value `solution` returned by `integrate_odes` is the same object returned from [`scipy.integrate.solve_ivp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html).


### Chemical reaction networks
There are also functions 
[`integrate_crn_odes`](https://gpac.readthedocs.io/en/latest/#gpac.crn.integrate_crn_odes) and 
[`plot_crn`](https://gpac.readthedocs.io/en/latest/#gpac.crn.plot_crn), 
which take as input a description of a set of chemical reactions, derives their ODEs, then integrates/plots them. They both use the function [`crn_to_odes`](https://gpac.readthedocs.io/en/latest/#gpac.crn.crn_to_odes), which converts chemical reactions into ODEs.

Reactions are constructed using operations on [`Specie`](https://gpac.readthedocs.io/en/latest/#gpac.crn.Specie) objects returned from the function [`species`](https://gpac.readthedocs.io/en/latest/#gpac.crn.species):

```python
# plot solution to ODEs of this CRN that computes f(x) = x^2, using the gpac.crn module
# 2X -> 2X+Y
# Y -> nothing
x,y = gpac.species('X Y')
rxns = [
    x+x >> x+x+y,
    y >> gpac.empty,
]
initial_values = {x:5}
t_eval = np.linspace(0, 5, 100)

# plot trajectory of concentrations
gpac.plot_crn(rxns, initial_values, t_eval=t_eval, figure_size=(20,4))
```

See [notebook.ipynb](notebook.ipynb) for more examples.

Although they appear similar, a [`Specie`](https://gpac.readthedocs.io/en/latest/#gpac.crn.Specie) object (such as `x` and `y` returned from the [`species`](https://gpac.readthedocs.io/en/latest/#gpac.crn.species) function above) is different from a [`sympy.Symbol`](https://docs.sympy.org/latest/modules/core.html#sympy.core.symbol.Symbol) object. The [`Specie`](https://gpac.readthedocs.io/en/latest/#gpac.crn.Specie) object is intended to help specify reactions using the notation above with the symbols `+`, `>>`, and `|` (as well as the `k` and `r` functions for specifying non-unit rate constants, see example [notebook](notebook.ipynb)). However, either [`Specie`](https://gpac.readthedocs.io/en/latest/#gpac.crn.Specie) or [`sympy.Symbol`](https://docs.sympy.org/latest/modules/core.html#sympy.core.symbol.Symbol) objects can be a key in the `inits` parameter to [`plot_crn`](https://gpac.readthedocs.io/en/latest/#gpac.crn.plot_crn) and [`integrate_crn_odes`](https://gpac.readthedocs.io/en/latest/#gpac.crn.integrate_crn_odes). 

#### Discrete chemical reaction networks
Going off-topic from the name of the package, gpac also supports discrete CRN simulation, using the blazingly fast package [rebop](https://pypi.org/project/rebop/) that implements the [Gillespie algorithm](https://en.wikipedia.org/wiki/Gillespie_algorithm). See the functions 
[`plot_gillespie`](https://gpac.readthedocs.io/en/latest/#gpac.crn.plot_gillespie), 
[`rebop_crn_counts`](https://gpac.readthedocs.io/en/latest/#gpac.crn.rebop_crn_counts), and
[`rebop_sample_future_configurations`](https://gpac.readthedocs.io/en/latest/#gpac.crn.rebop_sample_future_configurations).
