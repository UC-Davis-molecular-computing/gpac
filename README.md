# gpac Python package



## Table of contents

* [Overview](#overview)
* [Installation](#installation)
* [Example](#example)


## Overview
This will eventually be a package for simulating General-Purpose Analog Computers as defined and studied by Claude Shannon. So far it's just a front-end to scipy and sympy making it easier to numerically integrate systems of ODEs and plot their solutions.

## Installation
1. Install the dependencies by typing  
   ```
   pip install numpy scipy matplotlib sympy
   ```
   at the command line.

2. Clone this repo by typing 
   ```
   git clone https://github.com/UC-Davis-molecular-computing/gpac.git
   ```
   at the command line. You may need to install `git` first: https://git-scm.com/book/en/v2/Getting-Started-Installing-Git

3. Add the directory into which you cloned it to your PYTHONPATH environment variable. See https://www.geeksforgeeks.org/pythonpath-environment-variable-in-python/ for example if you don't know how to do this. After doing this you should be able to import the gpac package in your Python scripts/Jupyter notebooks with `import gpac`. Try testing this out in the Python interpreter:
    ```python
    $ python
    Python 3.9.12 (main, Apr  4 2022, 05:22:27) [MSC v.1916 64 bit (AMD64)] :: Anaconda, Inc. on win32
    Type "help", "copyright", "credits" or "license" for more information.
    >>> import gpac
    >>>
    ```

## Example
The following is an example of what can be done currently, which is simply to numerically integrate and plot a system of ODEs (ordinary differential equations).

The ODEs are specified by creating [sympy](https://www.sympy.org/) symbols and expressions (or if you like, Python strings), represented as a Python dict `odes` mapping each variable---a single sympy symbol or Python string---to an expression representing its time derivative, represented as a sympy expression composed of sympy symbols (or for convenience you can also use Python strings, or if the derivative is constant, a Python `int` or `float`). Every symbol that appears in any of the expressions must also be a key in this dict.

The initial values are specified as a Python dict `initial_values` mapping variables (again, sympy symbols or strings) to their initial values (floats). Here, if you leave out a symbol as a key to `initial_values`, it is assumed to have initial value 0.

Finally, you can specify the times at which to solve for the ODEs as an iterable of floats. (This is optional; if not specified it uses the time values 0.0, 0.01, 0.02, 0.03, ..., 0.98, 0.99, 1.0)

Remaining parameters are optional (see below for examples of them).

```python
from sympy import symbols
from gpac import plot
import numpy as np

a,b,c = symbols('a b c')

odes = {
    'a': -a*b + c*a, # keys can be a sympy Symbol or string
    b: '-b*c + a*b', # values can be a symbol Expr or string
    c: -c*a + b*c,
}
initial_values = {
    a: 10,
    b: 1,
    c: 1,
}
times = np.linspace(0, 3, 200)

plot(odes, initial_values, times=times, figure_size=(20,4), symbols_to_plot=[a,c])
```

![](images/rps-a-c.png)

See also the Jupyter notebook [notebook.ipynb](notebook.ipynb).

If you want the data itself from the ODE numerical integration (without plotting it), you can call `gpac.integrate_odes`:

```python
# print solution to rock-paper-scissors (RPS) oscillator described by these chemical reactions:
# A+B -> 2B
# B+C -> 2C
# C+A -> 2A

import sympy
import gpac
import numpy as np

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
times = np.linspace(0, 1, 11)

solution = gpac.integrate_odes(odes, initial_values, times=times)
print(f'times = {solution.t}')
print(f'a = {solution.y[0]}')
print(f'b = {solution.y[1]}')
print(f'c = {solution.y[2]}')
```
which prints
```
times = [0.  0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1. ]
a = [10.          9.08888293  6.46163896  3.37229016  1.40901231  0.58753815
  0.325435    0.31385821  0.53876259  1.2704187   3.07392998]
b = [1.         2.46492776 5.24048262 8.26584068 9.86672121 9.63512628
 7.58732937 4.44015128 1.97279354 0.79948706 0.38421121]
c = [1.         0.4461893  0.29787842 0.36186916 0.72426647 1.77733557
 4.08723563 7.24599051 9.48844387 9.93009424 8.54185881]
```