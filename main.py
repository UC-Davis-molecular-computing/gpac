import sympy
import gpac
import numpy as np

x,y,f,s = sympy.symbols('x y f s')

odes = {
    x: (y-f)*s,
    f: x**2 - f,
    y: 0,
    s: -s*s,
}
initial_values = {
    s: 1,
    x: 0,
    f: 0,
    y: 14**2,
}
times = np.linspace(0, 10, 20)

gpac.plot(odes, initial_values, times=times, figure_size=(20,4))