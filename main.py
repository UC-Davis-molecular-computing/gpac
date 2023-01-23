# plot solution to rock-paper-scissors (RPS) oscillator described by these chemical reactions:
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
    'a': 10,
    b: 1,
    c: 1,
}
times = np.linspace(0, 5, 200)

gpac.plot(odes, initial_values, times=times, figure_size=(20,4), symbols_to_plot=[a,c])