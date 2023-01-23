from sympy import symbols
from gpac import plot
import numpy as np

def main():
    x, y = symbols('x y')

    odes = {
        x: 0,
        y: y - x ** 2,
    }
    initial_values = {
        x: 5,
        y: 0,
    }
    times = np.linspace(0, 3, 200)

    plot(odes, initial_values, times=times, figure_size=(20, 4))

if __name__ == '__main__':
    main()