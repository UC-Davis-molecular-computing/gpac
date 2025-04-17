"""
gpac is a Python package for numerically simulating a general-purpose analog computer (GPAC),
defined by Claude Shannon in 1941 as an abstract model of programmable analog computational devices
such as the differential analyzer created by Vannevar Bush and Harold Locke Hazen in the 1920s.
See here for a description of GPACs:

    - [General Purpose Analog Computer](https://en.wikipedia.org/wiki/General_purpose_analog_computer)
    - [GPAC Paper on arXiv](https://arxiv.org/abs/1805.05729)

It also has support for a very common model governed by polynomial ODEs, the of continuous mass-action
chemical reaction networks:

    - [Chemical Reaction Network Theory Overview](https://en.wikipedia.org/wiki/Chemical_reaction_network_theory#Overview)

GPACs are typically defined by a circuit with gates that can add, multiply, introduce constants, and
integrate an input with respect to time. The most elegant way to specify a GPAC is by defining a set of
ordinary differential equations (ODEs) corresponding to the output wires of integrator gates in the GPAC
circuit.

So essentially, this package makes it easy to write down such ODEs and numerically integrate and plot them.

Although gpac has two submodules ode and crn, you can import all elements from both directly from gpac,
e.g., ``from gpac import plot, plot_crn``.
"""

from gpac.ode import *
from gpac.crn import *
