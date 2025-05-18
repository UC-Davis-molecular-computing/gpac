# gpac API

This is the API documentation for gpac. 
See the [Github page](https://github.com/UC-Davis-molecular-computing/gpac#readme)
for examples of usage and installation instructions.

gpac is a Python package for numerically simulating a general-purpose analog computer (GPAC),
defined by Claude Shannon in 1941 as an abstract model of programmable analog computational devices
such as the differential analyzer created by Vannevar Bush and Harold Locke Hazen in the 1920s.
General descriptions of GPACs can be found 
[here](https://en.wikipedia.org/wiki/General-purpose_analog_computer)
and [here](https://arxiv.org/abs/1805.05729).

It also has support for a very common model governed by polynomial ODEs, that of continuous mass-action
[chemical reaction networks](https://en.wikipedia.org/wiki/Chemical_reaction_network_theory#Overview).
And despite having nothing to do with GPAC or ODEs, 
it also can simulate discrete CRNs using the excellent rebop package; 
see the functions 
[`plot_gillespie`](#gpac.crn.plot_gillespie), 
[`rebop_crn_counts`](#gpac.crn.rebop_crn_counts), and
[`rebop_sample_future_configurations`](#gpac.crn.rebop_sample_future_configurations).

GPACs are typically defined by a circuit with gates that can add, multiply, introduce constants, and
integrate an input with respect to time. The most elegant way to specify a GPAC is by defining a set of
ordinary differential equations (ODEs) corresponding to the output wires of integrator gates in the GPAC
circuit.

So essentially, this package makes it easy to write down such ODEs and numerically integrate and plot them.

Although gpac has two submodules ode and crn, you can import all elements from both directly from gpac,
e.g., `#!py from gpac import plot, plot_crn`.

There are many examples in the 
[jupyter notebook](https://github.com/UC-Davis-molecular-computing/gpac/blob/main/notebook.ipynb) 
on Github.

## API Reference

::: gpac.ode
    options:
      show_root_heading: true
      show_source: false
      members_order: source
      show_if_no_docstring: false
      separate_signature: true
      show_signature_annotations: true
      show_symbol_type_heading: true
      docstring_section_style: table

::: gpac.crn
    options:
      show_root_heading: true
      show_source: false
      members_order: source
      show_if_no_docstring: false
      separate_signature: true
      show_signature_annotations: true
      show_symbol_type_heading: true
      docstring_section_style: table
