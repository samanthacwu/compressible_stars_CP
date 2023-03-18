# compstar
Code for running simulations of fully compressible convection in stellar interiors using dedalus' d3 framework.

## Installation 

To install compstar on your local machine, clone the repository and then navigate to the root compressible_stars/ directory (where setup.py is located), and type:

> pip3 install -e .

## Dependencies

Note that compstar uses the following packages; follow appropriate links below for installation instructions:

* [Dedalus v3](https://github.com/DedalusProject/dedalus) (master branch, [commit 29f3a59](https://github.com/DedalusProject/dedalus/commit/29f3a59c5ee7cbb7be5d846e35f0c514ac032af6)).
* [py_mesa_reader](https://github.com/wmwolf/py_mesa_reader) (for reading MESA models)
* [astropy](https://www.astropy.org/): pip-installable (pip3 install astropy)
* [plotpal](https://github.com/evanhanders/plotpal) (for some post-processing)
* [pygyre](https://pygyre.readthedocs.io/en/latest/installation.html) (for gyre post-processing)
* [pymsg](https://github.com/rhdtownsend/msg) (for msg post-processing)
