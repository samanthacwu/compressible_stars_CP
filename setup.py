from setuptools import setup

with open('README.md') as f:
    long_description = f.read()

setup(name='compstar',
      version='1.0.0',
      description='A plugin for running dynamical simulations of MESA models using Dedalus-v3',
      longdescription=long_description,
      url='https://github.com/evanhanders/compressible_stars',
      author='Evan H. Anders',
      author_email='evan.anders@northwestern.edu',
      classifiers=['Programming Langulage :: Python :: 3'],
      packages=setuptools.find_packages(),
      package_data={'': ['compstar.cfg']},
      license='GPL-3.0')
