from setuptools import setup

with open('README.md') as f:
    long_description = f.read()

setup(name='d3_stars',
      version='0.0.1',
      description='A plugin for running dynamical simulations of MESA models using Dedalus-v3',
      longdescription=long_description,
      url='https://github.com/evanhanders/d3_stars',
      author='Evan H. Anders',
      author_email='evan.anders@northwestern.edu',
      classifiers=['Programming Langulage :: Python :: 3'],
      packages=setuptools.find_packages(),
      package_data={'': ['d3_stars.cfg']},
      license='GPL-3.0')
