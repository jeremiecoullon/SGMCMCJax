from setuptools import setup

setup(
name='SGMCMCJax',
version='0.1.0',
author='Jeremie Coullon',
author_email='jeremie.coullon@gmail.com',
license='LICENSE.txt',
description='SGMCMC samplers in JAX',
install_requires=[
   "jax",
   "jaxlib",
   "tqdm"
],
)
