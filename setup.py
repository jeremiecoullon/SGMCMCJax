from setuptools import setup

EXTRAS = {
    "tests": [
        "pytest",
    ],
    "docs": [
        "furo==2020.12.30b24",
        "nbsphinx==0.8.1",
        "nb-black==1.0.7",
        "matplotlib==3.3.3",
        "sphinx-copybutton==0.3.5",
    ],
}

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
    extras_require=EXTRAS,
    setup_requires=['pytest-runner'],
    tests_require=['pytest']
)
