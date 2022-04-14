import pathlib

from setuptools import find_packages, setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

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
        "Sphinx==4.4.0",
        "isort==5.10.1",
        "sphinx-rtd-theme==1.0.0",
    ],
}

setup(
    name="SGMCMCJax",
    version="0.2.12",
    author="Jeremie Coullon",
    author_email="jeremie.coullon@gmail.com",
    packages=find_packages(".", exclude=["tests"]),
    license="LICENSE.txt",
    description="SGMCMC samplers in JAX",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/jeremiecoullon/SGMCMCJax",
    install_requires=["jax", "jaxlib", "tqdm", "optax"],
    extras_require=EXTRAS,
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
)
