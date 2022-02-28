# Guidelines for Contributing

SGMCMCJax welcomes contributions from interested individuals or groups. These guidelines help explain how you can contribute to the library

There are 4 main ways of contributing to the library (in descending order of difficulty or scope):

* Adding new or improved functionality to the existing codebase
* Fixing outstanding issues (bugs) with the existing codebase. They range from low-level software bugs to higher-level design problems.
* Contributing or improving the documentation (`docs`) or examples (`sgmcmcjax/examples`)
* Submitting issues related to bugs or desired enhancements

# Opening issues

Please open issues on [Github Issue Tracker](https://github.com/jeremiecoullon/SGMCMCJax/issues).


# Contributing code via pull requests

Please submit patches via pull requests.

The preferred workflow for contributing is to fork the [GitHub repository](https://github.com/jeremiecoullon/SGMCMCJax), clone it to your local machine, and develop on a feature branch.

## Steps:

1. Fork the [project repository](https://github.com/jeremiecoullon/SGMCMCJax) by clicking on the 'Fork' button near the top right of the main repository page. This creates a copy of the code under your GitHub user account.

2. Clone your fork of the SGMCMCJax repo from your GitHub account to your local disk, and add the base repository as a remote:

   ```bash
   $ git clone git@github.com:<your GitHub handle>/SGMCMCJax.git
   $ cd SGMCMCJax
   $ git remote add upstream git@github.com:SGMCMCJax.git
   ```

3. Create a ``feature`` branch to hold your development changes:

   ```bash
   $ git checkout -b my-feature
   ```

   Always use a ``feature`` branch. It's good practice to never routinely work on the ``master`` branch of any repository.

4. Project requirements are in ``requirements.txt``, and libraries used for development are in ``requirements-dev.txt``. 


   You may (probably in a [virtual environment](https://docs.python-guide.org/dev/virtualenvs/)) run:

   ```bash
   $ pip install -e .
   $ pip install -r requirements-dev.txt
   ```


5. Develop the feature on your feature branch. Add changed files using ``git add`` and then ``git commit`` files:

   ```bash
   $ git add modified_files
   $ git commit
   ```

   to record your changes locally.
   After committing, it is a good idea to sync with the base repository in case there have been any changes:
   ```bash
   $ git fetch upstream
   $ git rebase upstream/main
   ```

   Then push the changes to your GitHub account with:

   ```bash
   $ git push -u origin my-feature
   ```

6. Go to the GitHub web page of your fork of the SGMCMCJax repo. Click the 'Pull request' button to send your changes to the project's maintainer for review.

## Pull request checklist

We recommended that your contribution complies with the following guidelines before you submit a pull request:

*  If your pull request addresses an issue, please use the pull request title to describe the issue and mention the issue number in the pull request description. This will make sure a link back to the original issue is created.

*  All public methods must have informative docstrings 

*  Please prefix the title of incomplete contributions with `[WIP]` (to indicate a work in progress). WIPs may be useful to (1) indicate you are working on something to avoid duplicated work, (2) request broad review of functionality or API, or (3) seek collaborators.

*  All other tests pass when everything is rebuilt from scratch. 

* Documentation and high-coverage tests are necessary for enhancements to be accepted.


* Code with good test, check with:

  ```bash
  $ pip install -r requirements-dev.txt
  $ pytest -v sgmcmcjax tests --cov=sgmcmcjax --cov-report=html
  ```


#### This guide was derived from [PyMC's guide to contributing](https://github.com/pymc-devs/pymc/blob/main/CONTRIBUTING.md)
