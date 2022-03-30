# OpportunityML

Opportunity HAR Dataset - "Higher level"-Activity Recognition

## How to

- all executable files in src/experiments (and src/tests)
- to run something:
  - conda env required
  - in src/runner.py comment in the experiment- or test-python file you want to run (import experiments.example_pythonfile or import tests.test_example_pythonfile)
  - python3 src/runner.py

## Environment setup

We use a Formatter (Black) and a Linter (PyLint) for the code. The included vscode configuration lets them run on every save.

- Black: In VSCode, open this folder and save a file. If black is not installed, it will ask you what to do. Choose "Yes" for installing it.
- PyLint: A similar dialog should appear when PyLint is not installed. Install it.

## Guidelines

- ml coding is based on experiments
  - we explicitly allow to copy code (break the software development rule) in some cases
    - like the k-fold cross validation, there is no good modularity possible as its changes too often
