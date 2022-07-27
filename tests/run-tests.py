"""Convenience method for running unittest's

Unittest's can be run by hand, or they can be discovered automatically
and subsequently run. This script facilitates the latter.

Examples
--------

If your Python path is not already configured

$> export PYTHONPATH=`git rev-parse --show-toplevel`:$PYTHOPATH

Run all tests from the top level of the repo
$> python tests/run-tests.py --root-directory tests

From within a specific subdirectory of tests
$> cd tests/loss
$> python tests/run-tests.py

When not specifying a root directory, you may see a warning about
which directory the script implied.

Notes
-----

This script uses Python's unittest discovery framework. See that
documentation

https://docs.python.org/3/library/unittest.html#unittest.TestLoader.discover

for conditions on what is considered a test module, a test class, and
a test case.

"""

import logging
import unittest
from argparse import ArgumentParser
from pathlib import Path

arguments = ArgumentParser()
arguments.add_argument(
    "--root-directory",
    type=Path,
    help="""
Directory from which to start searching for test files.
""",
)
args = arguments.parse_args()

if args.root_directory is None:
    start = Path.cwd()
    logging.warning(f"No root specified: using {start}")
else:
    start = args.root_directory

suite = unittest.TestSuite()
for i in unittest.defaultTestLoader.discover(start):
    suite.addTests(i)
runner = unittest.TextTestRunner()
runner.run(suite)
