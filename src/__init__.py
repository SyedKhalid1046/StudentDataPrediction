"""
Student Academic Performance Machine Learning Package
"""
import os
import sys

# Ensure root directory is on python path
_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)
