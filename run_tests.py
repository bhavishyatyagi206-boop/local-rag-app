"""
Test Runner script for Local RAG Engine.
Executes pytest across the test suite and returns exit code.
"""

import sys
import pytest

if __name__ == "__main__":
    args = ["-v", "tests"] + sys.argv[1:]
    exit_code = pytest.main(args)
    sys.exit(exit_code)
