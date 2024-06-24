import os
import pytest
import sys

"""
Utilities for setting up tests.
@TODO implement app create method and connect to a test database / config for test database
"""


@pytest.fixture(scope="class")
def create_test_app(request):
    sys.path.append((os.path.dirname(os.path.abspath(__file__))))

    def teardown():
        """
        Cleanup after tests have run
        """
        pass

    return False
