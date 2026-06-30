import os
import pytest
from utils import *


# ref: https://stackoverflow.com/questions/22627659/run-code-before-and-after-each-test-in-py-test
@pytest.fixture(autouse=True)
def stop_server_after_each_test():
    # record servers that are already alive before the test
    before = set(server_instances)
    yield
    # only stop servers that were started by this test
    # (servers started by module-scoped fixtures stay alive)
    for srv in list(server_instances):
        if srv not in before:
            srv.stop()


@pytest.fixture(scope="module", autouse=True)
def do_something():
    # this will be run once per test session, before any tests
    ServerPreset.load_all()
