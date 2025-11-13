import pytest
from .matloader import load_mat_workspace

@pytest.fixture(scope="session")
def loadmat():
    return load_mat_workspace