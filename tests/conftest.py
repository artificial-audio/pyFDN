import pytest
from .matloader import load_mat_workspace

@pytest.fixture
def loadmat():
    return load_mat_workspace