#!/usr/bin/env python

"""Tests for `pyFDN` package."""

import numpy as np
import pytest
import pyFDN
from pyFDN import one_pole_absorption
from pyFDN import mag2db
from pyFDN import random_orthogonal
from pyFDN import process_fdn


@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # import requests
    # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')


def test_content(response):
    """Sample pytest test function with the pytest fixture as an argument."""
    # from bs4 import BeautifulSoup
    # assert 'GitHub' in BeautifulSoup(response.content).title.string


def test_top_level_exports():
    """Top-level package should expose main user-facing functions."""
    assert pyFDN.one_pole_absorption is one_pole_absorption
    assert pyFDN.random_orthogonal is random_orthogonal
    assert pyFDN.mag2db is mag2db
    assert pyFDN.process_fdn is process_fdn
    assert pyFDN.random_orthogonal(3).shape == (3, 3)
    assert pyFDN.mag2db([1.0])[0] == pytest.approx(0.0)
    assert pyFDN.one_pole_absorption(1.0, 0.5, [10, 20], 48_000).shape == (6, 2)
    assert pyFDN.process_fdn(
        np.array([1.0, 0.0]),
        delays=np.array([1]),
        feedback_matrix=np.array([[0.0]]),
        input_gain=np.array([[1.0]]),
        output_gain=np.array([[1.0]]),
        direct=np.array([[0.0]]),
    ).shape == (2,)


def test_process_fdn_no_camel_case_alias():
    """Only snake_case process_fdn should be exported."""
    assert callable(pyFDN.process_fdn)
    assert not hasattr(pyFDN, "processFDN")
