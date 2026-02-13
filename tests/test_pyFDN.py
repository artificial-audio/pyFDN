#!/usr/bin/env python

"""Tests for `pyFDN` package."""

import pytest
import pyFDN
from pyFDN.auxiliary.acoustics import one_pole_absorption
from pyFDN.auxiliary.utils import mag2db
from pyFDN.generate.random_orthogonal import random_orthogonal
from pyFDN.process import process_fdn


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


def test_process_fdn_camel_case_alias():
    """Backward-friendly camelCase alias should point to process_fdn."""
    assert pyFDN.processFDN is pyFDN.process_fdn
