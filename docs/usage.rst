=====
Usage
=====

All main functions are accessible directly from ``pyFDN``::

    import pyFDN

    feedback = pyFDN.random_orthogonal(4)
    absorption = pyFDN.first_order_absorption(1.2, 0.9, [100, 150, 200, 250], 48_000)
    gain_db = pyFDN.lin_to_db([0.5, 1.0, 2.0])

Or import specific functions::

    from pyFDN import random_orthogonal, first_order_absorption, dss_to_ss

    feedback = random_orthogonal(4)
