=====
Usage
=====

To use pyFDN in a project::

    from pyFDN import (
        dss2ss,
        one_pole_absorption,
        random_matrix_shift,
        random_orthogonal,
    )

    # Helpers from different subpackages are available at one import layer.
    feedback = random_orthogonal(4)
    shifted_feedback, _, _, _ = random_matrix_shift(2, feedback)
