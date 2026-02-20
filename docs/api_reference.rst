=============
API Reference
=============

All functions and classes are accessible from the top-level ``pyFDN`` namespace:

.. code-block:: python

   import pyFDN
   feedback = pyFDN.random_orthogonal(4)

----

Classes
-------

.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - :doc:`Filter Classes <api/filters>`
     - ``ZFilter``, ``ZFIR``, ``ZSOS``, ``ZTF``, ``ZScalar``, ``TFMatrix``
   * - :doc:`DSP Components <api/dsp>`
     - ``FilterMatrix``, ``FeedbackDelay``, ``DFiltMatrix``

Modules
-------

.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - :doc:`pyFDN.auxiliary.acoustics <api/acoustics>`
     - Absorption filters, RT60, echo density, EDC
   * - :doc:`pyFDN.auxiliary.delay <api/delay>`
     - Delay utilities, group delay, sample conversion
   * - :doc:`pyFDN.auxiliary.math <api/math>`
     - Matrix polynomials, interpolation, determinants
   * - :doc:`pyFDN.auxiliary.utils <api/utils>`
     - Unit conversion, normalisation, encoding
   * - :doc:`pyFDN.generate <api/generate>`
     - Random orthogonal, shift matrices, velvet, paraunitary
   * - :doc:`pyFDN.translate <api/translate>`
     - DSS → state-space, DSS → impulse response
   * - :doc:`pyFDN.process <api/process>`
     - FDN audio processing pipeline
   * - :doc:`pyFDN.auxiliary.plot <api/plot>`
     - Impulse response and system matrix visualisation


.. toctree::
   :hidden:
   :maxdepth: 1

   api/filters
   api/dsp
   api/acoustics
   api/delay
   api/math
   api/utils
   api/generate
   api/translate
   api/process
   api/plot
