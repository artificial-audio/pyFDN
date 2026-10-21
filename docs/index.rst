.. pyFDN documentation master file

=====
pyFDN
=====

Python library for Feedback Delay Networks

``pyFDN`` provides building blocks for designing, simulating, and analysing
Feedback Delay Networks (FDNs). The package focuses on reusable, tested helper
functions that simplify typical FDN workflows such as creating orthogonal
feedback matrices, designing loop filters, and inspecting pole locations.

----

.. grid:: 2
   :gutter: 4

   .. grid-item-card:: Getting Started
      :link: readme
      :link-type: doc

      Installation, overview, and a quick-start code example.

   .. grid-item-card:: Examples Gallery
      :link: examples_gallery
      :link-type: doc

      Interactive Marimo notebooks covering vanilla FDNs,
      absorption design, coupled rooms, and more.

   .. grid-item-card:: API Reference
      :link: api_reference
      :link-type: doc

      Complete reference for all classes, functions, and
      modules in the ``pyFDN`` package.

   .. grid-item-card:: Contributing
      :link: contributing
      :link-type: doc

      How to report bugs, request features, and set up
      a development environment.
   
   .. grid-item-card:: Audio Gallery
      :link: audio_gallery
      :link-type: doc

      Explore different audio examples.


Citing pyFDN
------------

If you use pyFDN in research, please cite the following technical report (not peer reviewed):

Sebastian J. Schlecht, Facundo Franchino, and Jeremy B. Bai (2026).
*pyFDN: A Python Toolbox for Feedback Delay Network Design, Analysis, and Optimization*.
`arXiv:2610.27217 <https://arxiv.org/abs/2610.27217>`_.
DOI: `10.48550/arXiv.2610.27217 <https://doi.org/10.48550/arXiv.2610.27217>`_.

.. code-block:: bibtex

   @techreport{pyFDN2026,
     author = {Schlecht, Sebastian J. and Franchino, Facundo and Bai, Jeremy B.},
     title = {{pyFDN}: A Python Toolbox for Feedback Delay Network Design, Analysis, and Optimization},
     institution = {arXiv},
     number = {arXiv:2610.27217},
     year = {2026},
     doi = {10.48550/arXiv.2610.27217},
     url = {https://arxiv.org/abs/2610.27217},
     note = {Technical report; not peer reviewed}
   }


.. toctree::
   :hidden:

   readme
   installation
   usage
   examples_gallery
   api_reference
   training_losses
   modules
   contributing
   authors
   history
   audio_gallery
