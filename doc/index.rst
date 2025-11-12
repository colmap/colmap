COLMAP
======

.. figure:: images/sparse.png
    :alt: Sparse reconstruction of central Rome.
    :figclass: align-center

    Sparse model of central Rome using 21K photos produced by COLMAP's SfM
    pipeline.

.. figure:: images/dense.png
    :alt: Dense reconstruction of landmarks.
    :figclass: align-center

    Dense models of several landmarks produced by COLMAP's MVS pipeline.


About
-----

COLMAP is a general-purpose Structure-from-Motion (SfM) and Multi-View Stereo
(MVS) pipeline with a graphical and command-line interface. It offers a wide
range of features for reconstruction of ordered and unordered image collections.
The software is licensed under the new BSD license. If you use this project for
your research, please cite::

    @inproceedings{schoenberger2016sfm,
        author={Sch\"{o}nberger, Johannes Lutz and Frahm, Jan-Michael},
        title={Structure-from-Motion Revisited},
        booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)},
        year={2016},
    }

    @inproceedings{schoenberger2016mvs,
        author={Sch\"{o}nberger, Johannes Lutz and Zheng, Enliang and Pollefeys, Marc and Frahm, Jan-Michael},
        title={Pixelwise View Selection for Unstructured Multi-View Stereo},
        booktitle={European Conference on Computer Vision (ECCV)},
        year={2016},
    }

If you use the image retrieval / vocabulary tree engine, please also cite::

    @inproceedings{schoenberger2016vote,
        author={Sch\"{o}nberger, Johannes Lutz and Price, True and Sattler, Torsten and Frahm, Jan-Michael and Pollefeys, Marc},
        title={A Vote-and-Verify Strategy for Fast Spatial Verification in Image Retrieval},
        booktitle={Asian Conference on Computer Vision (ACCV)},
        year={2016},
    }

The latest source code is available at `GitHub
<https://github.com/colmap/colmap>`_. COLMAP builds on top of existing works and
when using specific algorithms within COLMAP, please also cite the original
authors, as specified in the source code.


Download
--------

Executables and other resources can be downloaded from https://demuc.de/colmap/.


Getting Started
---------------

1. Download the `pre-built binaries <https://demuc.de/colmap/>`_ or build the
   library manually from `source <https://github.com/colmap/colmap>`_
   (see :ref:`Installation <installation>`).
2. Download one of the provided datasets (see :ref:`Datasets <datasets>`)
   or use your own images.
3. Use the **automatic reconstruction** to easily build models
   with a single click (see :ref:`Quickstart <quick-start>`).


Support
-------

Please, use `GitHub Discussions <https://github.com/colmap/colmap/discussions>`_
for questions and the `GitHub issue tracker <https://github.com/colmap/colmap>`_
for bug reports, feature requests/additions, etc.


Acknowledgments
---------------

COLMAP was originally written by `Johannes Schönberger <https://demuc.de/>`__ with
funding provided by his PhD advisors Jan-Michael Frahm and Marc Pollefeys.
The team of core project maintainers currently includes
`Johannes Schönberger <https://github.com/ahojnnes>`__,
`Paul-Edouard Sarlin <https://github.com/sarlinpe>`_, and
`Shaohui Liu <https://github.com/B1ueber2y>`_.

The Python bindings in PyCOLMAP were originally added by
`Mihai Dusmanu <https://github.com/mihaidusmanu>`_,
`Philipp Lindenberger <https://github.com/Phil26AT>`_, and
`Paul-Edouard Sarlin <https://github.com/sarlinpe>`_.

The project has also benefitted from countless community contributions, including
bug fixes, improvements, new features, third-party tooling, and community
support (special credits to `Torsten Sattler <https://tsattler.github.io>`_).


.. toctree::
   :hidden:
   :maxdepth: 2

   install
   tutorial
   concepts
   database
   cameras
   rigs
   format
   datasets
   gui
   cli
   pycolmap/index
   faq
   changelog
   contribution
   license
   bibliography
   legacy
