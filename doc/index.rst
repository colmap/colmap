:html_theme.sidebar_secondary.remove: true

.. rst-class:: hero__title

COLMAP
======

.. raw:: html

   <div class="hero">
     <p class="hero__tagline">General-purpose Structure-from-Motion &amp; Multi-View Stereo</p>
     <p class="hero__desc">
       COLMAP is a general-purpose Structure-from-Motion (SfM) and Multi-View
       Stereo (MVS) pipeline with a graphical and command-line interface. It
       offers a wide range of features for reconstruction of ordered and
       unordered image collections, and is free and open source.
     </p>
     <div class="hero__cta">
       <a class="hero__cta--primary" href="install.html">Get Started</a>
       <a class="hero__cta--secondary" href="#install-colmap">Install</a>
       <a class="hero__cta--secondary" href="https://github.com/colmap/colmap">GitHub</a>
     </div>
   </div>

.. figure:: images/sparse.png
   :class: hero__image
   :figclass: hero-figure
   :alt: Sparse reconstruction of central Rome.

   Sparse model of central Rome using 21K photos produced by COLMAP's SfM
   pipeline.


Install COLMAP
--------------

Select your platform below to get the recommended install command or download.

.. raw:: html

   <div id="colmap-install-selector"></div>

For all installation options and build-from-source instructions, see the
:ref:`installation guide <installation>`.


Features
--------

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: Structure-from-Motion
      :link: tutorial
      :link-type: doc

      Robust incremental SfM to recover camera poses and sparse 3D structure
      from ordered or unordered image collections.

   .. grid-item-card:: Multi-View Stereo
      :link: tutorial
      :link-type: doc

      Dense reconstruction with PatchMatch stereo and stereo fusion to produce
      detailed dense point clouds and meshes.

   .. grid-item-card:: Graphical & Command-Line
      :link: gui
      :link-type: doc

      A full-featured GUI for interactive reconstruction plus a scriptable
      command-line interface for automated pipelines.

   .. grid-item-card:: PyCOLMAP
      :link: pycolmap/index
      :link-type: doc

      Python bindings exposing most of COLMAP's functionality, from the
      reconstruction pipeline to robust geometric estimators.

   .. grid-item-card:: Camera Models & Rigs
      :link: cameras
      :link-type: doc

      A wide range of camera models and multi-camera rig support for diverse
      capture setups.

   .. grid-item-card:: Datasets & Formats
      :link: datasets
      :link-type: doc

      Ready-to-use sample datasets and well-documented input/output formats for
      easy integration.


Getting Started
---------------

1. Install COLMAP using the selector above, download the
   `pre-built binaries <https://github.com/colmap/colmap/releases>`_, or build
   from `source <https://colmap.github.io/install.html>`_
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


Citation
--------

If you use this project for your research, please cite::

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

If you use the global SfM pipeline (GLOMAP), please cite::

    @inproceedings{pan2024glomap,
        author={Pan, Linfei and Barath, Daniel and Pollefeys, Marc and Sch\"{o}nberger, Johannes Lutz},
        title={{Global Structure-from-Motion Revisited}},
        booktitle={European Conference on Computer Vision (ECCV)},
        year={2024},
    }

If you use the image retrieval / vocabulary tree engine, please cite::

    @inproceedings{schoenberger2016vote,
        author={Sch\"{o}nberger, Johannes Lutz and Price, True and Sattler, Torsten and Frahm, Jan-Michael and Pollefeys, Marc},
        title={A Vote-and-Verify Strategy for Fast Spatial Verification in Image Retrieval},
        booktitle={Asian Conference on Computer Vision (ACCV)},
        year={2016},
    }


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
   features
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
