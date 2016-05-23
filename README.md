COLMAP
======

COLMAP is a general-purpose Structure-from-Motion (SfM) pipeline with a
graphical and command-line interface. It offers a wide range of features for
reconstruction of ordered and unordered image collections. The software is
licensed under the GNU General Public License. If you use this project for your
research, please cite:

    @inproceedings{schoenberger2016sfm,
        author = {Sch\"{o}nberger, Johannes Lutz and Frahm, Jan-Michael},
        title = {Structure-from-Motion Revisited},
        booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
        year={2016},
    }

The source code is available at https://github.com/colmap/colmap. COLMAP builds
on top of existing work and when using specific algorithms within COLMAP, please
also consider citing the original works, as specified in the source code.


Modifications:
======

Add support for central camera such as Spherical Camera. The key is to change the world coordinates to 3D representation, that is Vector2D -> Vector3D. There exist two cases: [x, y, 1], or [x, y, z] where x^2+y^2+z^2 = 1.

For spherical cameras, focal length is set to the radius pixels and the princple point could be set in the center of the image which must lies on the equator of the sphere in the local camera cooridnate system.

The example photo is as follows:
![image](/doc/R0010160.jpg)

and the reconstruction result is:
![image](/doc/spherical_recon.jpg)
![image](/doc/spherical_recon2.jpg)

Getting Started
---------------

1. Download the pre-built binaries from
   http://people.inf.ethz.ch/jschoenb/colmap/ or build the library manually as
   described in the documentation.
2. Download one of the provided datasets at
   http://people.inf.ethz.ch/jschoenb/colmap/datasets/.
3. Watch the short introductory video at
   https://www.youtube.com/watch?v=P-EC0DzeVEU or read the tutorial
   in the documentation for more details.


Documentation
-------------

The documentation is available at https://colmap.github.io/.


Support
-------

Please, use the GitHub issue tracker at https://github.com/colmap/colmap for bug
reports, questions, suggestions, etc.


Acknowledgments
---------------

The library was written by Johannes L. Schönberger
(http://people.inf.ethz.ch/jschoenb/). Funding was provided by his PhD advisor
Jan-Michael Frahm (http://frahm.web.unc.edu/) through the grants NSF No.
IIS-1349074, No. CNS-1405847, and the MITRE Corp.


Contribution
------------

Contributions (bug reports, bug fixes, improvements, etc.) are very welcome and
should be submitted in the form of new issues and/or pull requests on GitHub.

Please, adhere to the Google coding style guide:

    http://google-styleguide.googlecode.com/svn/trunk/cppguide.html

by using the provided ".clang-format" file.

Document functions, methods, classes, etc. with inline documentation strings
describing the API, using the following format:

    // Short description.
    //
    // Longer description with a few sentences and multiple lines.
    //
    // @param parameter1            Description for parameter 1.
    // @param parameter2            Description for parameter 2.
    //
    // @return                      Description of optional return value.

Add unit tests for all newly added code and make sure that algorithmic
"improvements" generalize and actually improve the results of the pipeline on a
variety of datasets.


License
-------

The software is licensed under the GNU General Public License v3 or later. If
you are interested in licensing the software for commercial purposes, without
disclosing your modifications, please contact the authors.

    COLMAP - Structure-from-Motion.
    Copyright (C) 2016  Johannes L. Schoenberger <jsch at cs.unc.edu>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
