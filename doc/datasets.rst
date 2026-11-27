.. _datasets:

Datasets
========

The following datasets are available for download:

- **South Building**: 128 images of the "South" building at UNC Chapel Hill. The
  images are taken with the same camera, kindly provided by Christopher Zach.

  - `south-building.zip <https://github.com/colmap/colmap/releases/download/3.11.1/south-building.zip>`__

- **Gerrard Hall**: 100 high-resolution images of the "Gerrard" hall at UNC
  Chapel Hill, which is the building right next to the "South" building.
  The images are taken with the same camera but different focus
  using a wide-angle lens.

  - `gerrard-hall.zip <https://github.com/colmap/colmap/releases/download/3.11.1/gerrard-hall.zip>`__

- **Person Hall**: 330 high-resolution images of the "Person" hall at UNC Chapel
  Hill. The images are taken with the same camera using a wide-angle lens.

  - `person-hall.zip <https://github.com/colmap/colmap/releases/download/3.11.1/person-hall.zip>`__
  - `person-hall.z01 <https://github.com/colmap/colmap/releases/download/3.11.1/person-hall.z01>`__

- **Graham Hall**: 1273 high-resolution images of the interior and exterior of
  "Graham" memorial hall at UNC Chapel Hill. The images are taken with the same
  camera but different focus using a wide-angle lens.

  - `graham-hall.zip <https://github.com/colmap/colmap/releases/download/3.11.1/graham-hall.zip>`__
  - `graham-hall.z01 <https://github.com/colmap/colmap/releases/download/3.11.1/graham-hall.z01>`__
  - `graham-hall.z02 <https://github.com/colmap/colmap/releases/download/3.11.1/graham-hall.z02>`__
  - `graham-hall.z03 <https://github.com/colmap/colmap/releases/download/3.11.1/graham-hall.z03>`__
  - `graham-hall.z04 <https://github.com/colmap/colmap/releases/download/3.11.1/graham-hall.z04>`__
  - `graham-hall.z05 <https://github.com/colmap/colmap/releases/download/3.11.1/graham-hall.z05>`__

Person Hall and Graham Hall are split into multi-part archives. Download all
parts into the same directory and extract them using 7-Zip or, for example::

    zip -FF person-hall.zip --out person-hall-combined.zip
    unzip person-hall-combined.zip

A number of sample reconstructions produced by COLMAP can be viewed here:

**Sparse reconstructions**:

- https://youtu.be/PmXqdfBQxfQ
- https://youtu.be/DIv1aGKqSIk

**Dense reconstructions**:

- https://youtu.be/11awtGWSqQU
