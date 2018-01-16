.. _gui:

Graphical User Interface
========================

The graphical user interface of COLMAP provides access to most of the available
functionality and visualizes the reconstruction process in "real-time". To start
the GUI, you can run the pre-built packages (Windows: `COLMAP.bat`, Mac:
`COLMAP.app`), execute ``colmap gui`` if you installed COLMAP or execute
``./src/exe/colmap gui`` from the CMake build folder. The GUI application
requires an attached display with at least OpenGL 3.2 support. Registered images
are visualized in red and reconstructed points in their average point color
extracted from the images. The viewer can also visualize dense point clouds
produced from Multi-View Stereo.


Model Viewer Controls
---------------------

- **Rotate model**: Left-click and drag.
- **Shift model**: Right-click or <CTRL>-click (<CMD>-click) and drag.
- **Zoom model**: Scroll.
- **Change point size**: <CTRL>-scroll (<CMD>-scroll).
- **Change camera size**: <ALT>-scroll.
- **Adjust clipping plane**: <SHIFT>-scroll.
- **Select point**: Double-left-click point (change point size if too small).
  The green lines visualize the projections into the images that see the point.
  The opening window shows the projected locations of the point in all images.
- **Select camera**: Double-left-click camera (change camera size if too small).
  The purple lines visualize images that see at least one common point with the
  selected image. The opening window shows a few statistics of the image.
- **Reset view**: To reset all viewing settings, choose ``Render > Reset view``.


Render Options
--------------

The model viewer allows you to render the model with different settings,
projections, colormaps, etc. Please, choose ``Render > Render options``.


Create Screenshots
------------------

To create screenshots of the current viewpoint (without coordinate axes), choose
``Extras > Grab image`` and save the image in the format of your choice.


Create Screencast
-----------------

To create a video screen capture of the reconstructed model, choose ``Extras >
Grab movie``. This dialog allows you to set individual control viewpoints by
choosing ``Add``. COLMAP generates a fixed number of frames per second between
each control viewpoint by smoothly interpolating the linear trajectory, and to
interpolate the configured point and the camera sizes at the time of clicking
``Add``. To change the number of frames between two viewpoints or to reorder
individual viewpoints, modify the time of the viewpoint by double-clicking the
respective cell in the table. Note that the video capture requires to set the
perspective projection model in the render options. You can review the
trajectory in the viewer, which is rendered in light blue. Choose ``Assemble
movie``, if you are done creating the trajectory. The output directory then
contains the individual frames of the video capture, which can be assembled to a
movie using `FFMPEG <https://www.ffmpeg.org/>`_ with the following command::

    ffmpeg -i frame%06d.png -r 30 -vf scale=1680:1050 movie.mp4
