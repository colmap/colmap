:html_theme.sidebar_secondary.remove: true
:og:description: Inspect COLMAP sparse reconstructions directly in your browser with an interactive, private, local-only 3D viewer.

.. meta::
   :description: Inspect binary COLMAP sparse reconstructions directly in your browser with an interactive, private, local-only Three.js viewer.

3D Viewer
=========

Open a binary COLMAP sparse reconstruction directly in your browser. Processing
happens locally: reconstruction files and images are never uploaded.

The 3D Viewer follows the visual conventions and model-view controls of the
native :ref:`Graphical User Interface <gui>`, including navigation, point and
image selection, observation inspection, and point/camera scaling. It is a
read-only inspection tool with intentionally less functionality than the native
GUI: it does not run reconstruction pipelines, edit models, or visualize dense
point clouds and meshes.

.. raw:: html

   <div id="colmap-viewer-root">
     <noscript>This viewer requires JavaScript.</noscript>
   </div>
   <script type="module" src="_static/viewer/viewer.js"></script>


Supported Inputs
----------------

Drop a sparse model folder containing ``cameras.bin``, ``images.bin``, and
``points3D.bin``, or drop a workspace containing one or more sparse models.
Current reconstructions with ``rigs.bin`` and ``frames.bin`` and legacy binary
reconstructions are supported. If several models are found, choose one from the
toolbar.

To inspect image observations and reprojections, drop a workspace that also
contains the source image tree. You can also drop the images folder after the
model has loaded. Image paths are matched against the relative names recorded
in ``images.bin``.


Controls
--------

- **Rotate:** left-click and drag.
- **Pan:** right-click and drag.
- **Zoom:** scroll.
- **Point size:** <CTRL>-scroll (<CMD>-scroll) or use the toolbar.
- **Camera size:** <ALT>-scroll or use the toolbar.
- **Select:** double-click a point or camera.
- **Clear selection:** double-click the background or use the toolbar.

The viewer requires a current desktop browser with WebGL2. Folder selection is
available as a fallback when directory drag-and-drop is not supported by the
browser.

The viewer uses `Three.js <_static/viewer-licenses.txt>`_ under the MIT license.
