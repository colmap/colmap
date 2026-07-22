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

   <link rel="stylesheet" href="_static/viewer/viewer.css">
   <div id="colmap-viewer-root">
     <p>The interactive viewer is unavailable in this documentation build.</p>
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


Reusing the Viewer
------------------

The viewer is also an ES module component. Build it with ``npm run build`` in
the ``doc`` directory, copy the complete ``_static/viewer`` directory so that
the parser worker remains beside the module, and include both generated assets:

.. code-block:: html

   <link rel="stylesheet" href="/viewer/viewer.css">
   <div id="my-viewer"></div>
   <input id="model-folder" type="file" webkitdirectory multiple>
   <script type="module">
     import {mountColmapViewer} from "/viewer/component.js";

     const viewer = mountColmapViewer(document.querySelector("#my-viewer"));
     document.querySelector("#model-folder").addEventListener("change", async event => {
       const entries = [...event.target.files].map(file => ({
         path: file.webkitRelativePath || file.name,
         file,
       }));
       await viewer.load(entries);
     });

     // viewer.load() also accepts an already parsed Reconstruction object.
     // Call viewer.clear() or viewer.dispose() when appropriate.
   </script>

Each ``LocalFile`` entry has the shape ``{path, file}``, where ``path`` is the
file's relative path and ``file`` is a browser ``File`` object. Multiple
component instances can coexist on one page. The TypeScript source also exports
the mount settings and lifecycle types.

When embedding the component on another website, please include visible
attribution to the `COLMAP project <https://colmap.github.io/>`_ and reproduce
the full :doc:`COLMAP new BSD license notice <license>` in the website's legal
or third-party notices. Also retain the bundled
`Three.js license notice <_static/viewer-licenses.txt>`_ for that dependency.
