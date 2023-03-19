docker pull colmap/colmap:latest
docker run \
    -e QT_XCB_GL_INTEGRATION=xcb_egl \
    -e DISPLAY=:0 \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    --gpus all \
    --privileged \
    -it colmap/colmap:latest \
    colmap gui
