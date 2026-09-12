# SPDX-License-Identifier: BSD-3-Clause

# Command to produce video from images produced by COLMAP movie grabber tool.

ffmpeg -i frame%06d.png -r 30 -vf scale=1680:1050 out.mp4
