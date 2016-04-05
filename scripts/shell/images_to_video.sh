# COLMAP - Structure-from-Motion.
# Copyright (C) 2016  Johannes L. Schoenberger <jsch at cs.unc.edu>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# Command to produce video from images produced by COLMAP movie grabber tool.

ffmpeg -i frame%06d.png -r 30 -vf scale=1680:1050 out.mp4
