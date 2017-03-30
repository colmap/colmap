% COLMAP - Structure-from-Motion and Multi-View Stereo.
% Copyright (C) 2017  Johannes L. Schoenberger <jsch at inf.ethz.ch>
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.

function [depth_map, depth_map_rgb] = read_depth_map(path)

depth_map = read_array(path);

depth_range = prctile(depth_map(depth_map(:) > 0), [2, 98]);

depth_map(depth_map<=0) = nan;
depth_map_rgb = cmap2rgb(depth_map, [0 0 0; jet(2^15)], depth_range);

end
