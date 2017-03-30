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

function [xyz, normals, rgb] = read_ply(path)
% Read point cloud from PLY text file.

file = fopen(path, 'r');
type = fscanf(file, '%s', 1);
format = fscanf(file, '%s', 3);
data = fscanf(file, '%s', 2);
num_points = fscanf(file, '%d', 1);
fscanf(file, '%s', 3);
fscanf(file, '%s', 3);
fscanf(file, '%s', 3);
fscanf(file, '%s', 3);
fscanf(file, '%s', 3);
fscanf(file, '%s', 3);
fscanf(file, '%s', 3);
fscanf(file, '%s', 3);
fscanf(file, '%s', 3);
fscanf(file, '%s', 1);
points_data = textscan(file, '%f %f %f %f %f %f %f %f %f', num_points);
xyz = [points_data{1}, points_data{2}, points_data{3}];
rgb = [points_data{7}, points_data{8}, points_data{9}];
normals = [points_data{4}, points_data{5}, points_data{6}];

end
