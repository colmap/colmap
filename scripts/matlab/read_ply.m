% SPDX-License-Identifier: BSD-3-Clause

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
