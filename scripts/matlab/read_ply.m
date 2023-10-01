% Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
% All rights reserved.
%
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are met:
%
%     * Redistributions of source code must retain the above copyright
%       notice, this list of conditions and the following disclaimer.
%
%     * Redistributions in binary form must reproduce the above copyright
%       notice, this list of conditions and the following disclaimer in the
%       documentation and/or other materials provided with the distribution.
%
%     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
%       its contributors may be used to endorse or promote products derived
%       from this software without specific prior written permission.
%
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
% ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
% LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
% CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
% SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
% INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
% CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
% ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
% POSSIBILITY OF SUCH DAMAGE.


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
