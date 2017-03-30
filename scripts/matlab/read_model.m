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

function [cameras, images, points3D] = read_model(path)
% Read COLMAP model from folder, which contains a
% cameras.txt, images.txt, and points3D.txt.

if numel(path) > 0 && path(end) ~= '/'
    path = [path '/'];
end

cameras = read_cameras([path 'cameras.txt']);
images = read_images([path 'images.txt']);
points3D = read_points3D([path 'points3D.txt']);

end

function cameras = read_cameras(path)

cameras = containers.Map('KeyType', 'int64', 'ValueType', 'any');

fid = fopen(path);
tline = fgets(fid);
while ischar(tline)
    elems = strsplit(tline);
    if numel(elems) < 4 || strcmp(elems(1), '#')
        tline = fgets(fid);
        continue
    end

    if mod(cameras.Count, 10) == 0
        fprintf('Reading camera %d\n', cameras.length);
    end

    camera = struct;
    camera.camera_id = str2num(elems{1});
    camera.model = elems{2};
    camera.width = str2num(elems{3});
    camera.height = str2num(elems{4});

    camera.params = zeros(numel(elems) - 5, 1);
    for i = 5:numel(elems) - 1
        camera.params(i - 4) = str2double(elems{i});
    end

    cameras(camera.camera_id) = camera;

    tline = fgets(fid);
end

fclose(fid);

end

function images = read_images(path)

images = containers.Map('KeyType', 'int64', 'ValueType', 'any');

fid = fopen(path);
tline = fgets(fid);
while ischar(tline)
    elems = strsplit(tline);
    if numel(elems) < 4 || strcmp(elems(1), '#')
        tline = fgets(fid);
        continue
    end

    if mod(images.Count, 10) == 0
        fprintf('Reading image %d\n', images.length);
    end

    image = struct;
    image.image_id = str2num(elems{1});
    qw = str2double(elems{2});
    qx = str2double(elems{3});
    qy = str2double(elems{4});
    qz = str2double(elems{5});
    image.R = quat2rotmat([qw, qx, qy, qz]);
    tx = str2double(elems{6});
    ty = str2double(elems{7});
    tz = str2double(elems{8});
    image.t = [tx; ty; tz];
    image.camera_id = str2num(elems{9});
    image.name = elems{10};

    tline = fgets(fid);
    elems = sscanf(tline, '%f');
    elems = reshape(elems, [3, numel(elems) / 3]);
    image.xys = elems(1:2,:)';
    image.point3D_ids = elems(3,:)';

    images(image.image_id) = image;

    tline = fgets(fid);
end

fclose(fid);

end

function points3D = read_points3D(path)

points3D = containers.Map('KeyType', 'int64', 'ValueType', 'any');

fid = fopen(path);
tline = fgets(fid);
while ischar(tline)
    if numel(tline) == 0 || strcmp(tline(1), '#')
        tline = fgets(fid);
        continue;
    end

    elems = sscanf(tline, '%f');
    if numel(elems) == 0
        tline = fgets(fid);
        continue;
    end

    if mod(points3D.Count, 1000) == 0
        fprintf('Reading point %d\n', points3D.length);
    end

    point = struct;
    point.point3D_id = int64(elems(1));
    point.xyz = elems(2:4);
    point.rgb = int8(elems(5:7));
    point.error = elems(8);
    point.track = int64(elems(9:end));
    point.track = reshape(point.track, [2, numel(point.track) / 2])';
    point.track(:,2) = point.track(:,2) + 1;

    points3D(point.point3D_id) = point;

    tline = fgets(fid);
end

fclose(fid);

end
