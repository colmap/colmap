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

function plot_model(cameras, images, points)
% Visualize COLMAP model.

keys = images.keys;
camera_centers = zeros(images.length, 3);
view_dirs = zeros(3 * images.length, 3);
for i = 1:images.length
    image_id = keys{i};
    image = images(image_id);
    camera_centers(i,:) = -image.R' * image.t;
    view_dirs(3 * i - 2,:) = camera_centers(i,:);
    view_dirs(3 * i - 1,:) = camera_centers(i,:)' + image.R' * [0; 0; 0.3];
    view_dirs(3 * i,:) = nan;
end

keys = points.keys;
xyz = zeros(points.length, 3);
for i = 1:points.length
    point_id = keys{i};
    point = points(point_id);
    xyz(i,:) = point.xyz;
end

hold on;
plot3(camera_centers(:,1), camera_centers(:,2), camera_centers(:,3), 'xr');
plot3(view_dirs(:,1), view_dirs(:,2), view_dirs(:,3), '-b');
plot3(xyz(:,1), xyz(:,2), xyz(:,3), '.k');
hold off;

end
