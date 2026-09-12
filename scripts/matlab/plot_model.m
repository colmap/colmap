% SPDX-License-Identifier: BSD-3-Clause

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
