% SPDX-License-Identifier: BSD-3-Clause

function [depth_map, depth_map_rgb] = read_depth_map(path)

depth_map = read_array(path);

depth_range = prctile(depth_map(depth_map(:) > 0), [2, 98]);

depth_map(depth_map<=0) = nan;
depth_map_rgb = cmap2rgb(depth_map, [0 0 0; jet(2^15)], depth_range);

end
