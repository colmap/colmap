% SPDX-License-Identifier: BSD-3-Clause

function [normal_map, normal_map_rgb] = read_normal_map(path, varargin)

normal_map = read_array(path);

normal_map_rgb = -[normal_map(:,:,1) normal_map(:,:,2) ...
                   2 * normal_map(:,:,3) + 1];
normal_map_rgb = reshape(normal_map_rgb, ...
                         size(normal_map,1), size(normal_map,2), 3);
normal_map_rgb = uint8(((normal_map_rgb + 1) ./ 2) .* 255);

if length(varargin) == 1
    depth_map = varargin{1};
    normal_map_rgb(repmat(isnan(depth_map), [1, 1, 3])) = 0;
end

end
