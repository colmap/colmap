% SPDX-License-Identifier: BSD-3-Clause

function rgb = cmap2rgb(image, cmap, varargin)

if length(varargin) == 1
    clim = varargin{1};
    image(image <= clim(1)) = clim(1);
    image(image > clim(2)) = clim(2);
end

image_min = min(image(:));
image_max = max(image(:));
image = (image - image_min) / (image_max - image_min) * size(cmap, 1);

rgb = ind2rgb(uint32(image), cmap);

end
