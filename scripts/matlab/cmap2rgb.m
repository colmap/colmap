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
