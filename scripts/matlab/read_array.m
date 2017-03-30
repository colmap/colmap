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

function array = read_array(path, varargin)

if length(varargin) == 1
    dtype = varargin{1};
else
    dtype = 'single';
end

fid = fopen(path);

line = fscanf(fid, '%d&%d&%d&', [1, 3]);
width = line(1);
height = line(2);
channels = line(3);

num = width * height * channels;

array = fread(fid, num, dtype);
array = reshape(array, [width, height, channels]);
array = permute(array, [2 1 3]);
array = cast(array, dtype);

fclose(fid);

end
