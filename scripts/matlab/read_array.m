% SPDX-License-Identifier: BSD-3-Clause

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
