% SPDX-License-Identifier: BSD-3-Clause

function write_array(path, array)

fid = fopen(path, 'w');
fprintf(fid, '%d&%d&%d&', size(array, 2), size(array, 1), size(array, 3));
array = permute(array, [2 1 3]);
fwrite(fid, array, class(array));
fclose(fid);

end
