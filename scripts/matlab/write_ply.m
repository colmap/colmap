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

function write_ply(path, xyz, normals, rgb)
% Write point cloud to PLY text file.

file = fopen(path, 'W');

fprintf(file,'ply\n');
fprintf(file,'format ascii 1.0\n');
fprintf(file,'element vertex %d\n',size(xyz,1));
fprintf(file,'property float x\n');
fprintf(file,'property float y\n');
fprintf(file,'property float z\n');
fprintf(file,'property float nx\n');
fprintf(file,'property float ny\n');
fprintf(file,'property float nz\n');
fprintf(file,'property uchar diffuse_red\n');
fprintf(file,'property uchar diffuse_green\n');
fprintf(file,'property uchar diffuse_blue\n');
fprintf(file,'end_header\n');

for i = 1:size(xyz, 1)
    fprintf(file, '%f %f %f %f %f %f %d %d %d\n', ...
        xyz(i,1), xyz(i,2), xyz(i,3), ...
        normals(i,1), normals(i,2), normals(i,3), ...
        uint8(rgb(i,1)), uint8(rgb(i,2)), uint8(rgb(i,3)));
end

fclose(file);

end
