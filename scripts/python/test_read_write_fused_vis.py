# Copyright (c), ETH Zurich and UNC Chapel Hill.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
#       its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.


import filecmp

from read_write_fused_vis import read_fused, write_fused


def main():
    import sys

    if len(sys.argv) != 5:
        print(
            "Usage: python test_read_write_fused_vis.py "
            "path/to/input_fused.ply path/to/input_fused.ply.vis "
            "path/to/output_fused.ply path/to/output_fused.ply.vis"
        )
        return

    print(
        "Checking consistency of reading and writing fused.ply and fused.ply.vis files ..."
    )

    path_to_fused_ply_input = sys.argv[1]
    path_to_fused_ply_vis_input = sys.argv[2]
    path_to_fused_ply_output = sys.argv[3]
    path_to_fused_ply_vis_output = sys.argv[4]

    mesh_points = read_fused(
        path_to_fused_ply_input, path_to_fused_ply_vis_input
    )
    write_fused(
        mesh_points, path_to_fused_ply_output, path_to_fused_ply_vis_output
    )

    assert filecmp.cmp(path_to_fused_ply_input, path_to_fused_ply_output)
    assert filecmp.cmp(
        path_to_fused_ply_vis_input, path_to_fused_ply_vis_output
    )

    print("... Results are equal.")


if __name__ == "__main__":
    main()
