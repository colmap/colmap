# COLMAP - Structure-from-Motion and Multi-View Stereo.
# Copyright (C) 2017  Johannes L. Schoenberger <jsch at inf.ethz.ch>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import os
import string
import argparse
import subprocess


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    parser.add_argument("--exts", default=".h,.cc")
    parser.add_argument("--style", default="File")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    exts = map(string.lower, args.exts.split(","))

    for root, subdirs, files in os.walk(args.path):
        for f in files:
            name, ext = os.path.splitext(f)
            if ext.lower() in exts:
                file_path = os.path.join(root, f)
                proc = subprocess.Popen(["clang-format", "--style",
                                         args.style, file_path],
                                        stdout=subprocess.PIPE)

                text = "".join(proc.stdout)

                with open(file_path, "w") as fd:
                    fd.write(text)


if __name__ == "__main__":
    main()
