# Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
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
#
# Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

import os
import glob
import shutil
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--install_path", required=True,
                        help="The installation prefix, e.g., build/__install__")
    parser.add_argument("--app_path", required=True,
                        help="The application path, e.g., "
                             "build/COLMAP-dev-windows")
    args = parser.parse_args()
    return args


def mkdir_if_not_exists(path):
    assert os.path.exists(os.path.dirname(os.path.abspath(path)))
    if not os.path.exists(path):
        os.makedirs(path)


def main():
    args = parse_args()

    mkdir_if_not_exists(args.app_path)
    mkdir_if_not_exists(os.path.join(args.app_path, "bin"))
    mkdir_if_not_exists(os.path.join(args.app_path, "lib"))
    mkdir_if_not_exists(os.path.join(args.app_path, "lib/platforms"))

    # Copy batch scripts to app directory.
    shutil.copyfile(
        os.path.join(args.install_path, "COLMAP.bat"),
        os.path.join(args.app_path, "COLMAP.bat"))
    shutil.copyfile(
        os.path.join(args.install_path, "RUN_TESTS.bat"),
        os.path.join(args.app_path, "RUN_TESTS.bat"))

    # Copy executables to app directory.
    exe_files = glob.glob(os.path.join(args.install_path, "bin/*.exe"))
    for exe_file in exe_files:
        shutil.copyfile(exe_file, os.path.join(args.app_path, "bin",
                                               os.path.basename(exe_file)))

    # Copy shared libraries to app directory.
    dll_files = glob.glob(os.path.join(args.install_path, "lib/*.dll"))
    for dll_file in dll_files:
        shutil.copyfile(dll_file, os.path.join(args.app_path, "lib",
                                               os.path.basename(dll_file)))
    shutil.copyfile(
        os.path.join(args.install_path, "lib/platforms/qwindows.dll"),
        os.path.join(args.app_path, "lib/platforms/qwindows.dll"))

    # Create zip archive for deployment.
    shutil.make_archive(args.app_path, "zip", root_dir=args.app_path)


if __name__ == "__main__":
    main()
