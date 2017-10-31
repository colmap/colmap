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
import glob
import shutil
import fileinput
import platform
import argparse
import zipfile
import urllib.request
import subprocess
import multiprocessing


PLATFORM_IS_WINDOWS = platform.system() == "Windows"
PLATFORM_IS_LINUX = platform.system() == "Linux"
PLATFORM_IS_MAC = platform.system() == "Darwin"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build COLMAP and its dependencies locally under Windows, "
                    "Mac, and Linux. Note that under Mac and Linux, it is "
                    "usually easier and faster to use the available package "
                    "managers for the dependencies (see documentation). "
                    "However, if you are on a (cluster) system without root "
                    "access, this script might be useful. This script "
                    "downloads the necessary dependencies automatically from "
                    "the Internet. It assumes that CMake, Boost, Qt5, and CUDA "
                    "(optional) are already installed on the system. Under "
                    "Windows you must specify the location of these libraries.")
    parser.add_argument("--path", required=True)
    parser.add_argument("--qt_path", default="",
                        required=PLATFORM_IS_WINDOWS or PLATFORM_IS_MAC,
                        help="The path to the folder containing Qt, "
                             "e.g., under Windows: C:/Qt/5.9.1/msvc2013_64/ "
                             "or under Mac: /usr/local/opt/qt/")
    parser.add_argument("--boost_path", default="",
                        required=PLATFORM_IS_WINDOWS,
                        help="The path to the folder containing Boost, "
                             "e.g., under Windows: "
                             "C:/local/boost_1_64_0/lib64-msvc-12.0")
    parser.add_argument("--cuda_path", default="",
                        help="The path to the folder containing CUDA, "
                             "e.g., under Windows: C:/Program Files/NVIDIA GPU "
                             "Computing Toolkit/CUDA/v8.0")
    parser.add_argument("--cuda_multi_arch",
                        dest="cuda_multi_arch", action="store_true",)
    parser.add_argument("--no_cuda_multi_arch",
                        dest="cuda_multi_arch", action="store_false",
                        help="Whether to compile CUDA code for "
                             "multiple GPU architectures (default no)")
    parser.add_argument("--with_suite_sparse",
                        dest="with_suite_sparse", action="store_true")
    parser.add_argument("--without_suite_sparse",
                        dest="with_suite_sparse", action="store_false",
                        help="Whether to use SuiteSparse as a sparse solver "
                             "(default with SuiteSparse)")
    parser.add_argument("--colmap_branch", default="dev",
                        help="Which COLMAP branch to build")
    parser.add_argument("--colmap_update",
                        dest="colmap_update", action="store_true",
                        help="Whether to update the COLMAP code (default no)")
    parser.add_argument("--build_type", default="Release")
    parser.set_defaults(cuda_multi_arch=False)
    parser.set_defaults(with_suite_sparse=True)
    parser.set_defaults(colmap_update=False)
    args = parser.parse_args()

    args.path = os.path.abspath(args.path)
    args.download_path = os.path.join(args.path, "__download__")
    args.install_path = os.path.join(args.path, "__install__")

    args.cmake_config_args = []
    args.cmake_config_args.append(
        "-DCMAKE_BUILD_TYPE={}".format(args.build_type))
    args.cmake_config_args.append(
        "-DCMAKE_PREFIX_PATH={}".format(args.install_path))
    args.cmake_config_args.append(
        "-DCMAKE_INSTALL_PREFIX={}".format(args.install_path))
    if PLATFORM_IS_WINDOWS:
        args.cmake_config_args.append(
            "-DCMAKE_GENERATOR_PLATFORM=x64")

    args.cmake_build_args = ["--"]
    if PLATFORM_IS_WINDOWS:
        # Assuming that the build system is MSVC.
        args.cmake_build_args.append(
            "/maxcpucount:{}".format(multiprocessing.cpu_count()))
    else:
        # Assuming that the build system is Make.
        args.cmake_build_args.append(
            "-j{}".format(multiprocessing.cpu_count()))

    return args


def mkdir_if_not_exists(path):
    assert os.path.exists(os.path.dirname(os.path.abspath(path)))
    if not os.path.exists(path):
        os.makedirs(path)


def copy_file_if_not_exists(source, destination):
    if os.path.exists(destination):
        return
    shutil.copyfile(source, destination)


def download_zipfile(url, archive_path, unzip_path):
    if not os.path.exists(archive_path):
        urllib.request.urlretrieve(url, archive_path)
    with zipfile.ZipFile(archive_path, "r") as fid:
        fid.extractall(unzip_path)


def build_cmake_project(args, path, extra_config_args=[],
                        extra_build_args=[]):
    mkdir_if_not_exists(path)
    subprocess.call(["cmake"] + args.cmake_config_args
                     + extra_config_args + [".."], cwd=path)
    subprocess.call(["cmake",
                     "--build", ".",
                     "--target", "install",
                     "--config", args.build_type]
                     + args.cmake_build_args
                     + extra_build_args, cwd=path)


def build_eigen(args):
    path = os.path.join(args.path, "eigen")
    if os.path.exists(path):
        return

    url = "https://github.com/RLovelett/eigen/archive/3.3.4.zip"
    archive_path = os.path.join(args.download_path, "eigen.zip")
    download_zipfile(url, archive_path, args.path)
    shutil.move(os.path.join(args.path, "eigen-3.3.4"), path)

    build_cmake_project(args, os.path.join(path, "build"))


def build_freeimage(args):
    path = os.path.join(args.path, "freeimage")
    if os.path.exists(path):
        return

    if PLATFORM_IS_WINDOWS:
        url = "https://kent.dl.sourceforge.net/project/freeimage/" \
              "Binary%20Distribution/3.17.0/FreeImage3170Win32Win64.zip"
        archive_path = os.path.join(args.download_path, "freeimage.zip")
        download_zipfile(url, archive_path, args.path)
        shutil.move(os.path.join(args.path, "FreeImage"), path)
        copy_file_if_not_exists(
            os.path.join(path, "Dist/x64/FreeImage.h"),
            os.path.join(args.install_path, "include/FreeImage.h"))
        copy_file_if_not_exists(
            os.path.join(path, "Dist/x64/FreeImage.lib"),
            os.path.join(args.install_path, "lib/FreeImage.lib"))
        copy_file_if_not_exists(
            os.path.join(path, "Dist/x64/FreeImage.dll"),
            os.path.join(args.install_path, "bin/FreeImage.dll"))
    else:
        url = "https://kent.dl.sourceforge.net/project/freeimage/" \
              "Source%20Distribution/3.17.0/FreeImage3170.zip"
        archive_path = os.path.join(args.download_path, "freeimage.zip")
        download_zipfile(url, archive_path, args.path)
        shutil.move(os.path.join(args.path, "FreeImage"), path)

        if PLATFORM_IS_MAC:
            with fileinput.FileInput(os.path.join(path, "Makefile.gnu"),
                                     inplace=True, backup=".bak") as fid:
                for line in fid:
                    if "cp *.so Dist/" in line:
                        continue
                    if "FreeImage: $(STATICLIB) $(SHAREDLIB)" in line:
                        line = "FreeImage: $(STATICLIB)"
                    print(line, end="")
        elif PLATFORM_IS_LINUX:
            with fileinput.FileInput(
                    os.path.join(path, "Source/LibWebP/src/dsp/"
                                 "dsp.upsampling_mips_dsp_r2.c"),
                    inplace=True, backup=".bak") as fid:
                for i, line in enumerate(fid):
                    if i >= 36 and i <= 44:
                        line = line.replace("%[\"", "%[\" ")
                        line = line.replace("\"],", " \"],")
                    print(line, end="")
            with fileinput.FileInput(
                    os.path.join(path, "Source/LibWebP/src/dsp/"
                                 "dsp.yuv_mips_dsp_r2.c"),
                    inplace=True, backup=".bak") as fid:
                for i, line in enumerate(fid):
                    if i >= 56 and i <= 58:
                        line = line.replace("\"#", "\"# ")
                        line = line.replace("\"(%", " \"(%")
                    print(line, end="")

        subprocess.call(["make", "-f", "Makefile.gnu",
                         "-j{}".format(multiprocessing.cpu_count())], cwd=path)

        copy_file_if_not_exists(
            os.path.join(path, "Source/FreeImage.h"),
            os.path.join(args.install_path, "include/FreeImage.h"))
        copy_file_if_not_exists(
            os.path.join(path, "libfreeimage.a"),
            os.path.join(args.install_path, "lib/libfreeimage.a"))


def build_glew(args):
    path = os.path.join(args.path, "glew")
    if os.path.exists(path):
        return

    url = "https://kent.dl.sourceforge.net/project/glew/" \
          "glew/2.1.0/glew-2.1.0.zip"
    archive_path = os.path.join(args.download_path, "glew.zip")
    download_zipfile(url, archive_path, args.path)
    shutil.move(os.path.join(args.path, "glew-2.1.0"), path)

    build_cmake_project(args, os.path.join(path, "build/cmake/build"))


def build_gflags(args):
    path = os.path.join(args.path, "gflags")
    if os.path.exists(path):
        return

    url = "https://github.com/gflags/gflags/archive/v2.2.1.zip"
    archive_path = os.path.join(args.download_path, "gflags.zip")
    download_zipfile(url, archive_path, args.path)
    shutil.move(os.path.join(args.path, "gflags-2.2.1"), path)
    os.remove(os.path.join(path, "BUILD"))

    build_cmake_project(args, os.path.join(path, "build"))


def build_glog(args):
    path = os.path.join(args.path, "glog")
    if os.path.exists(path):
        return

    url = "https://github.com/google/glog/archive/v0.3.5.zip"
    archive_path = os.path.join(args.download_path, "glog.zip")
    download_zipfile(url, archive_path, args.path)
    shutil.move(os.path.join(args.path, "glog-0.3.5"), path)

    build_cmake_project(args, os.path.join(path, "build"))


def build_suite_sparse(args):
    if not args.with_suite_sparse:
        return

    path = os.path.join(args.path, "suite-sparse")
    if os.path.exists(path):
        return

    url = "https://codeload.github.com/jlblancoc/" \
          "suitesparse-metis-for-windows/zip/master"
    archive_path = os.path.join(args.download_path, "suite-sparse.zip")
    download_zipfile(url, archive_path, args.path)
    shutil.move(os.path.join(args.path,
                             "suitesparse-metis-for-windows-master"), path)

    build_cmake_project(args, os.path.join(path, "build"))

    if PLATFORM_IS_WINDOWS:
        lapack_blas_path = os.path.join(args.install_path,
                                        "lib64/lapack_blas_windows/*.dll")
        for library_path in glob.glob(lapack_blas_path):
            copy_file_if_not_exists(
                library_path, os.path.join(args.install_path, "bin",
                                           os.path.basename(library_path)))


def build_ceres_solver(args):
    path = os.path.join(args.path, "ceres-solver")
    if os.path.exists(path):
        return

    url = "https://github.com/ceres-solver/ceres-solver/archive/1.13.0.zip"
    archive_path = os.path.join(args.download_path, "ceres-solver.zip")
    download_zipfile(url, archive_path, args.path)
    shutil.move(os.path.join(args.path, "ceres-solver-1.13.0"), path)

    extra_config_args = [
        "-DBUILD_TESTING=OFF",
        "-DBUILD_EXAMPLES=OFF",
    ]

    if args.with_suite_sparse:
        extra_config_args.extend([
            "-DLAPACK=ON",
            "-DSUITESPARSE=ON",
        ])
        if PLATFORM_IS_WINDOWS:
            extra_config_args.extend([
                "-DLAPACK_LIBRARIES={}".format(
                    os.path.join(args.install_path,
                                 "lib64/lapack_blas_windows/liblapack.lib")),
                "-DBLAS_LIBRARIES={}".format(
                    os.path.join(args.install_path,
                                 "lib64/lapack_blas_windows/libblas.lib")),
            ])

    if PLATFORM_IS_WINDOWS:
        extra_config_args.append("-DCMAKE_CXX_FLAGS=/DGOOGLE_GLOG_DLL_DECL=")

    build_cmake_project(args, os.path.join(path, "build"),
                        extra_config_args=extra_config_args)


def build_colmap(args):
    path = os.path.join(args.path, "colmap-{}".format(args.colmap_branch))
    url = "https://codeload.github.com/" \
          "colmap/colmap/zip/{}".format(args.colmap_branch)
    archive_path = os.path.join(args.download_path,
                                "colmap-{}.zip".format(args.colmap_branch))
    if args.colmap_update:
        if os.path.exists(archive_path):
            os.remove(archive_path)
        download_zipfile(url, archive_path, args.path)
    elif not os.path.exists(path):
        download_zipfile(url, archive_path, args.path)

    extra_config_args = []
    if args.qt_path != "":
        extra_config_args.append("-DQt5Core_DIR={}".format(
            os.path.join(args.qt_path, "lib/cmake/Qt5Core")))
        extra_config_args.append("-DQt5OpenGL_DIR={}".format(
            os.path.join(args.qt_path, "lib/cmake/Qt5OpenGL")))

    if args.boost_path != "":
        extra_config_args.append(
            "-DBOOST_ROOT={}".format(args.boost_path))
        extra_config_args.append(
            "-DBOOST_LIBRARYDIR={}".format(args.boost_path))

    if args.cuda_path != "":
        extra_config_args.append(
            "-DCUDA_TOOLKIT_ROOT_DIR={}".format(args.cuda_path))

    if args.cuda_multi_arch:
        extra_config_args.append("-DCUDA_MULTI_ARCH=ON")
    else:
        extra_config_args.append("-DCUDA_MULTI_ARCH=OFF")

    if PLATFORM_IS_WINDOWS:
        extra_config_args.append("-DCMAKE_CXX_FLAGS=/DGOOGLE_GLOG_DLL_DECL=")

    build_cmake_project(args, os.path.join(path, "build"),
                        extra_config_args=extra_config_args)


def build_post_process(args):
    if PLATFORM_IS_WINDOWS:
        if args.qt_path:
            copy_file_if_not_exists(
                os.path.join(args.qt_path, "bin/Qt5Core.dll"),
                os.path.join(args.install_path, "bin/Qt5Core.dll"))
            copy_file_if_not_exists(
                os.path.join(args.qt_path, "bin/Qt5Gui.dll"),
                os.path.join(args.install_path, "bin/Qt5Gui.dll"))
            copy_file_if_not_exists(
                os.path.join(args.qt_path, "bin/Qt5Widgets.dll"),
                os.path.join(args.install_path, "bin/Qt5Widgets.dll"))


def main():
    args = parse_args()

    mkdir_if_not_exists(args.path)
    mkdir_if_not_exists(args.download_path)
    mkdir_if_not_exists(args.install_path)
    mkdir_if_not_exists(os.path.join(args.install_path, "include"))
    mkdir_if_not_exists(os.path.join(args.install_path, "bin"))
    mkdir_if_not_exists(os.path.join(args.install_path, "lib"))
    mkdir_if_not_exists(os.path.join(args.install_path, "share"))

    build_eigen(args)
    build_freeimage(args)
    build_glew(args)
    build_gflags(args)
    build_glog(args)
    build_suite_sparse(args)
    build_ceres_solver(args)
    build_colmap(args)
    build_post_process(args)

    print()
    print()
    print("Successfully installed COLMAP in: {}".format(args.install_path))
    if PLATFORM_IS_WINDOWS:
        print("  To run COLMAP, navigate to {} and run colmap.exe".format(
                    os.path.join(args.install_path, "bin")))
    else:
        print("  To run COLMAP, execute LD_LIBRARY_PATH={} {}".format(
                    os.path.join(args.install_path, "lib"),
                    os.path.join(args.install_path, "bin/colmap")))


if __name__ == "__main__":
    main()
