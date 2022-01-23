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

# Reads the output CSV file of the model_comparer and checks the errors.

import sys
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--errors_csv_path", required=True)
    parser.add_argument("--max_rotation_error",
                        type=float, default=float("inf"))
    parser.add_argument("--max_translation_error",
                        type=float, default=float("inf"))
    parser.add_argument("--max_proj_center_error",
                        type=float, default=float("inf"))
    parser.add_argument("--expected_num_images", type=int, default=-1)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    error = False
    with open(args.errors_csv_path, "r") as fid:
        num_images = 0
        for line in fid:
            line = line.strip()
            if len(line) == 0 or line.startswith("#"):
                continue
            rotation_error, translation_error, proj_center_error = \
                map(float, line.split(","))
            num_images += 1
            if rotation_error > args.max_rotation_error:
                print("Exceeded rotation error threshold:", rotation_error)
                error = True
            if translation_error > args.max_translation_error:
                print("Exceeded translation error threshold:",
                      translation_error)
                error = True
            if proj_center_error > args.max_proj_center_error:
                print("Exceeded projection error threshold:",
                      proj_center_error)
                error = True

    if args.expected_num_images >= 0 and num_images != args.expected_num_images:
        print("Unexpected number of images:", num_images)
        error = True

    if error:
        sys.exit(1)


if __name__ == "__main__":
    main()
