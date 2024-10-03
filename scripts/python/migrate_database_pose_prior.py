# Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
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


import argparse

import numpy as np
from database import COLMAPDatabase

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--database_path", type=str, required=True)
    parser.add_argument("--is_cartesian", action="store_true")
    parser.add_argument("--cleanup", action="store_true")
    args = parser.parse_args()

    db = COLMAPDatabase.connect(args.database_path)

    pose_priors = {}
    rows = db.execute("SELECT * FROM images")
    for image_id, _, _, *cam_from_world_prior in rows:
        if not cam_from_world_prior:  # newer format database
            continue
        qvec = np.array(cam_from_world_prior[:4], dtype=float)
        tvec = np.array(cam_from_world_prior[4:], dtype=float)
        if np.isfinite(qvec).any():
            print(
                f"Warning: rotation prior for image {image_id} "
                "will be lost during migration."
            )
        if np.isfinite(tvec).any():
            pose_priors[image_id] = tvec
    print(f"Found location priors for {len(pose_priors)} images.")

    coordinate_systems = {"UNKNOWN": -1, "WGS84": 0, "CARTESIAN": 1}
    coordinate_system = coordinate_systems[
        "CARTESIAN" if args.is_cartesian else "WGS84"
    ]
    db.create_pose_priors_table()
    for image_id, position in pose_priors.items():
        (exists,) = db.execute(
            "SELECT COUNT(*) FROM pose_priors WHERE image_id = ?",
            (image_id,),
        ).fetchone()
        if exists:
            print(f"Location prior for {image_id} already exists, skipping.")
            continue
        db.add_pose_prior(image_id, position, coordinate_system)

    if args.cleanup:
        for col in ["qw", "qx", "qy", "qz", "tx", "ty", "tz"]:
            db.execute(f"ALTER TABLE images DROP COLUMN prior_{col}")

    db.commit()
