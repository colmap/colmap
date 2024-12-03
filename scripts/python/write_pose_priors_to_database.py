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

from database import COLMAPDatabase, array_to_blob

import numpy as np

def update_pose_prior_from_image_name(
    colmap_db: COLMAPDatabase,
    image_name: str,
    position: np.array,
    coordinate_system: int = -1,
    position_covariance: np.array = None,
):
    """
    Update the pose prior for a specific image name in the database.
    If the pose prior doesn't exist, insert a new one.

    Args:
        colmap_db (COLMAPDatabase): colmap database to update.
        image_name (str): name of the image to update.
        position (np.array): Position as a 3-element array (x, y, z).
        coordinate_system (int): Coordinate system index (default: -1).
        position_covariance (np.array): 3x3 position covariance matrix (default: None).
    """
    # Get image_id from image_name
    cursor = colmap_db.execute(
        "SELECT image_id FROM images WHERE name = ?", (image_name,)
    )
    result = cursor.fetchone()
    if result is None:
        print(
            "ERROR: could not retrieve image {0} in database.".format(
                image_name
            )
        )
        return
    image_id = result[0]
    position = np.asarray(position, dtype=np.float64)
    if position_covariance is None:
        position_covariance = np.full((3, 3), np.nan, dtype=np.float64)

    # Check if the pose prior already exists
    cursor = colmap_db.execute("SELECT COUNT(*) FROM pose_priors WHERE image_id = ?", (image_id,))
    exists = cursor.fetchone()[0] > 0

    if exists:
        # Update the existing pose prior
        colmap_db.execute(
            """
            UPDATE pose_priors
            SET position = ?, coordinate_system = ?, position_covariance = ?
            WHERE image_id = ?
            """,
            (
                array_to_blob(position),
                coordinate_system,
                array_to_blob(position_covariance),
                image_id,
            ),
        )
    else:
        # Add a new one
        colmap_db.add_pose_prior(
            image_id, position, coordinate_system, position_covariance
        )

    print(
        "Pose Prior added to image #{0} ({1}): {2}".format(
            image_id, image_name, position
        )
    )


def get_pose_prior_from_line(
    line: str,
    delimiter: str,
    coordinate_system: int,
    position_covariance: np.array,
):
    els = line.split(delimiter)

    if len(els) == 13 and position_covariance is None:
        # Get image_name, postion and covariance (in row major format)
        image_name = els[0]
        position = np.array([float(els[1]), float(els[2]), float(els[3])])
        covariance_values = list(map(float, els[4:13]))
        position_covariance = np.array(covariance_values).reshape(3, 3)
        return (image_name, position, coordinate_system, position_covariance)
    elif len(els) >= 4 and position_covariance is not None:
        # Get image_name, position
        image_name = els[0]
        position = np.array([float(els[1]), float(els[2]), float(els[3])])
        return (image_name, position, coordinate_system, position_covariance)
    else:
        print("ERROR: Pose priors file lines should contain 4 or 13 elements.")
        print("Current line contains: {0}: #{1} elements".format(els, len(els)))

    return None


def write_pose_priors_to_database():
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--database_path", type=str, required=True)
    parser.add_argument("--pose_priors_path", type=str, required=True)
    parser.add_argument("--pose_priors_delimiter", type=str, default=" ")
    parser.add_argument(
        "--coordinate_system",
        type=int,
        default=0,
        help="(-1: unknwon, 0: WGS84, 1: Cartesian)")
    parser.add_argument(
        "--use_covariance_from_pose_priors_file",
        type=bool,
        default=False,
        help="If False, use prior_position_std options to set a common covariance to the priors.")
    parser.add_argument("--prior_position_std_x", type=float, default=1.0)
    parser.add_argument("--prior_position_std_y", type=float, default=1.0)
    parser.add_argument("--prior_position_std_z", type=float, default=1.0)
    args = parser.parse_args()

    if not os.path.exists(args.database_path):
        print("ERROR: database path does not exist.")
        return

    if not os.path.exists(args.pose_priors_path):
        print("ERROR: pose priors path already does not exist.")
        return

    # Open the database.
    db = COLMAPDatabase.connect(args.database_path)

    # Setup covariance matrix if required
    position_covariance = None
    if args.use_covariance_from_pose_priors_file is False:
        position_covariance = np.diag(
            [
                args.prior_position_std_x**2,
                args.prior_position_std_y**2,
                args.prior_position_std_z**2,
            ]
        )

    # Add pose priors from file.
    pose_prior_file = open(args.pose_priors_path, "r")
    for line in pose_prior_file:
        if line[0] == "#":
            continue
        pose_prior = get_pose_prior_from_line(
            line,
            args.pose_priors_delimiter,
            args.coordinate_system,
            position_covariance,
        )
        if pose_prior is not None:
            update_pose_prior_from_image_name(db, *pose_prior)

    # Commit the data to the file.
    db.commit()

    # Close database.
    db.close()


if __name__ == "__main__":
    write_pose_priors_to_database()
