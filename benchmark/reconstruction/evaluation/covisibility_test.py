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

import numpy as np

import pycolmap

from .covisibility import (
    Frustum,
    _build_image_point3D_sets,
    _estimate_depth_ranges,
    _is_pair_covisible_by_frustum,
    _sample_frustum_points,
)


def _make_camera() -> pycolmap.Camera:
    cam = pycolmap.Camera()
    cam.model = pycolmap.CameraModelId.PINHOLE
    cam.width = 640
    cam.height = 480
    cam.params = [500, 500, 320, 240]
    cam.camera_id = 1
    return cam


def _add_image(
    recon: pycolmap.Reconstruction,
    image_id: int,
    name: str,
    position: np.ndarray,
    rotation: pycolmap.Rotation3d | None = None,
    num_points2D: int = 0,
) -> None:
    if rotation is None:
        rotation = pycolmap.Rotation3d()
    translation = -rotation.matrix() @ position

    points2d = [pycolmap.Point2D() for _ in range(num_points2D)]
    for i, p in enumerate(points2d):
        p.xy = np.array([100.0 + i * 10, 200.0])

    frame = pycolmap.Frame()
    frame.rig_id = 1
    frame.frame_id = image_id

    img = pycolmap.Image()
    img.name = name
    img.camera_id = 1
    img.frame_id = image_id
    img.image_id = image_id
    if num_points2D > 0:
        img.points2D = pycolmap.Point2DList(points2d)

    frame.add_data_id(img.data_id)
    frame.rig_from_world = pycolmap.Rigid3d(rotation, translation)
    recon.add_frame(frame)
    recon.add_image(img)


class TestEstimateDepthRanges:
    def test_single_image_with_points(self):
        recon = pycolmap.Reconstruction()
        cam = _make_camera()
        recon.add_camera_with_trivial_rig(cam)
        _add_image(recon, 1, "img1.jpg", np.array([0, 0, 0]), num_points2D=20)

        depths = []
        for i in range(20):
            depth = 2.0 + i * 0.5
            depths.append(depth)
            recon.add_point3D(
                np.array([0.0, 0.0, depth]),
                pycolmap.Track([pycolmap.TrackElement(1, i)]),
            )

        ranges = _estimate_depth_ranges(recon)
        assert 1 in ranges
        near, far = ranges[1]
        np.testing.assert_almost_equal(near, np.percentile(depths, 2.0))
        np.testing.assert_almost_equal(far, np.percentile(depths, 98.0))

    def test_insufficient_points_returns_default(self):
        recon = pycolmap.Reconstruction()
        cam = _make_camera()
        recon.add_camera_with_trivial_rig(cam)
        _add_image(recon, 1, "img1.jpg", np.array([0, 0, 0]), num_points2D=5)

        for i in range(5):
            recon.add_point3D(
                np.array([0.0, 0.0, float(i + 1)]),
                pycolmap.Track([pycolmap.TrackElement(1, i)]),
            )

        ranges = _estimate_depth_ranges(recon)
        assert ranges[1] == (0.1, 100.0)

    def test_per_image_ranges_differ(self):
        recon = pycolmap.Reconstruction()
        cam = _make_camera()
        recon.add_camera_with_trivial_rig(cam)
        _add_image(recon, 1, "img1.jpg", np.array([0, 0, 0]), num_points2D=20)
        _add_image(recon, 2, "img2.jpg", np.array([0, 0, 100]), num_points2D=20)

        # Points near image 1 (depths 1-10 from image 1)
        for i in range(20):
            depth = 1.0 + i * 0.5
            recon.add_point3D(
                np.array([0.0, 0.0, depth]),
                pycolmap.Track([pycolmap.TrackElement(1, i)]),
            )

        # Points near image 2 (depths 1-10 from image 2, so z ~ 90-99)
        for i in range(20):
            depth = 1.0 + i * 0.5
            recon.add_point3D(
                np.array([0.0, 0.0, 100.0 + depth]),
                pycolmap.Track([pycolmap.TrackElement(2, i)]),
            )

        ranges = _estimate_depth_ranges(recon)
        near1, far1 = ranges[1]
        near2, far2 = ranges[2]
        # Image 1 sees nearby points, image 2 sees distant points
        assert near1 < 15
        assert near2 < 15


class TestSampleFrustumPoints:
    def test_identity_camera_shape(self):
        recon = pycolmap.Reconstruction()
        cam = _make_camera()
        recon.add_camera_with_trivial_rig(cam)
        _add_image(recon, 1, "img.jpg", np.array([0, 0, 0]))

        verts = _sample_frustum_points(recon.images[1], cam, 1.0, 10.0)
        # num_steps=5 default: 5*5 grid at 5 depths = 125 points
        assert verts.shape == (125, 3)

    def test_custom_num_steps(self):
        recon = pycolmap.Reconstruction()
        cam = _make_camera()
        recon.add_camera_with_trivial_rig(cam)
        _add_image(recon, 1, "img.jpg", np.array([0, 0, 0]))

        verts = _sample_frustum_points(
            recon.images[1], cam, 1.0, 10.0, num_steps=3
        )
        assert verts.shape == (27, 3)

    def test_vertices_depth_range(self):
        recon = pycolmap.Reconstruction()
        cam = _make_camera()
        recon.add_camera_with_trivial_rig(cam)
        # Identity pose: camera at origin looking +z
        _add_image(recon, 1, "img.jpg", np.array([0, 0, 0]))

        near, far = 2.0, 20.0
        verts = _sample_frustum_points(recon.images[1], cam, near, far)
        # In world coords, z should span [near, far] for identity pose
        np.testing.assert_almost_equal(verts[:, 2].min(), near)
        np.testing.assert_almost_equal(verts[:, 2].max(), far)

    def test_translated_camera(self):
        recon = pycolmap.Reconstruction()
        cam = _make_camera()
        recon.add_camera_with_trivial_rig(cam)
        offset = np.array([10.0, 20.0, 30.0])
        _add_image(recon, 1, "img.jpg", offset)

        verts = _sample_frustum_points(
            recon.images[1], cam, 1.0, 10.0, num_steps=2
        )
        # Center of near plane should be at camera position + [0,0,near]
        center = verts.mean(axis=0)
        np.testing.assert_allclose(center[:2], offset[:2], atol=1e-10)
        assert center[2] > offset[2]


class TestCheckFrustumCovisibility:
    def _build_two_camera_recon(
        self, pos1, pos2, rot1=None, rot2=None
    ) -> pycolmap.Reconstruction:
        recon = pycolmap.Reconstruction()
        cam = _make_camera()
        recon.add_camera_with_trivial_rig(cam)
        _add_image(recon, 1, "img1.jpg", np.array(pos1), rot1)
        _add_image(recon, 2, "img2.jpg", np.array(pos2), rot2)
        return recon

    def _check(self, recon, max_angle=90.0, near=1.0, far=100.0):
        cam = recon.cameras[1]
        imgs = [recon.images[1], recon.images[2]]
        frustums = {
            img.image_id: Frustum(
                points=_sample_frustum_points(img, cam, near, far),
                depth_range=(near, far),
            )
            for img in imgs
        }
        return _is_pair_covisible_by_frustum(
            imgs[0], imgs[1], recon, frustums, max_angle
        )

    def test_covisible_side_by_side(self):
        recon = self._build_two_camera_recon([-0.5, 0, -5], [0.5, 0, -5])
        assert self._check(recon)

    def test_not_covisible_opposite_directions(self):
        rot_180_y = pycolmap.Rotation3d(np.array([0, 1, 0, 0]))
        recon = self._build_two_camera_recon(
            [0, 0, -5], [0, 0, 5], rot2=rot_180_y
        )
        assert not self._check(recon)

    def test_rejected_by_viewing_angle(self):
        # 90 deg rotation around y-axis
        rot = pycolmap.Rotation3d(
            np.array([0, 1 / np.sqrt(2), 0, 1 / np.sqrt(2)])
        )
        recon = self._build_two_camera_recon([0, 0, -5], [5, 0, -5], rot2=rot)
        # Viewing angle is 90 deg, threshold is 45 -> should be rejected
        assert not self._check(recon, max_angle=45.0)

    def test_covisible_converging_cameras(self):
        # Two cameras angled inward looking at the same point
        # Camera 1 at (-2,0,0) looking at +x (toward origin)
        rot1 = pycolmap.Rotation3d(
            np.array([0, -1 / np.sqrt(2), 0, 1 / np.sqrt(2)])
        )
        # Camera 2 at (2,0,0) looking at -x (toward origin)
        rot2 = pycolmap.Rotation3d(
            np.array([0, 1 / np.sqrt(2), 0, 1 / np.sqrt(2)])
        )
        recon = self._build_two_camera_recon(
            [-5, 0, 0], [5, 0, 0], rot1=rot1, rot2=rot2
        )
        assert self._check(recon, max_angle=180.0)

    def test_not_covisible_far_apart_narrow_fov(self):
        # Two cameras far apart, both looking +z, with small frustum depth
        recon = self._build_two_camera_recon([0, 0, 0], [1000, 0, 0])
        assert self._check(recon, near=1.0, far=2.0) is False

    def test_not_covisible_depth_out_of_range(self):
        # Two cameras side by side looking +z. Their frustums project into
        # each other's images, but the depth ranges don't overlap: camera 1
        # expects depths 1-5 while camera 2 expects 50-100. Each camera's
        # frustum points land outside the other's depth range.
        recon = self._build_two_camera_recon([-0.5, 0, 0], [0.5, 0, 0])
        cam = recon.cameras[1]
        img1, img2 = recon.images[1], recon.images[2]
        frustums = {
            img1.image_id: Frustum(
                points=_sample_frustum_points(img1, cam, 1.0, 5.0),
                depth_range=(1.0, 5.0),
            ),
            img2.image_id: Frustum(
                points=_sample_frustum_points(img2, cam, 50.0, 100.0),
                depth_range=(50.0, 100.0),
            ),
        }
        assert not _is_pair_covisible_by_frustum(
            img1, img2, recon, frustums, 90.0
        )
        # Sanity: with matching depth ranges, these cameras are covisible.
        assert self._check(recon, near=1.0, far=100.0)

    def test_covisible_identical_cameras(self):
        recon = self._build_two_camera_recon([0, 0, 0], [0, 0, 0])
        assert self._check(recon)


def _make_recon_with_tracks(
    num_images: int,
    num_points2D_per_image: int,
    tracks: list[list[tuple[int, int]]],
) -> pycolmap.Reconstruction:
    """Build a reconstruction with images and 3D point tracks.

    Args:
        num_images: Number of images to create.
        num_points2D_per_image: Number of 2D points per image.
        tracks: List of tracks, where each track is a list of
            (image_id, point2D_idx) tuples.
    """
    recon = pycolmap.Reconstruction()
    cam = _make_camera()
    recon.add_camera_with_trivial_rig(cam)

    for i in range(1, num_images + 1):
        _add_image(
            recon,
            i,
            f"img{i}.jpg",
            np.array([i * 2.0, 0, 0]),
            num_points2D=num_points2D_per_image,
        )

    for track in tracks:
        xyz = np.array([0.0, 0.0, 5.0])
        elements = [pycolmap.TrackElement(img_id, idx) for img_id, idx in track]
        recon.add_point3D(xyz, pycolmap.Track(elements))

    return recon


class TestBuildImagePoint3DSets:
    def test_basic_tracks(self):
        # 2 images, 5 points2D each, 3 shared tracks
        recon = _make_recon_with_tracks(
            num_images=2,
            num_points2D_per_image=5,
            tracks=[
                [(1, 0), (2, 0)],
                [(1, 1), (2, 1)],
                [(1, 2), (2, 2)],
            ],
        )
        sets = _build_image_point3D_sets(recon)
        assert len(sets[1]) == 3
        assert len(sets[2]) == 3
        assert len(sets[1] & sets[2]) == 3

    def test_no_shared_tracks(self):
        recon = _make_recon_with_tracks(
            num_images=2,
            num_points2D_per_image=5,
            tracks=[
                [(1, 0)],
                [(1, 1)],
                [(2, 0)],
                [(2, 1)],
            ],
        )
        sets = _build_image_point3D_sets(recon)
        assert len(sets[1] & sets[2]) == 0

    def test_no_tracks(self):
        recon = _make_recon_with_tracks(
            num_images=2,
            num_points2D_per_image=5,
            tracks=[],
        )
        sets = _build_image_point3D_sets(recon)
        assert len(sets[1]) == 0
        assert len(sets[2]) == 0

    def test_partial_overlap(self):
        # 3 images: img1-img2 share 5 points
        # img2-img3 share 2, img1-img3 share 0
        recon = _make_recon_with_tracks(
            num_images=3,
            num_points2D_per_image=10,
            tracks=[
                [(1, 0), (2, 0)],
                [(1, 1), (2, 1)],
                [(1, 2), (2, 2)],
                [(1, 3), (2, 3)],
                [(1, 4), (2, 4)],
                [(2, 5), (3, 0)],
                [(2, 6), (3, 1)],
            ],
        )
        sets = _build_image_point3D_sets(recon)
        assert len(sets[1] & sets[2]) == 5
        assert len(sets[2] & sets[3]) == 2
        assert len(sets[1] & sets[3]) == 0
