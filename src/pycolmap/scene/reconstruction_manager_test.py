import os

import pycolmap


def test_reconstruction_manager_init():
    manager = pycolmap.ReconstructionManager()
    assert manager is not None


def test_reconstruction_manager_size():
    manager = pycolmap.ReconstructionManager()
    assert manager.size() == 0


def test_reconstruction_manager_add():
    manager = pycolmap.ReconstructionManager()
    idx = manager.add()
    assert idx == 0
    assert manager.size() == 1


def test_reconstruction_manager_get():
    manager = pycolmap.ReconstructionManager()
    idx = manager.add()
    reconstruction = manager.get(idx)
    assert reconstruction is not None


def test_reconstruction_manager_delete():
    manager = pycolmap.ReconstructionManager()
    idx = manager.add()
    manager.delete(idx)
    assert manager.size() == 0


def test_reconstruction_manager_clear():
    manager = pycolmap.ReconstructionManager()
    manager.add()
    manager.add()
    assert manager.size() == 2
    manager.clear()
    assert manager.size() == 0


def test_reconstruction_manager_write_read_roundtrip(tmp_path, simple_camera):
    manager = pycolmap.ReconstructionManager()
    idx = manager.add()
    reconstruction = manager.get(idx)
    reconstruction.add_camera(simple_camera)
    output_dir = str(tmp_path / "manager")
    os.makedirs(output_dir)
    manager.write(output_dir)
    loaded_manager = pycolmap.ReconstructionManager()
    loaded_manager.read(os.path.join(output_dir, "0"))
    assert loaded_manager.size() == 1
