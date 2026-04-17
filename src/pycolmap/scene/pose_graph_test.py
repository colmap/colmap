import pycolmap


def test_pose_graph_edge_default_init():
    edge = pycolmap.PoseGraphEdge()
    assert edge is not None


def test_pose_graph_edge_init_with_rigid3d():
    rigid = pycolmap.Rigid3d()
    edge = pycolmap.PoseGraphEdge(cam2_from_cam1=rigid)
    assert edge is not None


def test_pose_graph_edge_cam2_from_cam1_readwrite():
    edge = pycolmap.PoseGraphEdge()
    rigid = pycolmap.Rigid3d()
    edge.cam2_from_cam1 = rigid
    assert isinstance(edge.cam2_from_cam1, pycolmap.Rigid3d)


def test_pose_graph_edge_num_matches_readwrite():
    edge = pycolmap.PoseGraphEdge()
    edge.num_matches = 100
    assert edge.num_matches == 100


def test_pose_graph_edge_valid_readwrite():
    edge = pycolmap.PoseGraphEdge()
    edge.valid = True
    assert edge.valid is True
    edge.valid = False
    assert edge.valid is False


def test_pose_graph_edge_invert():
    edge = pycolmap.PoseGraphEdge()
    edge.valid = True
    edge.num_matches = 50
    edge.invert()


def test_pose_graph_default_init():
    graph = pycolmap.PoseGraph()
    assert graph is not None


def test_pose_graph_empty():
    graph = pycolmap.PoseGraph()
    assert graph.empty is True


def test_pose_graph_num_edges():
    graph = pycolmap.PoseGraph()
    assert graph.num_edges == 0


def test_pose_graph_add_edge():
    graph = pycolmap.PoseGraph()
    edge = pycolmap.PoseGraphEdge()
    edge.num_matches = 10
    edge.valid = True
    graph.add_edge(1, 2, edge)
    assert graph.num_edges == 1


def test_pose_graph_has_edge():
    graph = pycolmap.PoseGraph()
    edge = pycolmap.PoseGraphEdge()
    edge.num_matches = 10
    edge.valid = True
    graph.add_edge(1, 2, edge)
    assert graph.has_edge(1, 2)
    assert not graph.has_edge(3, 4)


def test_pose_graph_get_edge():
    graph = pycolmap.PoseGraph()
    edge = pycolmap.PoseGraphEdge()
    edge.num_matches = 42
    edge.valid = True
    graph.add_edge(1, 2, edge)
    retrieved_edge = graph.get_edge(1, 2)
    assert retrieved_edge.num_matches == 42


def test_pose_graph_delete_edge():
    graph = pycolmap.PoseGraph()
    edge = pycolmap.PoseGraphEdge()
    edge.num_matches = 10
    edge.valid = True
    graph.add_edge(1, 2, edge)
    result = graph.delete_edge(1, 2)
    assert result is True
    assert graph.num_edges == 0


def test_pose_graph_update_edge():
    graph = pycolmap.PoseGraph()
    edge = pycolmap.PoseGraphEdge()
    edge.num_matches = 10
    edge.valid = True
    graph.add_edge(1, 2, edge)
    updated_edge = pycolmap.PoseGraphEdge()
    updated_edge.num_matches = 99
    updated_edge.valid = True
    graph.update_edge(1, 2, updated_edge)
    retrieved = graph.get_edge(1, 2)
    assert retrieved.num_matches == 99


def test_pose_graph_clear():
    graph = pycolmap.PoseGraph()
    edge = pycolmap.PoseGraphEdge()
    edge.num_matches = 10
    edge.valid = True
    graph.add_edge(1, 2, edge)
    graph.clear()
    assert graph.num_edges == 0
    assert graph.empty is True


def test_pose_graph_edges_property():
    graph = pycolmap.PoseGraph()
    edge = pycolmap.PoseGraphEdge()
    edge.num_matches = 10
    edge.valid = True
    graph.add_edge(1, 2, edge)
    edges = graph.edges
    assert len(edges) == 1


def test_pose_graph_edge_map_type():
    graph = pycolmap.PoseGraph()
    edges = graph.edges
    assert isinstance(edges, pycolmap.PoseGraphEdgeMap)
