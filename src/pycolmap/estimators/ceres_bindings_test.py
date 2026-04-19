import pycolmap


def test_pyceres_submodule_exists():
    assert hasattr(pycolmap._core, "pyceres")


def test_minimizer_type_enum():
    pyceres = pycolmap._core.pyceres
    assert {m.name: int(m) for m in pyceres.MinimizerType} == {
        "LINE_SEARCH": 0,
        "TRUST_REGION": 1,
    }


def test_linear_solver_type_enum():
    pyceres = pycolmap._core.pyceres
    assert {m.name: int(m) for m in pyceres.LinearSolverType} == {
        "DENSE_NORMAL_CHOLESKY": 0,
        "DENSE_QR": 1,
        "SPARSE_NORMAL_CHOLESKY": 2,
        "DENSE_SCHUR": 3,
        "SPARSE_SCHUR": 4,
        "ITERATIVE_SCHUR": 5,
    }


def test_trust_region_strategy_type_enum():
    pyceres = pycolmap._core.pyceres
    assert {m.name: int(m) for m in pyceres.TrustRegionStrategyType} == {
        "LEVENBERG_MARQUARDT": 0,
        "DOGLEG": 1,
    }


def test_logging_type_enum():
    pyceres = pycolmap._core.pyceres
    assert {m.name: int(m) for m in pyceres.LoggingType} == {
        "SILENT": 0,
        "PER_MINIMIZER_ITERATION": 1,
    }


def test_termination_type_enum():
    pyceres = pycolmap._core.pyceres
    assert {m.name: int(m) for m in pyceres.TerminationType} == {
        "CONVERGENCE": 0,
        "NO_CONVERGENCE": 1,
        "FAILURE": 2,
        "USER_SUCCESS": 3,
        "USER_FAILURE": 4,
    }


def test_solver_options_default_init():
    options = pycolmap._core.pyceres.SolverOptions()
    assert options is not None


def test_solver_options_is_valid():
    options = pycolmap._core.pyceres.SolverOptions()
    valid, error = options.IsValid()
    assert valid is True
    assert error == ""


def test_solver_summary_default_init():
    summary = pycolmap._core.pyceres.SolverSummary()
    assert summary is not None


def test_solver_summary_brief_report():
    summary = pycolmap._core.pyceres.SolverSummary()
    report = summary.BriefReport()
    assert isinstance(report, str)


def test_solver_summary_full_report():
    summary = pycolmap._core.pyceres.SolverSummary()
    report = summary.FullReport()
    assert isinstance(report, str)


def test_solver_summary_is_solution_usable():
    summary = pycolmap._core.pyceres.SolverSummary()
    result = summary.IsSolutionUsable()
    assert isinstance(result, bool)


def test_iteration_summary_accessible():
    summary = pycolmap._core.pyceres.IterationSummary()
    assert summary is not None
