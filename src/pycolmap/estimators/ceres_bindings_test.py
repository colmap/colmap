import pycolmap


def test_pyceres_submodule_exists():
    assert hasattr(pycolmap._core, "pyceres")


def test_minimizer_type_enum():
    pyceres = pycolmap._core.pyceres
    assert pyceres.MinimizerType.LINE_SEARCH is not None
    assert pyceres.MinimizerType.TRUST_REGION is not None


def test_linear_solver_type_enum():
    pyceres = pycolmap._core.pyceres
    assert pyceres.LinearSolverType.DENSE_NORMAL_CHOLESKY is not None
    assert pyceres.LinearSolverType.DENSE_QR is not None
    assert pyceres.LinearSolverType.SPARSE_NORMAL_CHOLESKY is not None
    assert pyceres.LinearSolverType.DENSE_SCHUR is not None
    assert pyceres.LinearSolverType.SPARSE_SCHUR is not None
    assert pyceres.LinearSolverType.ITERATIVE_SCHUR is not None


def test_trust_region_strategy_type_enum():
    pyceres = pycolmap._core.pyceres
    assert pyceres.TrustRegionStrategyType.LEVENBERG_MARQUARDT is not None
    assert pyceres.TrustRegionStrategyType.DOGLEG is not None


def test_logging_type_enum():
    pyceres = pycolmap._core.pyceres
    assert pyceres.LoggingType.SILENT is not None
    assert pyceres.LoggingType.PER_MINIMIZER_ITERATION is not None


def test_termination_type_enum():
    pyceres = pycolmap._core.pyceres
    assert pyceres.TerminationType.CONVERGENCE is not None
    assert pyceres.TerminationType.NO_CONVERGENCE is not None
    assert pyceres.TerminationType.FAILURE is not None
    assert pyceres.TerminationType.USER_SUCCESS is not None
    assert pyceres.TerminationType.USER_FAILURE is not None


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
