Contribution
============

Contributions (bug reports, bug fixes, improvements, etc.) are very welcome and
should be submitted in the form of new issues and/or pull requests on GitHub.

Please, adhere to the Google coding style guide::

    https://google.github.io/styleguide/cppguide.html

by using the provided ".clang-format" file.

Document functions, methods, classes, etc. with inline documentation strings
describing the API, using the following format::

    // Short description.
    //
    // Longer description with a few sentences and multiple lines.
    //
    // @param parameter1            Description for parameter 1.
    // @param parameter2            Description for parameter 2.
    //
    // @return                      Description of optional return value.

Add unit tests for all newly added code and make sure that algorithmic
"improvements" generalize and actually improve the results of the pipeline on a
variety of datasets.

Running Tests
-------------

Build with ``-DTESTS_ENABLED=ON`` and run the test suite from the build
directory::

    cd build
    ctest --output-on-failure

Tests that run expensive neural-network inference (e.g. the full-size LoMa
smoke test) are skipped by default and only run when the
``COLMAP_TEST_HEAVY`` environment variable is set::

    COLMAP_TEST_HEAVY=1 ctest --output-on-failure

Continuous integration always runs the test suite with ``COLMAP_TEST_HEAVY``
set, so heavy tests are covered there. When adding a test whose runtime is
dominated by model loading or inference, prefer a fast default path and gate
the expensive variant behind this variable.
