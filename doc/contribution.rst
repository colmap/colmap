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
