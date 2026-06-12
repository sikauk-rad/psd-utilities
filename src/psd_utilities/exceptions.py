class PSDUtilitiesBaseError(Exception):

    """
    Base exception class for PSD utilities package.
    """
    ...


class PSDError(ValueError, PSDUtilitiesBaseError):

    """
    Exception raised for errors in particle size distribution (PSD) processing.

    This error is typically raised when a PSD array fails validation checks, such as
    having duplicate sieve sizes, non-monotonic cumulative fractions, or incorrect
    normalisation.

    Parameters
    ----------
        *args : Any
        Arguments passed to the base ValueError.
        **kwargs : Any
        Keyword arguments passed to the base ValueError.

    See Also
    --------
        ValueError : Base exception for value errors.
    """

    ...


class QuantileError(ValueError, PSDUtilitiesBaseError):

    """
    Exception raised for errors in quantile processing.

    This error is typically raised when a quantile fails validation checks.

    Parameters
    ----------
        *args : Any
        Arguments passed to the base ValueError.
        **kwargs : Any
        Keyword arguments passed to the base ValueError.

    See Also
    --------
        ValueError : Base exception for value errors.
    """

    ...


class NoSolutionError(ValueError, PSDUtilitiesBaseError):

    """
    Exception raised for errors in optimisation.

    This error is typically raised when a PSD optimisation fails.

    Parameters
    ----------
        *args : Any
        Arguments passed to the base ValueError.
        **kwargs : Any
        Keyword arguments passed to the base ValueError.

    See Also
    --------
        ValueError : Base exception for value errors.
    """

    ...