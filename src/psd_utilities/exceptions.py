class PSDError(ValueError):

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