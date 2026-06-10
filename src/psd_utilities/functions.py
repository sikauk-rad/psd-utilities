from collections import defaultdict
from collections.abc import Iterable
from typing import Literal

import numpy as np
from scipy.interpolate import PchipInterpolator

from .datatypes import TwoDFloatArrayType, TwoColumnArrayType
from .exceptions import PSDError

def deduplicate_sort_sieves(
    psd: TwoColumnArrayType,
    material_name: str = 'material'
) -> TwoColumnArrayType:

    """
    Remove duplicate sieves and sort the particle size distribution (psd) array.

    Ensures that the input psd array contains unique sieve sizes (first column) and is sorted.
    If duplicate sieve sizes are found, raises a ValueError indicating the problematic sieves.

    Parameters
    ----------
    psd : np.ndarray[tuple[int, Literal[2]], np.dtype[np.number]]
        An (n,2) NumPy array representing the particle size distribution.
        The first column should contain sieve sizes and the second sieve fractions.
    material_name : str, optional
        Name of the material, used for error messages. Default is 'material'.

    Returns
    -------
    np.ndarray[tuple[int, Literal[2]], np.dtype[np.number]]
        A 2D NumPy array with unique, sorted sieve fractions that add to 1.

    Raises
    ------
    PSDError
        If duplicate sieve sizes are found in the input psd array.

    Examples
    --------
    >>> import numpy as np
    >>> psd = np.array([[4.75, 10], [2.36, 20], [4.75, 15]])
    >>> deduplicate_sort_sieves(psd, material_name='Sand')
    array([[2.36, 20.  ],
           [4.75, 10.  ]])
    """

    psd_unique_sorted = np.unique(psd, axis = 0, sorted = True)
    unique_sieve_mask = psd_unique_sorted[:-1,0] != psd_unique_sorted[1:,0]
    if not unique_sieve_mask.all():
        problematic_sieves = psd_unique_sorted[(~unique_sieve_mask).nonzero(), 0]
        problematic_sieves_text = np.array2string(problematic_sieves)
        raise PSDError(
            f'The psd of {material_name} has duplicated sieves {problematic_sieves_text}.'
        )
    else:
        return psd_unique_sorted


def check_cumulative_psd(
    psd_unique_sorted: TwoColumnArrayType,
    *,
    passing: bool,
    material_name: str = 'material'
) -> None:

    """
    Validate that a cumulative particle size distribution (psd) is monotonic and has 
    correct boundary values.

    This function checks that a cumulative psd is:
        - Monotonic with respect to sieve size (increasing for passing, decreasing for 
        retained)
        - 0 at the smallest sieve and 1 at the largest sieve for passing distributions
        - 1 at the smallest sieve and 0 at the largest sieve for retained distributions

    Parameters
    ----------
    psd_unique_sorted : np.ndarray[tuple[int, Literal[2]], np.dtype[np.number]]
        A 2D NumPy array of shape (N, 2), where the first column contains sieve sizes
        (sorted in ascending order and unique), and the second column contains cumulative
        fractions (either passing or retained).
    passing : bool
        If True, checks that cumulative passing fractions are non-decreasing with sieve 
        size.
        If False, checks that cumulative retained fractions are non-increasing with sieve
         size.
    material_name : str, optional
        Name of the material for error messages. Default is 'material'.

    Raises
    ------
    PSDError
        If the cumulative fractions are not monotonic in the expected direction, or if
        the boundary values at the smallest or largest sieve are incorrect. The error
        message indicates the sieve sizes where the issue occurs.

    Examples
    --------
    >>> import numpy as np
    >>> psd = np.array([[2.36, 0.0], [4.75, 0.5], [9.5, 1.0]])
    >>> check_cumulative_psd(psd, passing=True)
    # No exception raised

    >>> psd_invalid = np.array([[2.36, 0.0], [4.75, 0.6], [9.5, 0.5]])
    >>> check_cumulative_psd(psd_invalid, passing=True)
    Traceback (most recent call last):
        ...
    PSDError: Cumulative passing psd fractions must increase with sieve size. There are issues with material at [9.5].
    """
    
    if passing:
        operator, name, prefix = np.less_equal, 'passing', 'in'
        if psd_unique_sorted[0,1]:
            raise PSDError(
                'The fraction on the smallest sieve in a cumulative passing '\
                f'distribution must be 0 (failed for {material_name}).'
            )
        elif not np.isclose(psd_unique_sorted[-1,1], 1, rtol = 0, atol = 1e-4):
            raise PSDError(
                'The fraction on the largest sieve in a cumulative passing '\
                f'distribution must be 1 (failed for {material_name}).'
            )
    else:
        operator, name, prefix = np.greater_equal, 'retained', 'de'
        if not np.isclose(psd_unique_sorted[0,1], 1, rtol = 0, atol = 1e-4):
            raise PSDError(
                'The fraction on the smallest sieve in a cumulative retained '\
                f'distribution must be 1 (failed for {material_name}).'
            )
        elif psd_unique_sorted[-1,1]:
            raise PSDError(
                'The fraction on the largest sieve in a cumulative retained '\
                f'distribution must be 0 (failed for {material_name}).'
            )

    fractions_ordered_mask = operator(psd_unique_sorted[:-1,1], psd_unique_sorted[1:,1])
    if not np.all(fractions_ordered_mask):
        problematic_sieves = psd_unique_sorted[(~fractions_ordered_mask).nonzero(), 0]
        problematic_sieves_text = np.array2string(problematic_sieves)
        raise PSDError(
            f'Cumulative {name} psd fractions must {prefix}crease with sieve size. '\
            f'There are issues with {material_name} at {problematic_sieves_text}.'
        )


def check_range_psd(
    psd_unique_sorted: TwoColumnArrayType,
    material_name: str = 'material',
    _check_first_zero: bool = True,
    _check_last_zero: bool = True,
) -> None:

    """
    Validate that a range particle size distribution (psd) is properly normalised.

    This function checks that the fractions in a range psd:
        - Sum to 1 (i.e., the distribution is normalised).
        - Are zero at the smallest and largest sieve sizes.

    Parameters
    ----------
    psd_unique_sorted : np.ndarray[tuple[int, Literal[2]], np.dtype[np.number]]
        A 2D NumPy array of shape (N, 2), where the first column contains sieve sizes
        (sorted in ascending order and unique), and the second column contains range 
        fractions.
    material_name : str, optional
        Name of the material for error messages. Default is 'material'.

    Raises
    ------
    PSDError
        If the fractions do not sum to 1, or if the first or last sieve fraction is not 
        zero.

    Examples
    --------
    >>> import numpy as np
    >>> psd = np.array([[2.36, 0.0], [4.75, 0.5], [9.5, 0.5], [19.0, 0.0]])
    >>> check_range_psd(psd)
    # No exception raised

    >>> psd_invalid = np.array([[2.36, 0.1], [4.75, 0.5], [9.5, 0.4], [19.0, 0.0]])
    >>> check_range_psd(psd_invalid)
    Traceback (most recent call last):
        ...
    PSDError: Range psds for material must sum to 1.
    """

    if not np.isclose(psd_unique_sorted[:,1].sum(), 1, rtol = 0, atol = 1e-4):
        raise PSDError(f'Range psds must sum to 1 (failed for {material_name}).')
    elif _check_first_zero and psd_unique_sorted[0,1]:
        raise PSDError(
            'The first sieve in a range psd must have a fraction of 0 (failed for '\
            f'{material_name}).'
        )
    elif _check_last_zero and psd_unique_sorted[0,-1]:
        raise PSDError(
            'The last sieve in a range psd must have a fraction of 0 (failed for '\
            f'{material_name}).'
        )


def convert_between_retained_and_passing(
    psd_unique_sorted: TwoColumnArrayType,
    *,
    cumulative: bool,
    copy: bool = True
) -> TwoColumnArrayType:

    """
    Convert between retained and passing fractions in a particle size distribution (psd).

    For cumulative psds, this function converts retained fractions to passing fractions,
    or vice versa, by computing 1 minus the supplied fraction at each sieve. The 
    transformation is symmetric: applying it twice returns the original data. For range 
    (non-cumulative) psds, retained and passing fractions are identical, so the input is 
    returned unchanged.

    Parameters
    ----------
    psd_unique_sorted : np.ndarray[tuple[int, Literal[2]], np.dtype[np.number]]
        A 2D NumPy array of shape (N, 2), where the first column contains sieve sizes
        (sorted in ascending order and unique), and the second column contains cumulative
        or range retained or passing fractions.
    cumulative : bool
        If True, applies the transformation (1 - fraction) to the second column.
        If False, returns the input unchanged, as range retained and range passing are
        identical.
    copy : bool, optional
        If True, returns a copy of the input array. If False, modifies the input array
        in place. Default is True.

    Returns
    -------
    psd_out : np.ndarray[tuple[int, Literal[2]], np.dtype[np.number]]
        A 2D NumPy array of the same shape as the input, with the second column 
        containing the converted fractions if cumulative is True, or unchanged if
        cumulative is False.

    Examples
    --------
    >>> import numpy as np
    >>> psd = np.array([[2.36, 1.0], [4.75, 0.7], [9.5, 0.2], [19.0, 0.0]])
    >>> convert_between_retained_and_passing(psd, cumulative=True)
    array([[2.36, 0. ],
           [4.75, 0.3],
           [9.5 , 0.8],
           [19. , 1. ]])
    >>> convert_between_retained_and_passing(_, cumulative=True)
    array([[2.36, 1. ],
           [4.75, 0.7],
           [9.5 , 0.2],
           [19. , 0. ]])
    """

    psd_passing = psd_unique_sorted.copy() if copy else psd_unique_sorted
    if cumulative:
        psd_passing[:,1] = 1 - psd_unique_sorted[:,1]
    return psd_passing


def convert_cumulative_to_range(
    psd_unique_sorted: TwoColumnArrayType,
    *,
    passing: bool,
    copy: bool = True,
) -> TwoColumnArrayType:

    """
    Convert cumulative psd fractions to range (incremental) fractions.

    For cumulative passing or retained fractions, this function computes the range
    (incremental) fraction for each sieve interval.

    Parameters
    ----------
    psd_unique_sorted : np.ndarray[tuple[int, Literal[2]], np.dtype[np.number]]
        A 2D NumPy array of shape (N, 2), where the first column contains sieve sizes
        (sorted in ascending order and unique), and the second column contains cumulative
        passing or retained fractions.
    passing : bool
        If True, interprets the second column as cumulative passing fractions.
        If False, interprets the second column as cumulative retained fractions.
    copy : bool, optional
        If True, returns a copy of the input array. If False, modifies the input array
        in place. Default is True.

    Returns
    -------
    psd_range : np.ndarray[tuple[int, Literal[2]], np.dtype[np.number]]
        A 2D NumPy array of the same shape as the input, with the second column
        containing range (incremental) fractions.

    Examples
    --------
    >>> import numpy as np
    >>> psd_cum = np.array([[2.36, 0.0], [4.75, 0.3], [9.5, 0.8], [19.0, 1.0]])
    >>> convert_cumulative_to_range(psd_cum, passing=True)
    array([[2.36, 0.0],
           [4.75, 0.3],
           [9.5 , 0.5],
           [19. , 0.2]])
    """

    psd_range = psd_unique_sorted.copy() if copy else psd_unique_sorted
    if passing:
        psd_range[:,1] = np.diff(psd_range[:,1], prepend = 0)
    else:
        psd_range[:,1] = np.abs(np.diff(psd_range[:,1], prepend = 1))
    return psd_range


def convert_range_to_cumulative(
    psd_unique_sorted: TwoColumnArrayType,
    *,
    passing: bool,
    copy: bool = True,
) -> TwoColumnArrayType:

    """
    Convert range (incremental) psd fractions to cumulative fractions.

    For range (incremental) passing or retained fractions, this function computes the
    cumulative fraction for each sieve.

    Parameters
    ----------
    psd_unique_sorted : np.ndarray[tuple[int, Literal[2]], np.dtype[np.number]]
        A 2D NumPy array of shape (N, 2), where the first column contains sieve sizes
        (sorted in ascending order and unique), and the second column contains range
        (incremental) passing or retained fractions.
    passing : bool
        If True, computes cumulative passing fractions as the cumulative sum of the range fractions.
        If False, computes cumulative retained fractions as the reverse cumulative sum of the range fractions.
    copy : bool, optional
        If True, returns a copy of the input array. If False, modifies the input array
        in place. Default is True.

    Returns
    -------
    psd_cumulative : np.ndarray[tuple[int, Literal[2]], np.dtype[np.number]]
        A 2D NumPy array of the same shape as the input, with the second column
        containing cumulative fractions.

    Examples
    --------
    >>> import numpy as np
    >>> psd_range = np.array([[2.36, 0.0], [4.75, 0.3], [9.5, 0.5], [19.0, 0.2]])
    >>> convert_range_to_cumulative(psd_range, passing=True)
    array([[2.36, 0.0],
           [4.75, 0.3],
           [9.5 , 0.8],
           [19. , 1.0]])
    """

    psd_cumulative = psd_unique_sorted.copy() if copy else psd_unique_sorted
    if passing:
        psd_cumulative[:,1] = psd_unique_sorted[:,1].cumsum()
    else:
        psd_cumulative[-1,1] = 0
        psd_cumulative[:-1,1] = psd_unique_sorted[:0:-1,1].cumsum()[::-1]
    return psd_cumulative


def normalise_psd(
    psd: TwoColumnArrayType,
    *,
    cumulative: bool,
    passing: bool,
    material_name: str | None = None,
    _prepend_zero: bool = False,
) -> TwoColumnArrayType:

    """
    Process and normalise the original PSD of a material, returning the result as the 
    normalised PSD.

    This function deduplicates and sorts the original particle size distribution (PSD) 
    data, optionally prepends a zero fraction at a small sieve size, normalises the data 
    as either a range or cumulative distribution, converts range to cumulative 
    distributions, and converts from retained to passing as needed. The normalised PSD 
    is validated and returned.

    The following internal helper functions are used:
        - `deduplicate_sort_sieves`: Ensures unique, sorted sieve sizes.
        - `check_range_psd`: Validates normalization and boundary values for range PSDs.
        - `check_cumulative_psd`: Validates monotonicity and boundary values for 
        cumulative PSDs.
        - `convert_range_to_cumulative`: Converts range (incremental) fractions to 
        cumulative fractions.
        - `convert_between_retained_and_passing`: Converts between passing and retained 
        forms.

    Parameters
    ----------
    psd : np.ndarray[tuple[int, Literal[2]], np.dtype[np.number]]
        The psd to normalise.
    cumulative : bool
        Whether the psd is cumulative.
    passing : bool
        Whether the psd is passing or retained.
    _prepend_zero : bool, optional
        If True, prepend a zero fraction at a small sieve size (default is False).

    Returns
    -------
    np.ndarray[tuple[int, Literal[2]], np.dtype[np.number]]
        A new array containing the normalised psd.

    Raises
    ------
    PSDError
        If the PSD data is invalid, contains duplicate sieves, is not properly 
        normalized, or fails monotonicity or boundary checks as enforced by the helper 
        functions.

    Notes
    -----
    - The normalised PSD is always validated for normalization and monotonicity according 
    to the specified parameters.
    - If `cumulative` is True, the PSD is normalised so that the last fraction is 1.
    - If `cumulative` is False, the PSD is normalised so that the first fraction is 0,
    then converted to a cumulative distribution.
    - If `passing` is False, the PSD is converted from passing to retained form (or vice 
    versa) as appropriate.
    - The `_prepend_zero` argument is intended for internal use and should generally be 
    left as False.

    Examples
    --------
    >>> import numpy as np
    >>> psd_original=np.array([[2.36, 0.0], [4.75, 0.3], [9.5, 0.7], [19.0, 1.0]])
    ... 
    >>> psd_normalised = normalise_psd(mat, cumulative = True, passing = True)
    >>> psd_normalised
    array([[2.36, 0.0],
           [4.75, 0.3],
           [9.5 , 0.4],
           [19. , 0.3]])
    """

    material_name = material_name or 'material'

    psd_unique_sorted = deduplicate_sort_sieves(
        psd = psd,
        material_name = material_name,
    )
    if _prepend_zero and psd_unique_sorted[0,1]:
        to_prepend = np.array(
            [psd_unique_sorted[0,0]/10, 0.],
            dtype = np.float32,
        )[None]
        psd_unique_sorted: TwoColumnArrayType = np.append(
            to_prepend, 
            psd_unique_sorted, 
            axis = 0,
        )
    if not cumulative:
        psd_unique_sorted[:,1] /= psd_unique_sorted[:,1].sum()
        check_range_psd(
            psd_unique_sorted=psd_unique_sorted, 
            material_name = material_name,
        )
        psd_cumulative = convert_range_to_cumulative(
            psd_unique_sorted, 
            passing = passing,
        )
    else:
        psd_unique_sorted[:,1] /= psd_unique_sorted[-int(passing),1]
        check_cumulative_psd(
            psd_unique_sorted = psd_unique_sorted,
            passing = passing,
            material_name = material_name,
        )
        psd_cumulative = psd_unique_sorted
        
    if not passing:
        convert_between_retained_and_passing(
            psd_unique_sorted = psd_cumulative,
            cumulative = True,
            copy = False,
        )
    return psd_cumulative


def reindex_psds[T: int](
    psds: Iterable[TwoColumnArrayType],
) -> tuple[
    np.ndarray[tuple[T], np.dtype[np.float32]],
    np.ndarray[tuple[T, int], np.dtype[np.float32]],
]:

    """
    Reindex multiple normalised PSDs onto a common set of unique sieve sizes.

    This function takes an iterable of normalised particle size distributions (PSDs), each 
    as a (n, 2) array of sieve sizes and fractions, and constructs:
        - A sorted array of all unique sieve sizes present across all PSDs.
        - A 2D array where each column corresponds to a PSD, and each row to a unique 
        sieve size.
          Fractions are aligned to the correct sieve size; missing values are filled with 
          NaN.

    All input PSDs must have been normalised and validated (deduplicated, sorted, 
    normalised, and checked for monotonicity/boundaries) prior to calling this function.

    Parameters
    ----------
    psds : Iterable of np.ndarray[tuple[int, Literal[2]], np.dtype[np.number]]
        An iterable of normalised PSD arrays, each of shape (n, 2), where the first column
        contains sorted, unique sieve sizes and the second column contains normalized 
        fractions.

    Returns
    -------
    all_unique_sieves : np.ndarray[tuple[T: int], np.dtype[np.float32]]
        A 1D array of all unique sieve sizes found across all input PSDs, sorted in 
        ascending order.
    psds_with_same_sieves : np.ndarray[tuple[T: int, int], np.dtype[np.float32]]
        A 2D array of shape (number of unique sieves, number of PSDs), where each column
        corresponds to a PSD and each row to a unique sieve size. Fractions are aligned 
        to the correct sieve size; missing values are filled with NaN.

    Notes
    -----
    - All input PSDs must be normalised using the provided helper functions (e.g., 
    `deduplicate_sort_sieves`, `check_range_psd`, `check_cumulative_psd`, etc.) before 
    being passed to this function. Use `add_normalised_psd` for safety.
    - The output can be used for further analysis or interpolation across a common sieve 
    grid. 
    - The function does not modify the input PSDs.

    Examples
    --------
    >>> import numpy as np
    >>> psd1 = np.array([[2.36, 0.0], [4.75, 0.3], [9.5, 0.7], [19.0, 0.0]])
    >>> psd2 = np.array([[4.75, 0.2], [9.5, 0.5], [19.0, 0.3]])
    >>> all_sieves, aligned_psds = reindex_psds([psd1, psd2])
    >>> print(all_sieves)
    [2.36 4.75 9.5 19.0]
    >>> print(aligned_psds)
    [[0.0  nan]
     [0.3 0.2]
     [0.7 0.5]
     [0.0 0.3]]
    """

    if not psds:
        return (
            np.empty(shape = (0,), dtype = np.float32),
            np.empty(shape = (0,0), dtype = np.float32),
        )

    grouped_psds = defaultdict(list)
    tuple_to_array_map = {}
    all_unique_sieves_set = set()
    for index, psd in enumerate(psds, start = 0):
        sieves_list = psd[:,0].tolist()
        all_unique_sieves_set.update(sieves_list)
        sieves_tuple = (*sieves_list,)
        grouped_psds[sieves_tuple].append([psd[:,1], index])
        if sieves_tuple not in tuple_to_array_map:
            tuple_to_array_map[sieves_tuple] = psd[:,0]
    else:
        n_materials = index

    all_unique_sieves = np.fromiter(
        all_unique_sieves_set,
        dtype = np.float32,
        count = len(all_unique_sieves_set),
    )
    all_unique_sieves.sort()

    psds_with_same_sieves = np.full(
        shape = (all_unique_sieves.shape[0], n_materials + 1),
        fill_value = np.nan,
        dtype = np.float32,
    )
    for sieves_tuple, psds_indices_list in grouped_psds.items():
        psds_list, indices_list = zip(*psds_indices_list)
        sieves = tuple_to_array_map[sieves_tuple]
        psds_array = np.column_stack(psds_list)
        insert_positions = all_unique_sieves.searchsorted(sieves, side = 'left')
        psds_with_same_sieves[insert_positions[:,None], [indices_list]] = psds_array

    return all_unique_sieves, psds_with_same_sieves



def fill_top_and_bottom_of_reindexed_range_psds_with_0[T: TwoDFloatArrayType](
    psds: T,
    copy: bool = True,
) -> T:

    """
    Fill the top and bottom of reindexed range PSDs with zeros where fractions are 
    inactive.

    This function sets to zero any values at the top (smallest sieves) and bottom 
    (largest sieves) of each column in a 2D array of reindexed range particle size 
    distributions (PSDs) where the cumulative sum is still zero. This ensures that 
    inactive sieve fractions at the boundaries are explicitly set to zero, which is 
    essential for interpolation.

    The input PSDs must be sorted by ascending sieve size and normalised as per the
    processing functions described above (deduplicated, sorted, normalised and 
    reindexed).

    Parameters
    ----------
    psds : np.ndarray[tuple[int, int], np.dtype[np.floating]]
        A 2D array of shape (number of sieves, number of PSDs), where each column is a
        reindexed range PSD aligned to a common set of sieves. May contain NaN for 
        missing values.
    copy : bool, optional
        If True (default), returns a new array with zeros filled in. If False, modifies 
        the input array in place.

    Returns
    -------
    psds_out : np.ndarray[tuple[int, int], np.dtype[np.number]]
        The PSD array with zeros filled at the top and bottom inactive regions for each 
        column.

    Notes
    -----
    - The function assumes that the input PSDs are sorted by ascending sieve size and are
      normalized (i.e., fractions sum to 1, with zeros at the boundaries where 
      appropriate).
    - Inactive regions are defined as those where the cumulative sum (from the top or 
    bottom) is still zero, indicating no material has yet been retained or passed.
    - NaN values are preserved except where replaced by zeros.

    Examples
    --------
    >>> import numpy as np
    >>> psds = np.array([
    ...     [0.1,    np.nan],
    ...     [0.3,    0.0   ],
    ...     [0.0,    0.5   ],
    ...     [np.nan, 0.0   ]
    ... ])
    >>> fill_top_and_bottom_of_reindexed_range_psds_with_0(psds)
    array([[0.1, 0. ],
           [0.3, 0. ],
           [0.0, 0.5],
           [0.0, 0. ]])
    """

    small_sieves_not_active_mask = np.nancumsum(psds, axis = 0) == 0
    large_sieves_not_active_mask = (np.nancumsum(psds[::-1], axis = 0) == 0)[::-1]
    fill_with_zero_mask = small_sieves_not_active_mask | large_sieves_not_active_mask
    if copy:
        return np.where(fill_with_zero_mask, 0, psds)
    else:
        psds[fill_with_zero_mask] = 0
        return psds


def fill_top_and_bottom_of_reindexed_cumulative_psds_with_0_and_1[T: TwoDFloatArrayType](
    psds: T,
    copy: bool = True,
) -> T:

    """
    Fill the top with zeros and the bottom with ones of reindexed cumulative PSDs where 
    fractions are inactive.

    This function sets to zero any values at the top (smallest sieves) and to one at any 
    values at the bottom (largest sieves) of each column in a 2D array of reindexed
    cumulative particle size distributions (PSDs) where the cumulative sum is still zero.
    This ensures that inactive sieve fractions at the boundaries are explicitly set to 
    zero, which is essential for interpolation.

    The input PSDs must be sorted by ascending sieve size and normalised as per the
    processing functions described above (deduplicated, sorted, normalised and 
    reindexed).

    Parameters
    ----------
    psds : np.ndarray[tuple[int, int], np.dtype[np.floating]]
        A 2D array of shape (number of sieves, number of PSDs), where each column is a
        reindexed cumulative PSD aligned to a common set of sieves. Should contain NaN 
        for missing values.
    copy : bool, optional
        If True (default), returns a new array with zeros filled in. If False, modifies 
        the input array in place.

    Returns
    -------
    psds_out : np.ndarray[tuple[int, int], np.dtype[np.number]]
        The PSD array with zeros filled at the top and bottom inactive regions for each 
        column.

    Notes
    -----
    - The function assumes that the input PSDs are sorted by ascending sieve size and are
      normalized (i.e., fractions sum to 1, with zeros at the boundaries where 
      appropriate).
    - Inactive regions are defined as those where the cumulative sum (from the top or 
    bottom) is still zero, indicating no material has yet been retained or passed.
    - NaN values are preserved except where replaced by zeros.

    Examples
    --------
    >>> import numpy as np
    >>> psds = np.array([
    ...     [0.1,    np.nan],
    ...     [0.3,    0.0   ],
    ...     [0.0,    0.5   ],
    ...     [np.nan, 0.0   ]
    ... ])
    >>> fill_top_and_bottom_of_reindexed_range_psds_with_0(psds)
    array([[0.1, 0. ],
           [0.3, 0. ],
           [0.0, 0.5],
           [0.0, 0. ]])
    """


    out = psds.copy() if copy else psds

    not_nan_mask = ~np.isnan(psds)
    nan_mask = ~not_nan_mask

    # First row per column where a real value > 0 appears
    has_positive = not_nan_mask & (psds > 0)
    first_positive_row = np.where(
        has_positive.any(axis=0),
        has_positive.argmax(axis=0),
        psds.shape[0],  # sentinel: no such row
    )

    # Last row per column where a real value < 1 appears
    has_not_one = not_nan_mask & (psds < 1)
    last_not_one_row = np.where(
        has_not_one.any(axis=0),
        psds.shape[0] - 1 - has_not_one[::-1].argmax(axis=0),
        -1,  # sentinel: no such row
    )

    rows = np.arange(psds.shape[0])[:, None]

    # Fill boundary NaNs
    out[nan_mask & (rows < first_positive_row)] = 0.0
    out[nan_mask & (rows > last_not_one_row)] = 1.0

    return out


def interpolate_reindexed_filled_psds[T: int, U: int](
    *,
    psds: np.ndarray[tuple[T, U], np.dtype[np.floating]],
    sieves: np.ndarray[tuple[T], np.dtype[np.floating]],
    cumulative: bool,
    copy: bool = True,
) -> np.ndarray[tuple[T, U], np.dtype[np.floating]]:

    """
    Interpolate missing values in reindexed, filled PSDs using PCHIP interpolation.

    This function fills NaN values in a 2D array of reindexed, filled range particle size
    distributions (PSDs) by interpolating along the sieve axis for each PSD (column).
    For each unique pattern of missing data, interpolation is performed using a monotonic
    piecewise cubic Hermite interpolating polynomial (PCHIP), ensuring smooth and 
    physically plausible results.

    The input PSDs must be aligned to a common set of sieves, sorted in ascending order,
    have zeros filled at the smallest inactive regions, and zeros or ones filled at the 
    largest inactive regions (as by `fill_top_and_bottom_of_reindexed_range_psds_with_0` 
    or `fill_top_and_bottom_of_reindexed_cumulative_psds_with_0_and_1`). Each column may
    contain NaNs where data was missing for a particular sieve.

    Parameters
    ----------
    psds : np.ndarray[tuple[T: int, U: int], np.dtype[np.floating]]
        A 2D array of shape (number of sieves, number of PSDs), where each column is a
        reindexed, filled PSD. NaN values indicate missing data to be interpolated.
    sieves : np.ndarray[tuple[T: int], np.dtype[np.floating]]
        A 1D array of sieve sizes, sorted in ascending order, corresponding to the rows 
        of `psds`.
    cumulative : bool
        Whether the psds are cumulative.
    copy : bool, optional
        If True (default), returns a new array with interpolated values. If False, 
        modifies the input array in place.

    Returns
    -------
    psds_interpolated : np.ndarray[tuple[T, U], np.dtype[np.floating]]
        A 2D array of the same shape as `psds`, with NaN values replaced by interpolated
        values using PCHIP. PSDs with no missing values are unchanged.

    Notes
    -----
    - The function groups PSDs by their pattern of missing data to optimize 
    interpolation.
    - Interpolation is performed only for columns with missing values; columns with no 
    NaNs are left unchanged.
    - The function assumes that the sieves are sorted in ascending order.
    - The monotonic PCHIP interpolator is used to avoid introducing spurious 
    oscillations.

    Examples
    --------
    >>> import numpy as np
    >>> psds = np.array([
    ...     [0.0   ,  0.0],
    ...     [0.3   ,  0.0],
    ...     [np.nan,  0.5],
    ...     [0.0   ,  0.0]
    ... ])
    >>> sieves = np.array([2.36, 4.75, 9.5, 19.0])
    >>> interpolate_reindexed_filled_range_psds(psds=psds, sieves=sieves, cumulative=False)
    array([[0. , 0. ],
           [0.3, 0. ],
           [0.2, 0.5],
           [0. , 0. ]])
    """

    psds_transposed = psds.T
    psds_transposed_notnan_mask = ~np.isnan(psds_transposed)
    psd_has_no_nans_mask = psds_transposed_notnan_mask.all(axis = 1)
    if psd_has_no_nans_mask.all():
        return psds
    psd_has_nans_mask = ~psd_has_no_nans_mask
    psds_to_interpolate = psds_transposed[psd_has_nans_mask]

    psds_to_interpolate_notnan_mask = psds_transposed_notnan_mask[psd_has_nans_mask]
    notnan_masks_unique, notnan_mask_index_order = np.unique(
        ar = psds_to_interpolate_notnan_mask, 
        axis = 0, 
        return_inverse = True, 
    )

    notnan_mask_grouping_order = notnan_mask_index_order.argsort()
    notnan_mask_inverse_grouping_order = notnan_mask_grouping_order.argsort()
    notnan_mask_index_order_sorted = notnan_mask_index_order[notnan_mask_grouping_order]
    psds_to_interpolate_ordered = psds_to_interpolate[notnan_mask_grouping_order]
    split_indices = (
        np.not_equal(
            notnan_mask_index_order_sorted[:-1],
            notnan_mask_index_order_sorted[1:],
        )
        .nonzero()[0]
        + 1
    )

    psds_to_interpolate_grouped = np.split(psds_to_interpolate_ordered, split_indices, axis = 0)

    interpolated_psds_list = []
    for notnan_mask, psds_to_interpolate_group in zip(notnan_masks_unique, psds_to_interpolate_grouped):
        known_sieves = sieves[notnan_mask]
        known_psds_group = psds_to_interpolate_group[:,notnan_mask]
        interpolator = PchipInterpolator(
            x = known_sieves,
            y = known_psds_group,
            axis = 1,
        )
        interpolated_psds_list.append(interpolator(sieves))

    interpolated_psds = np.concatenate(interpolated_psds_list, axis = 0)
    if copy:
        psds_to_return = np.empty_like(psds_transposed)
        psds_to_return[psd_has_no_nans_mask] = psds_transposed[psd_has_no_nans_mask]
    else:
        psds_to_return = psds_transposed
    
    psds_to_return[psd_has_nans_mask] = interpolated_psds[notnan_mask_inverse_grouping_order]
    if cumulative:
        np.clip(psds_to_return, 0.0, 1.0)
    else:
        psds_to_return /= psds_to_return.sum(axis = 1, keepdims = True)
    
    return psds_to_return.T


def get_psd_quantile_values[T: int, U: int, V: int, X: type[np.floating]](
    sieves: np.ndarray[tuple[T], np.dtype[np.number]],
    psds: np.ndarray[tuple[T, U], np.dtype[np.floating]],
    quantiles: np.ndarray[tuple[V], np.dtype[np.floating]],
    cumulative: bool,
    *,
    dtype: X = np.float32,
) -> np.ndarray[tuple[U, V], np.dtype[X]]:


    """
    Compute sieve sizes corresponding to specified PSD quantiles.

    This function returns the sieve sizes at which each particle size distribution (PSD)
    reaches the requested quantiles. If `cumulative` is True, the input PSDs are assumed
    to already contain cumulative fractions. If `cumulative` is False, the input PSDs are
    interpreted as range (incremental) fractions and are first converted internally to
    cumulative form.

    For each PSD and each quantile:
        - If the quantile exactly matches a cumulative fraction, the corresponding sieve
          size is returned directly.
        - Otherwise, the quantile value is estimated by linear interpolation between the
          two neighbouring sieve points that bracket the quantile.

    Parameters
    ----------
    sieves : np.ndarray[tuple[T], np.dtype[np.number]]
        A 1D NumPy array of sieve sizes, sorted in ascending order.
    psds : np.ndarray[tuple[T, U], np.dtype[np.floating]]
        A 2D NumPy array of shape (number of sieves, number of PSDs), where each column
        is a PSD aligned to `sieves`.
    quantiles : np.ndarray[tuple[V], np.dtype[np.floating]]
        A 1D NumPy array of quantiles to evaluate, expressed as fractions between 0 and
        1 inclusive (for example, 0.1 for D10, 0.5 for D50, and 0.9 for D90).
    cumulative : bool
        If True, `psds` is interpreted as containing cumulative passing fractions.
        If False, `psds` is interpreted as containing range fractions and is internally
        converted to cumulative fractions before evaluating quantiles.
    dtype : type[np.floating], optional
        Floating-point dtype of the returned array. Default is `np.float32`.

    Returns
    -------
    np.ndarray[tuple[U, V], np.dtype[X]]
        A 2D NumPy array of shape (number of PSDs, number of quantiles), where each row
        corresponds to a PSD and each column corresponds to a requested quantile.

    Raises
    ------
    PSDError
        If any quantile is less than 0 or greater than 1.
        If `cumulative` is True and the PSDs are not monotonically non-decreasing or do
        not reach 1 within tolerance.
        If `cumulative` is False and the range fractions do not sum to 1 within
        tolerance.

    Notes
    -----
    - The function assumes that `sieves` are sorted in ascending order.
    - For cumulative PSDs, each column is expected to represent a cumulative passing PSD.
    - For range PSDs, each column is expected to be normalised so that fractions sum to
      1.
    - Quantile values are obtained by linear interpolation in fraction–sieve space.

    Examples
    --------
    >>> import numpy as np
    >>> sieves = np.array([2.36, 4.75, 9.5, 19.0])
    >>> psds = np.array([
    ...     [0.0, 0.0],
    ...     [0.3, 0.2],
    ...     [0.8, 0.7],
    ...     [1.0, 1.0],
    ... ])
    >>> quantiles = np.array([0.1, 0.5, 0.9])
    >>> get_psd_quantile_values(sieves, psds, quantiles, cumulative=True)
    array([[ 3.156667,  6.65    , 14.75    ],
           [ 3.555   ,  7.464286, 15.833333]], dtype=float32)
    """

        
    if not (quantiles <= 1).all():
        raise PSDError('not all quantiles deceed 1.')
    elif not (quantiles >= 0).all():
        raise PSDError('not all quantiles exceed 0.')
    elif cumulative:
        max_to_1_check = np.isclose(psds.max(axis = 0), 1, rtol = 0, atol = 0.001)
        monotonicity_check = psds[1:] >= psds[:-1]
        if not max_to_1_check.all():
            raise PSDError('not all fractions deceed 1.')
        elif not monotonicity_check.all():
            raise PSDError('cumulative PSDs are not all monotonically increasing.')
        else:
            fractions_t_cum = psds.T
    else:
        sum_to_1_check = np.isclose(psds.sum(axis = 0), 1, rtol = 0, atol = 0.001)
        if not sum_to_1_check.all():
            raise PSDError('not all fractions sum to 1.')
        else:
            fractions_t_cum = psds.cumsum(axis = 0).T

    fractions_t_cum_3d, sieves_3d, quantiles_3d = np.broadcast_arrays(
        fractions_t_cum[:,None],
        sieves[None,None,:],
        quantiles[None,:,None],
    )
    fractions_t_quantiles_diffs = fractions_t_cum_3d - quantiles_3d
    fractions_t_quantiles_diffs_is_0 = fractions_t_quantiles_diffs == 0
    diffs_contain_0 = fractions_t_quantiles_diffs_is_0.any(axis = -1)
    diffs_contain_no_0 = ~diffs_contain_0
    quantile_results = np.empty(diffs_contain_0.shape, dtype)

    # handling cases where quantile is exactly present in fractions
    quantile_idx = fractions_t_quantiles_diffs_is_0[diffs_contain_0].argmax(axis = 1)
    quantile_results[diffs_contain_0] = sieves[quantile_idx]

    # handling cases where quantile is not exactly present, so requires interpolation
    fractions_t_quantiles_diffs_to_interpolate = fractions_t_quantiles_diffs[diffs_contain_no_0]
    right_idx = (fractions_t_quantiles_diffs_to_interpolate > 0).argmax(axis = 1)
    left_idx = right_idx - 1
    left_idx_lt_0 = left_idx < 0
    left_idx[left_idx_lt_0] = right_idx[left_idx_lt_0]

    # fitting a line between the two fractions wrt sieves to interpolate between them
    fractions_to_interpolate_between = fractions_t_cum_3d[diffs_contain_no_0]
    row_idx = np.arange(right_idx.shape[0])
    fractions_right = fractions_to_interpolate_between[row_idx,right_idx]
    fractions_left = fractions_to_interpolate_between[row_idx,left_idx]
    sieves_to_interpolate_between = sieves_3d[diffs_contain_no_0]
    sieves_right = sieves_to_interpolate_between[row_idx,right_idx]
    sieves_left = sieves_to_interpolate_between[row_idx,left_idx]
    gradients = (
        (fractions_left - fractions_right)
        /
        (sieves_left - sieves_right)
    )
    intercepts = fractions_right - (gradients * sieves_right)
    quantile_results[diffs_contain_no_0] = (
        (quantiles_3d[diffs_contain_no_0, 0] - intercepts)
        /
        gradients
    )
    return quantile_results


def match_psds[T: int, U: int](
    sieves: np.ndarray[tuple[T], np.dtype[np.floating]],
    target_psd: np.ndarray[tuple[T], np.dtype[np.floating]],
    component_psds: np.ndarray[tuple[T, U], np.dtype[np.floating]],
    component_bounds: np.ndarray[tuple[U, Literal[2]]],
) -> tuple[
        np.ndarray[tuple[U], np.dtype[np.float32]], 
        np.ndarray[tuple[T], np.dtype[np.float32]],
    ]:


    """
    Match a target PSD as a bounded weighted combination of component PSDs.

    This function solves an optimisation problem to determine the proportions of a set of
    component particle size distributions (PSDs) that best approximate a target PSD. The
    objective minimises the weighted integrated squared difference between the target PSD
    and the blended PSD across the sieve range, subject to lower and upper bounds on the
    component proportions and the constraint that the proportions sum to 1.

    The weighting is based on the logarithmic spacing of adjacent sieve sizes, so the
    objective approximates the area between curves on a log-sieve scale using the
    trapezium rule.

    Parameters
    ----------
    sieves : np.ndarray[tuple[T], np.dtype[np.floating]]
        A 1D NumPy array of sieve sizes, sorted in ascending order.
    target_psd : np.ndarray[tuple[T], np.dtype[np.floating]]
        A 1D NumPy array containing the target cumulative passing PSD evaluated at
        `sieves`.
    component_psds : np.ndarray[tuple[T, U], np.dtype[np.floating]]
        A 2D NumPy array of shape (number of sieves, number of components), where each
        column is a component cumulative passing PSD evaluated at `sieves`.
    component_bounds : np.ndarray[tuple[U, Literal[2]]]
        A 2D NumPy array of shape (number of components, 2), where each row contains the
        lower and upper bounds for the corresponding component proportion.

    Returns
    -------
    optimised_component_quantities : np.ndarray[tuple[U], np.dtype[np.float32]]
        A 1D NumPy array containing the optimised proportions of each component.
    optimised_psd : np.ndarray[tuple[T], np.dtype[np.float32]]
        A 1D NumPy array containing the blended PSD obtained from the optimised component
        proportions.

    Raises
    ------
    PSDError
        If the optimisation problem does not solve to an optimal solution, or if no
        solution is returned by the solver.

    Notes
    -----
    - All PSDs supplied to this function must be cumulative passing PSDs on the same
      sieve grid.
    - The optimisation variable is constrained to be non-negative and to lie within the
      bounds specified in `component_bounds`.
    - The component proportions are additionally constrained to sum to 1.
    - The optimisation is solved using `cvxpy` with the OSQP solver.

    Examples
    --------
    >>> import numpy as np
    >>> sieves = np.array([2.36, 4.75, 9.5, 19.0], dtype=np.float32)
    >>> target_psd = np.array([0.0, 0.25, 0.75, 1.0], dtype=np.float32)
    >>> component_psds = np.array([
    ...     [0.0, 0.0],
    ...     [0.1, 0.4],
    ...     [0.6, 0.9],
    ...     [1.0, 1.0],
    ... ], dtype=np.float32)
    >>> component_bounds = np.array([
    ...     [0.0, 1.0],
    ...     [0.0, 1.0],
    ... ], dtype=np.float32)
    >>> quantities, blended_psd = match_psds(
    ...     sieves=sieves,
    ...     target_psd=target_psd,
    ...     component_psds=component_psds,
    ...     component_bounds=component_bounds,
    ... )
    >>> quantities.shape
    (2,)
    >>> blended_psd.shape
    (4,)
    """


    import cvxpy as cp

    sieves = sieves.astype(np.float32)
    component_psds = component_psds.astype(np.float32)
    target_psd = target_psd.astype(np.float32)
    component_bounds = component_bounds.astype(np.float32)

    component_quantities = cp.Variable(
        name = 'quantities', 
        shape = component_bounds.shape[0],
        nonneg = True,
        bounds = [*component_bounds.T],
    )

    sieve_weights = np.log(sieves[1:] / sieves[:-1])
    if not np.isfinite(sieve_weights[0]):
        sieve_weights[0] = sieve_weights[1]

    current_psd = component_psds @ component_quantities
    square_diffs = (target_psd - current_psd)**2
    trapezium_areas = cp.multiply(
        (square_diffs[1:] + square_diffs[:-1]),
        sieve_weights,
    )
    to_minimise = cp.sum(trapezium_areas)

    problem = cp.Problem(
        objective = cp.Minimize(to_minimise),
        constraints = [
            cp.sum(component_quantities) == 1,
        ],
    )

    problem.solve(
        solver = 'OSQP',
        verbose = 2,
        eps_abs = 0.001,
        time_limit = 100,
    )

    if (problem.status != 'optimal') or (component_quantities.value is None):
        raise PSDError('optimisation failed.')

    optimised_component_quantities = component_quantities.value
    optimised_psd = component_psds @ optimised_component_quantities
    return optimised_component_quantities, optimised_psd