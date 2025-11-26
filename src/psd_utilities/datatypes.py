from typing import Literal

import numpy as np

type TwoDFloatArrayType = np.ndarray[tuple[int, int], np.dtype[np.floating]]
type TwoColumnArrayType = np.ndarray[tuple[int, Literal[2]], np.dtype[np.number]]