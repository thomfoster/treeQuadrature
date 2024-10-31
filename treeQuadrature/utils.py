import numpy as np
from typing import Any, Dict, Union
from collections.abc import MutableMapping

from .container import Container


def scale(ys):
    """
    Scale a numpy array so that all values are between 0 and 1

    Parameter
    ----------
    ys : 1-d array
        the array to be scaled

    Return
    ------
    scaled array of the same shape as ys
    """
    ys = np.array(ys)
    high = np.max(ys)
    low = np.min(ys)

    if high == low:
        raise Exception("cannot scale an array with all entries the same")

    return (ys - low) / (high - low)


def handle_bound(value, D, default_value) -> np.ndarray:
    if value is None:
        return np.array([default_value] * D)
    elif isinstance(value, (int, float)):
        return np.array([float(value)] * D)
    elif isinstance(value, (list, np.ndarray)) and len(value) == D:
        return np.array(value)
    else:
        raise ValueError(
            "bound must be a float, list, or numpy.ndarray "
            f"with length {D} when given as a list or numpy.ndarray. \n"
            f"got {type(value).__name__} with length {len(value)}"
        )


class ResultDict(MutableMapping):
    """
    A dictionary-like object designed for integrators \n
    Only accepts float values for the 'estimate' key.
    """

    def __init__(self, estimate: float, **kwargs):
        """
        Parameters
        ----------
        estimate : float
            The estimate of the integral.
        **kwargs : Any
            Other keys and values to be added to the dictionary. \n
            Not compulsory, but can be used to add other keys like: \n
            - n_evals (int): The number of evaluations.
            - std (float): The standard deviation of the estimate. \n
            For tree integrators:
            - containers (List[Container]):
                The containers used in the integration process.
            - contributions (List[float]):
                The contributions of each container to the estimate.
            - stds (List[float]):
                The standard deviations of the contributions.

        Example
        -------
        >>> result = ResultDict(estimate=1.0, n_evals=100)
        >>> result['estimate']
        1.0
        >>> # Adding estimate as a string will raise a TypeError
        >>> try:
        >>>    result['estimate'] = '1.0'
        >>> except TypeError as e:
        >>>    print(e)
        'estimate' must be a float, got str
        """
        if not isinstance(estimate, float):
            raise TypeError(
                f"'estimate' must be a float, got {type(estimate).__name__}"
            )
        self._data: Dict[str, Any] = {"estimate": estimate}
        self.update(kwargs)  # Add other keys if provided

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        if key == "estimate":
            if not isinstance(value, float):
                raise TypeError(
                    f"'estimate' must be a float, got {type(value).__name__}"
                )
        elif key == "n_evals":
            if not isinstance(value, int):
                raise TypeError("'n_evals' must be an int, "
                                f"got {type(value).__name__}")
        elif key == "containers":
            if not isinstance(value, list) or not all(
                isinstance(v, Container) for v in value
            ):
                raise TypeError(
                    "'containers' must be a List[Container], "
                    f"got {type(value).__name__}"
                )
        elif key == "contributions":
            if not isinstance(value, list) or not all(
                isinstance(v, float) for v in value
            ):
                raise TypeError(
                    "'contributions' must be a List[float], "
                    f"got {type(value).__name__}"
                )
        elif key == "stds":
            if not isinstance(value, list) or not all(
                isinstance(v, float) for v in value
            ):
                raise TypeError(
                    "'stds' must be a List[float], "
                    f"got {type(value).__name__}"
                )
        elif key == "std":
            if not isinstance(value, float):
                raise TypeError("'std' must be a float, "
                                f"got {type(value).__name__}")

        self._data[key] = value

    def __delitem__(self, key: str) -> None:
        if key == "estimate":
            raise KeyError("'estimate' key cannot be deleted")
        del self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return repr(self._data)

    def update(self, other: Union[Dict[str, Any], "ResultDict"]) -> None:
        for key, value in other.items():
            self[key] = value

    def setdefault(self, key: str, default: Any = None) -> Any:
        if key == "estimate" and not isinstance(default, float):
            raise TypeError("'estimate' must be a float, "
                            f"got {type(default).__name__}")
        return self._data.setdefault(key, default)


def integrand_for_cuba(integrand_func):
    """
    Creates an Integrand function in the format expected by Cuba.

    Parameters
    ----------
    integrand_func : callable
        A function that takes a numpy.ndarray of shape (N, D)
        and returns an array of shape (N, 1).
        This is the usual form of TQ package.
    
    Returns
    -------
    Integrand : function
        A function compatible with Cuba's expected integrand format.
    """
    def Integrand(ndim, xx, ncomp, ff, userdata):
        # Convert `xx` (a ctypes array) to a numpy array for easier manipulation
        xx_array = np.ctypeslib.as_array(xx, shape=(ndim.contents.value,))
        xx_reshaped = xx_array.reshape(1, -1)  # Reshape to (1, D) for compatibility

        # Compute the function value using `integrand_func`
        result = integrand_func(xx_reshaped)

        # Assign the first result to `ff[0]`
        ff[0] = result[0]

        return 0

    return Integrand


def compile_cuba_results(results):
    """
    Takes the results from cuba integrators
    and convert that to ResultDict used by
    TreeQuadrature.
    """
    contributions = []
    errors = []
    fitness = []

    for comp in results['results']:
        contributions.append(comp['integral'])
        errors.append(comp['error'])
        fitness.append(comp['prob'])

    resultDict = ResultDict(estimate=sum(contributions))
    resultDict['n_evals'] = results['neval']
    resultDict['contributions'] =contributions
    resultDict['stds'] = errors
    resultDict['chi2_fitness'] = fitness

    return resultDict
