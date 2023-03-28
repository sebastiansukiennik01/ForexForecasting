"""
Module with functions for generating different processes e.g. autoregressive, stationary, non-stationary, etc.
"""
import numpy as np


def autoregressive(
    n: int, seed: int = 0, start=0, order=1, ro: list = [1], c: int = 0
) -> np.array:
    """
    Generates example of autoregressive process.
    args:
        n : number of elements in process
        seed : integer seed represenetation for random generator
        start : first starting point from which process will be generated, and will oscilate aruond
        order : order of autoregression process
        ro : list of parameterers, where i-th parameter is multiplied by i-th lagged AR value
    return : array of values in generate AR process
    """
    _check_input(order=order, ro=ro)

    np.random.seed(seed)
    e = list(np.random.randn(n))
    ar = [start + e[0]]

    for i in range(1, n):
        if len(ar) >= order:
            y_order = ar[::-1][:order]
            element = c + sum(np.multiply(y_order, ro)) + e[i]
        else:
            element = c + sum(np.multiply(ar[::-1], ro[: len(ar)])) + e[i]

        ar.append(element)

    return ar


def moving_average(
    n: int, seed: int = 0, start=0, order=1, ro: list = [1], c: int = 0
) -> np.array:
    """
    Generates example of moving average process.
    args:
        n : number of elements in process
        seed : integer seed represenetation for random generator
        start : first starting point from which process will be generated, and will oscilate aruond
        order : order of calculating weighted average
        ro : list of parameterers, where i-th parameter is multiplied by i-th lagged MA error
    return : array of values in generate MA process
    """
    _check_input(order=order, ro=ro)

    np.random.seed(seed)
    e = list(np.random.randn(n))
    ar = [c + e[0]]

    for i in range(1, n):
        if i - order >= 0:
            e_order = e[i - order : i][::-1]
            element = c + np.matmul(ro, e_order) + e[i]
        else:
            e_order = e[:i][::-1]
            element = c + np.matmul(ro[: len(e_order)], e_order) + e[i]

        ar.append(element)

    return ar


def arima(
    n: int,
    p: int,
    d: int,
    q: int,
    ar_ro: list,
    ma_ro: list,
    seed: int = 0,
    start: int = 0,
    c: int = 0,
) -> np.array:
    """
    Generates example of autoregresive integrated moving average process.
    args:
        n : number of elements in process
        p : order of autoregressive part
        d : order of differencing in autoregressive part
        q : order of moving average part
        ar_ro : list of AR parameterers, where i-th parameter is multiplied by i-th lagged AR error
        ma_ro : list of MA parameterers, where i-th parameter is multiplied by i-th lagged MA error
        seed : integer seed represenetation for random generator
        start : first starting point from which process will be generated, and will oscilate aruond
        c : constant in AR and MA

    return : array of values in generate ARIMA process
    """
    ar = autoregressive(n=n, seed=seed, start=start, order=p, ro=ar_ro, c=c)
    ma = moving_average(n=n, seed=seed, start=start, order=q, ro=ma_ro, c=c)
    arima = np.add(ar, ma)

    return arima


def _check_input(order: int, ro: list) -> None:
    """
    Check if inputs for autoregressive and moving_average processes are correct.
    """
    assert (
        len(ro) == order
    ), f"Length of parameters list must be the same as the order! \
        Got {len(ro)} and {order}"
    assert _check_ro(
        ro=ro
    ), "Ro paramters do not follow restrictions. Revise provided parameters!"


def _check_ro(ro: list) -> bool:
    """
    Asserts if ro parameters for autoregressive/moving_average are correct.
    """
    if len(ro) == 1 and (-1 < ro[0] <= 1):
        return True
    elif (
        len(ro) == 2
        and (-1 < ro[1] < 1)
        and (ro[0] + ro[1] < 1)
        and (ro[1] - ro[0] < 1)
    ):  # AR(2)
        return True
    elif (
        len(ro) == 2
        and (-1 < ro[1] < 1)
        and (ro[0] + ro[1] > -1)
        and (ro[0] - ro[1] < 1)
    ):  # MA(2)
        return True
    elif len(ro) > 3 or len(ro) < 1:
        return True

    return False
