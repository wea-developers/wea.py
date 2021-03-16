import numpy as np


def checkdims(dims: tuple):
    """
    Calculate the total number of elements

    :param dims: Dimension tuple
    :type dims: tuple
    :return: Number of elements
    :rtype: int
    """
    number: int = 1

    def multipy(dim: int):
        nonlocal number
        number = number * dim
    [multipy(dim) for dim in dims]
    return int(number)


def roundup(a: int, b: int):
    """
    Calculate the biggest buffer depending on b

    : param a: Parameter to check
    : type a: int
    : param b: Reference to check against
    : type b: int
    """
    add = a + (b - 1)
    cld = np.ceil(add / b)*b
    return int(cld)
