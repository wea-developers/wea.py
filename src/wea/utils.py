"""
Some helper utilities
"""
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
    for dim in dims:
        multipy(dim)
    return int(number)


def roundup(a_val: int, b_val: int):
    """
    Calculate the biggest buffer depending on b

    : param a_val: Parameter to check
    : type a_val: int
    : param b_val: Reference to check against
    : type b_val: int
    """
    add = a_val + (b_val - 1)
    cld = np.ceil(add / b_val)*b_val
    return int(cld)
