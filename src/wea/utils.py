"""
Copyright(c) 2016: Éric Thiébaut https: // github.com/emmt

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files(the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and / or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.
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
    cld = np.ceil(add / b_val) * b_val
    return int(cld)
