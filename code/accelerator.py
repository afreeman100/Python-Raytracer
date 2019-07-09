import math
import numpy as np
from numba import jit


@jit(nopython=True)
def normalize(vector):
    length = len(vector)

    mag = 0
    for i in range(length):
        mag += vector[i] * vector[i]
    mag = math.sqrt(mag)

    for i in range(length):
        vector[i] = vector[i] / mag

    return vector


@jit(nopython=True)
def cross(vec1, vec2):
    result = np.zeros(3)
    result[0] = vec1[1] * vec2[2] - vec1[2] * vec2[1]
    result[1] = vec1[2] * vec2[0] - vec1[0] * vec2[2]
    result[2] = vec1[0] * vec2[1] - vec1[1] * vec2[0]
    return result


# The same functions but using the numpy library  - these are slower than the above implementations
def normalize_(vector):
    return vector/np.linalg.norm(vector)


def cross_(a, b):
    return np.cross(a, b)
