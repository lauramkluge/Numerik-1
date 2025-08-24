import copy

import numpy as np


def weightedaverage(matrix, weights):
    """

    :param matrix: beliebige Matrix in R(n,m) als numpy array
    :param weights: Gewichtsmatrix in R+(n,m) und Eintragssumme=1
    :return: gewichteter Mittelwert
    ''"""

    s = 0
    n = matrix.shape[0]
    m = matrix.shape[1]

    for k in range(n):
        for l in range(m):
            s += weights[k, l] * matrix[k, l]
    return s


def min_index(A):
    rowlength = A.shape[1]
    listmin = A.argmin()
    x = listmin // rowlength
    y = listmin % rowlength
    return x, y


def weightedmedian(inmatrix, weights):
    """

    :param inmatrix: beliebige Matrix in R(n,m) als numpy array
    :param weights: Gewichtsmatrix in R+(n,m) und Eintragssumme=1
    :return: gewichteter Median (Median aller Einträge)
    """
    matrix = copy.deepcopy(inmatrix)
    checked = np.amax(matrix) + 1
    s = 0
    nextlargest = 0

    while s < 0.5:
        imin, jmin = min_index(matrix)       # Koordinaten des nächstkleinsten Elements
        nextlargest = matrix[imin, jmin]
        matrix[imin, jmin] = checked                # Element will not get looked at again
        s += weights[imin, jmin]
    if s == 0.5:
        return 0.5 * (nextlargest + np.amin(matrix))
    else:
        return nextlargest


def comparision(n, m):
    """

    :param n: Zeilenzahl der Vergleichsmatrix
    :param m: Spaltenzahl der Vergleichsmatrix
    :return: Abweichung (vom Mittelwert, vom Median)
    """
    A = np.random.uniform(-100, 100, (n, m))    # creates array
    W = np.ones((n, m)) * (1/(n*m))              # gleichgewichtet

    mediandif = abs(weightedmedian(A, W) - np.median(A))
    averagedif = abs(weightedaverage(A, W) - np.mean(A))

    return averagedif, mediandif


def hundretcomparisions():
    """

    :return: maximale Abweichung der Numpy Funktionen von eigenem (Mittelwert, Median)
    """
    comptuple = [0, 0]

    for i in range(100):
        n, m = np.random.randint(1, 100, 2)
        x, y = comparision(n, m)

        if x > comptuple[0]:        # größere Abweichung im Mittelwert gefunden
            comptuple[0] = x                # dann merke diese
        if y > comptuple[1]:
            comptuple[1] = y

    return comptuple
