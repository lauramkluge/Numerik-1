import math

import numpy
from skimage import io


def img2nparr(image):
    """

    :param image: Bilddatei
    :return: Array der Bildpixel
    """
    array = io.imread(image)
    return array


def arr2svdpic(picarr, name):
    """

    :param picarr: Array mit Bildpixeln
    :param name: gewünschter Speichername
    :return: None, Funktion speichert Bild unter name
    """
    picarr = picarr.astype('uint8')
    io.imsave(name, picarr)


def contpic(picarr, s):
    """

    :param picarr: Array von Bildpixeln
    :param s: gewünschte Randdicke
    :return: Setzt picarr sinnvoll fort auf Rand (Ziehe Randpixel in die Länge nach außen, in den Ecken der Eckpixel)
    """
    m, n = picarr.shape
    bigpic = numpy.zeros((m + 2 * s, n + 2 * s))
    for i in range(m):
        for j in range(n):
            bigpic[i + s, j + s] = picarr[i, j]
    for i in range(s):  # oberer Rand spaltenweise fortgesetzt mit letztem pixel
        for j in range(n):
            bigpic[i, j + s] = picarr[0, j]
            bigpic[s + m + i, j + s] = picarr[m - 1, j]  # unterer Rand
    for i in range(m):
        for j in range(s):
            bigpic[i + s, j] = picarr[i, 0]  # linker und rechter Rand
            bigpic[i + s, s + n + j] = picarr[i, n - 1]
    for i in range(s):  # Ecken
        for j in range(s):
            bigpic[i, j] = picarr[0, 0]
            bigpic[i, j + n + s] = bigpic[0, n - 1]
            bigpic[i + m + s, j] = picarr[m - 1, 0]
            bigpic[i + m + s, j + n + s] = picarr[m - 1, n - 1]
    return bigpic


def gaussian(sigma):
    """

    :param sigma: Bestimmt Größe des Gaußfilters
    :return: Array mit Gaußfiltergewichten
    """
    M = int(3 * (sigma // 1))  # sigma abrunden und mal 3
    G = numpy.zeros((2 * M + 1, 2 * M + 1))
    for k in range(-M, M + 1):
        for l in range(-M, M + 1):
            G[k + M, l + M] = math.exp(-1 * (k ** 2 + l ** 2) / (2 * sigma ** 2))
    G = G / numpy.sum(G)
    return G


def sqareavg(s):
    """

    :param s: Bestimmt Größe des Rechteckfilters
    :return: Gewichtearray des Rechteckfilters
    """
    W = numpy.zeros((2 * s + 1, 2 * s + 1))
    ave = 1 / (2 * s + 1) ** 2
    for k in range(-s, s + 1):
        for l in range(-s, s + 1):
            W[k + s, l + s] = ave
    return W


def partofarr(array, upperleft, lowerright):
    """

    :param array: Großes Inputarray
    :param upperleft: linke obere Ecke des Teilarrays
    :param lowerright: rechte untere Ecke des Teilarray
    :return: gibt gewünschtes Teilarray aus
    """
    return array[upperleft[0]:(lowerright[0]), upperleft[1]:(lowerright[1])]