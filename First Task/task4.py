import basics as bs
import numpy
import math

from task2 import show_image_list


def ws(x, y, s):
    '''

    :param x: Variable der Gewichtsfunktion
    :param y: Variable der Gewichtsfunktion
    :param s: Variable der Gewichtsfunktion
    :return: Wert der Gewichtsfunktion w_s
    '''
    return math.exp(-(x ** 2 + y ** 2) / (2 * s ** 2))


def wr(x, r):
    """

    :param x: Variable der Gewichtsfunktion
    :param r: Variable der Gewichtsfunktion
    :return: Wert der Gewichtsfunktion w_r
    """
    return math.exp(-(x ** 2) / (2 * r ** 2))


def bilateralfilter(pic, sigr, sigs, s):
    """

    :param pic: zu bearbeitendes Bild
    :param sigr: signum_r Wert für Bearbeitung ( zB 75), berücksichtigt Abstand der Grauwerte
    :param sigs: signum_s Wert für Bearbeitung ( zB 3), berücksichtigt Abstand der Pixel
    :param s: Größe des Bereichs aus dem Farbwert gebildet wird
    :return: numpy array des bearbeiteten Bildes
    """
    picarr = bs.img2nparr(pic)
    m, n = picarr.shape
    conpic = bs.contpic(picarr, s)
    filteredpic = numpy.zeros((m, n))

    for k in range(m):
        for l in range(n):
            summe = 0
            teiler = 0
            for i in range(-s + k, s + 1 + k):
                for j in range(-s + l, s + 1 + l):
                    summe += ws(k - i, l - j, sigs) * wr(picarr[k, l] - conpic[i + s, j + s], sigr) * conpic[
                        i + s, j + s]
            for u in range(-s + k, s + 1 + k):
                for v in range(-s + l, s + 1 + l):
                    teiler += ws(k - u, l - v, sigs) * wr(picarr[k, l] - conpic[u + s, v + s], sigr)
            filteredpic[k, l] = summe / teiler

    return filteredpic.astype('uint8')


def showfilteredbil(sigs, sigr, s):
    """

    :param sigs: signum_s Wert für Bearbeitung ( zB 3), berücksichtigt Abstand der Pixel
    :param sigr: signum_r Wert für Bearbeitung ( zB 75), berücksichtigt Abstand der Grauwerte
    :param s: Größe des Bereichs aus dem Farbwert gebildet wird
    :return: None, zeigt Originalbilder und gefilterte Bilder an
    """
    bildata = []
    original = []

    for pic in ('B1.png', 'B2.png', 'C.png'):
        bildata.append(bilateralfilter(pic, sigr, sigs, s))
        original.append(bs.img2nparr(pic))
    imagedata = original + bildata

    show_image_list(imagedata, ['Original B1', 'Original B2', 'Original C', 'Bilateral B1',
                                'Bilateral B2', 'Bilateral C'])
