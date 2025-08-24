import basics as bs
import numpy
import task1
from task2 import show_image_list


def wghtmedfilter(picture, weights):
    """

    :param picture: Eingabebild
    :param weights: Gewichtsmatrix
    :return: Array mit Pixeln des bearbeiteten Bildes
    """
    s = int((weights.shape[0] - 1) / 2)
    picarr = bs.img2nparr(picture)
    m, n = picarr.shape
    conpic = bs.contpic(picarr, s)
    filteredpic = numpy.zeros((m, n))

    for i in range(m):
        for j in range(n):
            filteredpic[i, j] = task1.weightedmedian(bs.partofarr(conpic, (i, j), (i + 2*s + 1, j+2*s+1)), weights)       # gewichte entsprechenden Bildausschnitt

    return filteredpic


def showfilteredmed(sigma, s):
    """

    :param sigma: sigma für Gaußfilter
    :param s: Größe des Bereichs aus dem der Median gebildet wird
    :return: None, zeigt Originalbilder und gefilterte Bilder an
    """
    gaussdata = []
    evendata = []               # GLEICHGEWICHTET
    original = []

    for pic in ('B1.png', 'B2.png', 'C.png'):
        gaussdata.append(wghtmedfilter(pic, bs.gaussian(sigma)))
        evendata.append(wghtmedfilter(pic, bs.sqareavg(s)))
        original.append(bs.img2nparr(pic))
    imagedata = original + gaussdata + evendata

    show_image_list(imagedata, ['Original B1', 'Original B2', 'Original C', 'Gauss B1', 'Gauss B2', 'Gauss C',
                                'Gleich B1', 'Gleich B2', 'Gleich C'])
