import copy

import numpy as np
import scipy.sparse.linalg as lina
from scipy import sparse as sp
import matplotlib.pyplot as plt
from skimage import io


def veclap(n, m):
    """

    :param n: Zeilen
    :param m: Spalten
    :return: vektorisierter Laplace-Operator für ein Bild der Größe n x m
    ''"""

    def D2(k):
        diagonals = [[-2 for _ in range(k)], [1 for _ in range(k - 1)],
                     [1 for _ in range(k - 1)]]  # hinten aus k k-1 gemacht
        d2 = sp.diags(diagonals, [0, -1, 1])
        return d2

    vl = sp.kron(sp.eye(m), D2(n)) + sp.kron(D2(m), sp.eye(n))
    return vl


def scaletoint8(array):
    """

    :param array: enthält reelle Zahlen
    :return: Runde ganzzahlig und schneide ab, was nicht in [0, 255] ist
    """
    return np.clip(array, 0, 255).astype(np.uint8)


def inner(array):
    """

    :param array: Matrix
    :return: Matrix mit weggestrichenen Randzeilen/spalten
    """
    n, m = array.shape
    return array[1: n - 1, 1: m - 1]


def removeinner(array):
    """

    :param array: Matrix
    :return: Behält nur die Werte auf dem Rand und setzt inneres auf 0
    """
    n, m = array.shape
    array_copy = copy.deepcopy(array)
    z = np.zeros((n - 2, m - 2))
    array_copy[1:n - 1, 1:m - 1] = z
    return array_copy


def vec(array):
    """

    :param array: Matrix
    :return: spaltenweise vektorisierte Matrix
    """
    n, m = array.shape
    return np.ndarray.flatten(array, 'F').reshape((n * m, 1))


def Dv(n):
    """

    :param n: Größe
    :return: Operator zum Vorwärtsdifferenzieren
    """
    return sp.diags([1, -1], [1, 0], shape=(n, n)).toarray()  # eventuell ohne toarray()


def Dr(n):
    """

    :param n: Größe
    :return: Operator zum Rückwärtsdifferenzieren
    """
    return sp.diags([-1, 1], [-1, 0], shape=(n, n)).toarray()


def gradient(array):
    """

    :param array: Matrix
    :return: Diskreter Gradient Einträge x, Diskreter Gradient Einträge y
    """
    n, m = array.shape
    x = np.matmul(Dv(n), array)
    y = np.matmul(array, Dv(m).transpose())
    return x, y


def div(array):
    """

    :param array: Matrix
    :return: Diskrete Divergenz nach Formel aus Übung
    """
    n, m = array.shape[0:2]
    return np.matmul(Dr(n), array[0:n, 0:m, 0]) + np.matmul(array[0:n, 0:m, 1], Dr(m).transpose())


def v(array_1, array_2):
    """

    :param array_1: Matrix
    :param array_2: Matrix
    :return: Vektor v wie in Aufgabe beschrieben
    """
    n, m = array_1.shape
    ve = np.zeros((n, m, 2))
    grad_f = gradient(array_1)
    grad_g = gradient(array_2)
    for i in range(n):
        for j in range(m):
            if grad_f[0][i, j] ** 2 + grad_f[1][i, j] ** 2 > grad_g[0][i, j] ** 2 + grad_g[1][i, j] ** 2:  # Norm
                # vergleichen
                ve[i, j, 0], ve[i, j, 1] = grad_f[0][i, j], grad_f[1][i, j]
            else:
                ve[i, j, 0], ve[i, j, 1] = grad_g[0][i, j], grad_g[1][i, j]
    return ve


def rightside(om, im, mode):
    """

    :param om: Pixelmatrix des Originalbildes im Ausschnitt der ersetzt wird
    :param im: Pixelmatrix des einzusetzenden Bildes
    :param mode: 'lap' für Laplace-Operator und 'div' für gemischte Gradienten als rechte Seite
    :return: rechte Seite des LGs, das der cg-solver bekommt
    """
    n, m = om.shape
    L_big = veclap(n, m)
    vec_g = vec(im)
    vec_outerf = vec(removeinner(om))

    corf = vec(inner(np.reshape((L_big * vec_outerf), (n, m), 'F')))  # Korrekturterm der rechten Seite für Rand von f

    if mode == 'lap':
        lapg = vec(inner(np.reshape((L_big * vec_g), (n, m), 'F')))  # Laplace g wird für das innere des
        # Ausschnitts berechnet und spaltenweise vektorisiert
        return lapg - corf
    elif mode == 'div':
        div_v = div(v(om, im))  # n x m
        return vec(inner(
            div_v)) - corf  # Div v wurde für das innere des Ausschnitts berechnet und spaltenweise vektorisiert,
        # f Korrekturterm wird abgezogen


def greycg(arorg, arins, pos=(10, 10), mode='lap'):
    """

    :param arorg: Pixelmatrix des Originalbildes(Graustufen!)
    :param arins: Pixelmatrix des einzusetzenden Bildes(Graustufen!)
    :param pos: Einsetzposition
    :param mode: 'lap' für Laplace-Operator und 'div' für gemischte Gradienten als rechte Seite
    :return: Array von (seamless cloning von arins in arorg an pos)
    """

    i, j = pos
    arorg = arorg.astype('int')  # uint8 Fehler umgehen
    arins = arins.astype('int')
    n, m = arins.shape

    croporg = arorg[i: i + n, j: j + m]  # zu bearbeitender Bildausschnitt

    L_small = veclap(n - 2, m - 2)

    rs = rightside(croporg, arins, mode)

    vec_ins = lina.cg(L_small, rs)[0]  # Löse Problem
    ins = np.reshape(vec_ins, (n - 2, m - 2), 'F')

    arorg[i + 1: i + n - 1, j + 1:j + m - 1] = ins  # an entsprechender Position ersetzen
    # io.imsave('test.jpg', scaletoint8(arorg))

    return scaletoint8(arorg)


def colorcg(orgpic, inspic, pos=(10, 10), mode='lap'):
    arins = io.imread(inspic)
    arorg = io.imread(orgpic)

    o_red, o_green, o_blue = np.dsplit(arorg, 3)  # Zerlegung in Farbkanäle
    i_red, i_green, i_blue = np.dsplit(arins, 3)

    o_red, o_green, o_blue = np.squeeze(o_red), np.squeeze(o_green), np.squeeze(
        o_blue)  # Dimension von n x m x 1 zu n x m
    i_red, i_green, i_blue = np.squeeze(i_red), np.squeeze(i_green), np.squeeze(i_blue)

    n_red = greycg(o_red, i_red, pos, mode)  # Lösung für einzelne Farbkanäle
    n_green = greycg(o_green, i_green, pos, mode)
    n_blue = greycg(o_blue, i_blue, pos, mode)

    new = np.dstack((n_red, n_green, n_blue))  # Verschmelzen der Farbkanäle zu einem Bild
    io.imsave('test.jpg', new)
    return new


def primitivecloning(orgpic, inspic, pos=(0, 0)):
    arins = io.imread(inspic)
    arorg = io.imread(orgpic)
    n, m = arins.shape[0:2]
    i, j = pos

    arorg[i: i + n, j:j + m] = arins
    # io.imsave('ptest.jpg', arorg)
    return arorg


def show_image_list(list_images, list_titles=None, figsize=(10, 10),  # Nur zur Veranschaulichung
                    title_fontsize=10):
    """
    Shows a grid of images, where each image is a Numpy array. The images can be either
    RGB or grayscale.
    Parameters:
    ----------
    images: list
        List of the images to be displayed.
    list_titles: list or None
        Optional list of titles to be shown for each image.
    figsize: tuple of width, height
        Value to be passed to pyplot.figure()
    title_fontsize: int
        Value to be passed to set_title().
    """

    num_images = len(list_images)
    num_cols = 3
    num_rows = 2

    # Create a grid of subplots.
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)

    # Create list of axes for easy iteration.
    if isinstance(axes, np.ndarray):
        list_axes = list(axes.flat)
    else:
        list_axes = [axes]

    for i in range(num_images):
        img = list_images[i]
        title = list_titles[i] if list_titles is not None else 'Image %d' % i
        cmap = 'gray'

        list_axes[i].imshow(img, cmap=cmap)
        list_axes[i].set_title(title, fontsize=title_fontsize)

    for i in range(num_images, len(list_axes)):
        list_axes[i].set_visible(False)

    fig.tight_layout()
    _ = plt.show()


def solution():
    print(
        'Es werden 2 Fenster geöffnet. Eins beinhaltet die bearbeiteten Bilder und das andere die grafische '
        'Darstellung des 5 x 7 vektorisierten Laplace-Operators.')
    sol_data = []
    sol_data = sol_data + [colorcg('water.jpg', 'bear.jpg', (10, 10)),
                           colorcg('water.jpg', 'bear.jpg', (10, 10), 'div'),
                           primitivecloning('water.jpg', 'bear.jpg', (10, 10))]
    sol_data = sol_data + [colorcg('bird.jpg', 'plane.jpg', (10, 400)),
                           colorcg('bird.jpg', 'plane.jpg', (10, 400), 'div'),
                           primitivecloning('bird.jpg', 'plane.jpg', (10, 400))]
    plt.imshow(veclap(5, 7).toarray(), cmap='gray')

    show_image_list(sol_data, ['Laplace-Operator', 'gemischte Gradienten', 'primitives Einfügen', '', '', ''])


solution()
