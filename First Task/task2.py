import task1
import basics as bs
import numpy
import matplotlib.pyplot as plt


def wghtaverfilter(picture, weights):
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
            filteredpic[i, j] = task1.weightedaverage(bs.partofarr(conpic, (i, j), (i + 2*s + 1, j+2*s+1)), weights)       # gewichte entsprechenden Bildausschnitt

    return filteredpic


# not needed
def showweights(weights):
    """

    :param weights: Gewichtsmatrix
    :return: None, shows weigths of matrix as picture where higher weights have more brightness
    """
    plt.imshow(weights, cmap='gray')
####


def show_image_list(list_images, list_titles=None, figsize=(10, 10),
                    title_fontsize=10):
    '''
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
    '''

    num_images = len(list_images)
    num_cols = 3
    num_rows = 4

    # Create a grid of subplots.
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)

    # Create list of axes for easy iteration.
    if isinstance(axes, numpy.ndarray):
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


def showfilteredave(sigma, s):
    gaussdata = []
    squaredata = []
    original = []

    for pic in ('B1.png', 'B2.png', 'C.png'):
        gaussdata.append(wghtaverfilter(pic, bs.gaussian(sigma)))
        squaredata.append(wghtaverfilter(pic, bs.sqareavg(s)))
        original.append(bs.img2nparr(pic))
    imagedata = original + gaussdata + squaredata
    imagedata.append(bs.gaussian(sigma))

    show_image_list(imagedata, ['Original B1', 'Original B2', 'Original C', 'Gauss B1', 'Gauss B2', 'Gauss C',
                                'Rechteck B1', 'Rechteck B2', 'Rechteck C', 'Gewichte Gauss-Filter'])
