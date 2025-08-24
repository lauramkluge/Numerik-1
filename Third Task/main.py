import numpy as np
from matplotlib import pyplot as plt

imgs = np.fromfile('train-images-idx3-ubyte', dtype=np.uint8)
imgs = np.reshape(imgs[16:], [-1, 28, 28])
labs = np.fromfile('train-labels-idx1-ubyte', dtype=np.uint8)
labs = labs[8:]


def empirical_mittelwert(imgs):
    """

    Parameters
    ----------
    imgs : The set of training pictures as 28x28 matrices

    Returns
    -------
    Sum : the mean matrix of size 28x28 over this set.

    """
    Sum = 0
    for i in range(np.shape(imgs)[0]):
        Sum += imgs[i]
    Sum = Sum * (1 / np.shape(imgs)[0])
    return Sum


def empirical_variance(imgs):
    """

    Parameters
    ----------
    imgs : The set of training pictures as 28x28 matrices

    Returns
    -------
    V : the variance matrix of size 28x28 over this set

    """
    V = 0
    b = empirical_mittelwert(imgs)
    for i in range(np.shape(imgs)[0]):
        V += np.square(imgs[i] - b)
    V = V * (1 / np.shape(imgs)[0])
    return V


def position(imgs, labs):
    """

    Parameters
    ----------
    imgs : The set of training pictures as 28x28 matrices
    labs : The set of digits labels of the training set as list of integers 0 to 9

    Returns
    -------
    outp : returns a list of lists containing in order the indices of matrices
    corresponding to digits 0 to 9.

    """
    outp = []
    for i in range(10):
        outp.append(np.where(labs == i)[0])
    return outp


def class_means(imgs, labs):
    """

    Parameters
    ----------
    imgs : The set of training pictures as 28x28 matrices
    labs : The set of digits labels of the training set as list of integers 0 to 9

    Returns
    -------
    outp : a list of matrices as np.arrays corresponding to the mean class of
    every digit in order 0 to 9.

    """
    outp = []
    labels = position(imgs, labs)
    for i in labels:
        class_list = []
        for j in i:
            class_list.append(imgs[j])
        outp.append(empirical_mittelwert(class_list))

    return outp


def class_variances(imgs, labs):
    """

    Parameters
    ----------
    imgs : The set of training pictures as 28x28 matrices
    labs : The set of digits labels of the training set as list of integers 0 to 9

    Returns
    -------
    outp : a list of matrices as np.arrays corresponding to the variance class of
    every digit in order 0 to 9.

    """
    outp = []
    labels = position(imgs, labs)
    for i in labels:
        class_list = []
        for j in i:
            class_list.append(imgs[j])
        outp.append(empirical_variance(class_list))

    return outp


def viz_mean_var(N=100):
    """

    Parameters
    ----------
    N : first N-elements who's mean and variance want to be visualized.
    The default is 100.

    Returns
    -------
    plot of means and variances of N-first training pictures of all 10 digits.

    """
    a = class_means(imgs, labs[:N])
    b = class_variances(imgs, labs[:N])
    fig = plt.figure()

    ax1 = fig.add_subplot(10, 2, 1)
    ax1.imshow(a[0], cmap='gray')

    ax1.set_title('Mittelwert')

    ax2 = fig.add_subplot(10, 2, 2)
    ax2.imshow(b[0], cmap='gray')

    ax2.set_title('Varianz')

    ax1 = fig.add_subplot(10, 2, 3)
    ax1.imshow(a[1], cmap='gray')
    ax2 = fig.add_subplot(10, 2, 4)
    ax2.imshow(b[1], cmap='gray')
    ax1 = fig.add_subplot(10, 2, 5)
    ax1.imshow(a[2], cmap='gray')
    ax2 = fig.add_subplot(10, 2, 6)
    ax2.imshow(b[2], cmap='gray')
    ax1 = fig.add_subplot(10, 2, 7)
    ax1.imshow(a[3], cmap='gray')
    ax2 = fig.add_subplot(10, 2, 8)
    ax2.imshow(b[3], cmap='gray')
    ax1 = fig.add_subplot(10, 2, 9)
    ax1.imshow(a[4], cmap='gray')
    ax2 = fig.add_subplot(10, 2, 10)
    ax2.imshow(b[4], cmap='gray')
    ax1 = fig.add_subplot(10, 2, 11)
    ax1.imshow(a[5], cmap='gray')
    ax2 = fig.add_subplot(10, 2, 12)
    ax2.imshow(b[5], cmap='gray')
    ax1 = fig.add_subplot(10, 2, 13)
    ax1.imshow(a[6], cmap='gray')
    ax2 = fig.add_subplot(10, 2, 14)
    ax2.imshow(b[6], cmap='gray')
    ax1 = fig.add_subplot(10, 2, 15)
    ax1.imshow(a[7], cmap='gray')
    ax2 = fig.add_subplot(10, 2, 16)
    ax2.imshow(b[7], cmap='gray')
    ax1 = fig.add_subplot(10, 2, 17)
    ax1.imshow(a[8], cmap='gray')
    ax2 = fig.add_subplot(10, 2, 18)
    ax2.imshow(b[8], cmap='gray')
    ax1 = fig.add_subplot(10, 2, 19)
    ax1.imshow(a[9], cmap='gray')
    ax2 = fig.add_subplot(10, 2, 20)
    ax2.imshow(b[9], cmap='gray')

    plt.show()


def matrix_to_vektor(matrix):
    """

    Parameters
    ----------
    matrix : Any quadratic matrix of size NxN

    Returns
    -------
    vector : the matrix as vector, column vise added up.

    """
    M = np.transpose(matrix)
    vector = np.array(M[0])
    for i in range(1, len(M)):
        vector = np.concatenate((vector, M[i]))

    return vector


def empirical_covariance(imgs):
    """

    Parameters
    ----------
    imgs : The set of training pictures as 28x28 matrices

    Returns
    -------
    S : The covariance matrix of size 784x784 and the centered data matrix Y.

    """
    imgs = imgs[:1000]
    X = []
    for matrix in imgs:
        X.append(np.reshape(matrix, (784,)))
    X = np.array(X).T
    b = np.ndarray.mean(X, axis=1)
    Y = X - b[:, None]
    S = np.dot(Y, np.transpose(Y))

    return S, Y, b


def is_symmetric(A):
    """

    Parameters
    ----------
    A : Any matrix

    Returns
    -------
    If yes or no the matrix is symmetric

    """
    B = np.transpose(A)
    if (A - B).all() == 0:
        return 'yes'
    else:
        return 'no'


def eigenvalues(matrix):
    """

    Parameters
    ----------
    matrix : Any matrix

    Returns
    -------
    a list of all eigenvalues of the input matrix

    """
    return np.linalg.eigh(matrix)[0]


def SVD(matrix):
    """

    Parameters
    ----------
    matrix : any matrix

    Returns
    -------
    The SVD Zerlegung of input matrix

    """
    return np.linalg.svd(matrix)


def viz_task2(imgs, N=1000):
    S, Y = empirical_covariance(imgs[:N])[:2]
    eig_matrix = np.diag(eigenvalues(S)[:50])
    svd = SVD(Y)

    fig = plt.figure()

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(eig_matrix)

    ax1.set_title('Eigenwerte')

    ax3 = fig.add_subplot(1, 2, 2)
    ax3.imshow(np.diag(svd[1][:50] ** 2))

    ax3.set_title('quadrierte Singulaerwerte')

    plt.show()


def Hauptkomponenten(Data, d=5, N=1000):
    """

    Parameters
    ----------
    Data : A dataset of quadratic matrices of dim lxl
    d : dimesion of the desired affine Subspace. The default is 5.
    N : Restriction of number of elements to work with from data set.
    The default is 1000.

    Returns
    -------
    b : empirical mean of Data.
    eigenvectorSet : The matrix A of dim lxd of the subspace.

    output: A,b from {At+b,t in R^d}

    """
    S, Y, b = empirical_covariance(Data[:1000])
    LM, SIG = SVD(Y)[:2]
    LM = LM.T
    eigenvectorSet = LM[:d]
    return b, eigenvectorSet.T


def viz_hauptkomponenten(imgs):
    b, EV = Hauptkomponenten(imgs)
    fig = plt.figure(figsize=(60, 12))
    ax1 = fig.add_subplot(5, 2, 1)
    ax1.imshow(vektor_to_matrix(b))

    ax1.set_title('Mittelwert')

    EV = EV.T
    ax2 = fig.add_subplot(5, 2, 2)
    ax2.imshow(vektor_to_matrix(EV[0]))

    ax2.set_title('5 ersten Hauptkomponenten')

    ax3 = fig.add_subplot(5, 2, 4)
    ax3.imshow(vektor_to_matrix(EV[1]))
    ax4 = fig.add_subplot(5, 2, 6)
    ax4.imshow(vektor_to_matrix(EV[2]))
    ax5 = fig.add_subplot(5, 2, 8)
    ax5.imshow(vektor_to_matrix(EV[3]))
    ax6 = fig.add_subplot(5, 2, 10)
    ax6.imshow(vektor_to_matrix(EV[4]))

    plt.show()


def Four_rdProjections():
    """
    Picks 4 random images from imgs and computes their projection on
    the 5 first Hauptkomponenten

    Returns
    -------
    The plotted projections of these images

    """
    L = np.random.choice(6000, 4)
    toplot = []
    b, A = Hauptkomponenten(imgs)
    for i in L:
        vec = np.reshape(imgs[i], (784,))
        outp = np.dot(np.dot(A, A.T), (vec - b)) + b
        outp = np.reshape(outp, (28, 28))
        toplot.append(outp)

    fig = plt.figure(figsize=(60, 12))
    ax1 = fig.add_subplot(4, 2, 1)
    ax1.imshow(imgs[L[0]], cmap='gray')

    ax1.set_title('original Ziffer')

    ax2 = fig.add_subplot(4, 2, 2)
    ax2.imshow(toplot[0], cmap='gray')

    ax2.set_title('projektiertes Ziffer')

    ax3 = fig.add_subplot(4, 2, 3)
    ax3.imshow(imgs[L[1]], cmap='gray')
    ax4 = fig.add_subplot(4, 2, 4)
    ax4.imshow(toplot[1], cmap='gray')
    ax5 = fig.add_subplot(4, 2, 5)
    ax5.imshow(imgs[L[2]], cmap='gray')
    ax6 = fig.add_subplot(4, 2, 6)
    ax6.imshow(toplot[2], cmap='gray')
    ax7 = fig.add_subplot(4, 2, 7)
    ax7.imshow(imgs[L[3]], cmap='gray')
    ax8 = fig.add_subplot(4, 2, 8)
    ax8.imshow(toplot[3], cmap='gray')
    plt.show()


def vektor_to_matrix(vector):
    return np.reshape(vector, (28, 28))


def Datensatz_erzeuger(ziffer1, ziffer2, N=1000):
    """
    generates 2 lists of matrices, each corresponding to ziffer1 or ziffer2

    Parameters
    ----------
    ziffer1 : integer between 0 and 9.
    ziffer2 : integer between 0 and 9.
    N : the desired length of the lists . The default is 100.

    Returns
    -------
    imageZ1 : list of matrices corresponding to ziffer1.
    imageZ2 : list of matrices corresponding to ziffer2.

    """
    pos_ziffer1 = np.where(labs == ziffer1)[0][:N]
    pos_ziffer2 = np.where(labs == ziffer2)[0][:N]
    imageZ1 = []
    imageZ2 = []
    for i in pos_ziffer1:
        imageZ1.append(imgs[i])

    for i in pos_ziffer2:
        imageZ2.append(imgs[i])
    return imageZ1, imageZ2


def klassen_projektion(data1, data2, N=1000):
    """

    Parameters
    ----------
    ziffer1 : integer between 0 and 9.
    ziffer2 : integer between 0 and 9.
    N : the desired length of the lists. The default is 100.

    Returns
    -------
    To a 2-dim affine subspace projected data of the N-first matrices
    corresponding to ziffer1 and ziffer2. Also returns the mean of the
    projected data for both classes.

    """
    # data1,data2 = Datensatz_erzeuger(ziffer1, ziffer2,N)
    b1, A1 = Hauptkomponenten(data1, d=2)
    b2, A2 = Hauptkomponenten(data2, d=2)

    proj1 = []
    proj2 = []
    for x in data1:
        y = np.dot(A1.T, np.reshape(x, (784,)) - b1)
        proj1.append(y)

    for x in data2:
        y = np.dot(A2.T, np.reshape(x, (784,)) - b2)
        proj2.append(y)

    return np.array(proj1).T, np.array(proj2).T, np.dot(A1.T, b1), np.dot(A2.T, b2), A1, A2


def zwei_mean_algo(data1, data2, r1, r2, itr=100, N=100):
    """

    Parameters
    ----------
    Z1 : an integer between 0 and 9.
    Z2 : an integer between 0 and 9.
    itr : the number of desired loop of the K-mean algo. The default is 100.
    N : 2*N will be the number of data points to be classified.
    Each class will have N data points. The default is 100.

    Returns
    -------
    Two lists of classified data points to either Z1 or Z2, aswell as
    the number of misclassified points for each class.

    """
    # data1,data2,r1,r2 = klassen_projektion(Z1, Z2,N)

    # Initialisierung
    DATA = np.concatenate((data1, data2), axis=1)
    DATA = DATA.T
    C1 = np.empty((0, 2), float)
    C2 = np.empty((0, 2), float)

    # labeling the data
    D1 = np.empty((0, 2), float)
    D2 = np.empty((0, 2), float)
    for x in DATA[:N]:
        D1 = np.append(D1, np.array([x]), axis=0)
    for x in DATA[N:]:
        D2 = np.append(D2, np.array([x]), axis=0)

    # Wiederhole
    while itr != 0:
        for x in DATA:
            d1 = np.linalg.norm(x - r1) ** 2
            d2 = np.linalg.norm(x - r2) ** 2
            '''
            if d1>d2:
                if np.size(np.where(np.all(C2==x)))> 0:
                    C2=np.append(C2, np.array([x]), axis=0)
                if np.size(np.where(np.all(C1==x)))> 0:
                    C1 = np.delete(C1, np.where(C1==x)[0],axis=0)
            if d2>d1:
                if np.size(np.where(np.all(C1==x)))> 0:
                    C1=np.append(C1, np.array([x]), axis=0)
                if np.size(np.where(np.all(C2==x)))> 0:
                    C2 = np.delete(C2, np.where(C2==x)[0],axis=0)
            '''
            if (d1 > d2):
                i = searchIndex(C2, x)
                j = searchIndex(C1, x)
                if i == 'not inside':
                    C2 = np.append(C2, np.array([x]), axis=0)
                if j != 'not inside':
                    C1 = np.delete(C1, j, axis=0)

            if (d1 < d2):
                i = searchIndex(C1, x)
                j = searchIndex(C2, x)
                if i == 'not inside':
                    C1 = np.append(C1, np.array([x]), axis=0)
                if j != 'not inside':
                    C2 = np.delete(C2, j, axis=0)

        if len(C1) == 0:
            return C2.T
        if len(C2) == 0:
            return C1.T
        r1 = C1.mean(axis=0)
        r2 = C2.mean(axis=0)
        itr -= 1

    # checking misclassified
    correct_C1 = 0
    correct_C2 = 0
    for x in C1:
        if x in D1:
            correct_C1 += 1
    for x in C2:
        if x in D2:
            correct_C2 += 1

    return C1.T, C2.T, correct_C1, correct_C2


def searchIndex(M, x):
    """
    A manual search for np.arrays of shape (2,) in another np.array.
    The build in search function of numpy did not work, and comparison was
    impossible.
    This part is responsible for a longer running time.

    Parameters
    ----------
    M : numpy array containg arrays or nothing.
    x : array to be searched for.

    Returns
    -------
    either the position of x or the string 'not inside'.

    """
    for i in range(len(M)):
        if np.all(M[i] == x) == True:
            return i
    return 'not inside'


'''
def Training_set(Z1,Z2,N=1000):
    return Datensatz_erzeuger(Z1, Z2,N)
'''


def Test_set(Z1, Z2, N=100):
    """
    Parameters
    ----------
    Z1 : integer between 0 and 9.
    Z2 : integer between 0 and 9.
    N : desired length of test_set. The default is 100.

    Returns
    -------
    L1 : test set of Z1.
    L2 : test set of Z2.
    L : training set; is a tuple of 2 sets of length 500 each for Z1 and Z2.
    """
    L = Datensatz_erzeuger(Z1, Z2, 500)
    i = np.random.choice(len(L[0]), N, replace=False)
    j = np.random.choice(len(L[1]), N, replace=False)
    L1 = np.array([L[0][k] for k in i])
    L2 = np.array([L[1][k] for k in j])
    return L1, L2, L


def project_test(X, Y, A1, A2, b1, b2):
    """
    Given two sets X,Y of class 1 and 2, computes the projections
    A1.X-b1 and A2.Y-b2

    Parameters
    ----------
    A1 : The hautptkomponente of dimesion d=2 of some training set.
    A2 : The hauptkomponente of dimesion d=2 of some training set.
    b1 : mean of training set of class 1.
    b2 : mean of training set of class 2.

    Returns
    -------
    The sets of vectors projected to the affine subspace generated by training
    set (for each class 1 and 2).
    """
    proj1 = []
    proj2 = []
    for x in X:
        y = np.dot(A1.T, np.reshape(x, (784,)) - b1)
        proj1.append(y)

    for x in Y:
        y = np.dot(A2.T, np.reshape(x, (784,)) - b2)
        proj2.append(y)

    return np.array(proj1).T, np.array(proj2).T


def Task_4(Z1, Z2, itr1=10, itr2=100):
    print('training classifier...')
    L1, L2, L = Test_set(Z1, Z2)
    CLASS1, CLASS2, b1, b2, A1, A2 = klassen_projektion(L[0], L[1])

    # plot of training data
    X, Y = zwei_mean_algo(CLASS1, CLASS2, b1, b2, itr1, N=500)[:2]
    plt.figure(figsize=(12, 7))
    plt.subplot(2, 2, 1)
    plt.scatter(X[0], X[1], label=f"class {Z1}")
    plt.scatter(Y[0], Y[1], label=f"class {Z2}")
    plt.legend()
    plt.title("K-Means für Trainingdaten")

    print('training completed !')
    print('proceeding with testing data...')

    # plot of test data
    mean784_1 = np.dot(A1, b1)
    mean784_2 = np.dot(A2, b2)
    T1, T2 = project_test(L1, L2, A1, A2, mean784_1, mean784_2)

    G1, G2, correct1, correct2 = zwei_mean_algo(T1, T2, b1, b2, itr2, N=50)
    plt.subplot(2, 2, 2)
    plt.scatter(G1[0], G1[1], label=f"class {Z1}")
    plt.scatter(G2[0], G2[1], label=f"class {Z2}")
    plt.legend()
    plt.title("K-Means für Testdaten")

    # plot of misclassified data
    plt.subplot(2, 2, 4)
    plt.box(on=None)
    ax = plt.gca()
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)
    rows = ["richtig", "falsch"]
    cols = [f"class {Z1}", f"class{Z2}"]
    data = [[100 - correct1, 100 - correct2], [correct1, correct2]]
    plt.table(data, rowLabels=rows, colLabels=cols, loc="top")

    plt.show()
    print('done !')


benutzereingabe = 5
while benutzereingabe != 0:
    benutzereingabe = input("Die Loesung welcher Aufgabe soll gezeigt werden? (moegliche Werte: 1,2,3,4) Wähle 0 zum "
                            "verlassen des Programms.")
    benutzereingabe = int(benutzereingabe)
    if benutzereingabe == 1:
        print('Mittelwert und Varianz aus den ersten 100 Trainingsbildern :')
        viz_mean_var()
    elif benutzereingabe == 2:
        print('Nach Berechnung der Kovarianzmatrix der ersten 1000 Trainingsbilder, sehen die 50 ersten' 
              'Eigenwerte und 50 ersten quadrierten Singulaerwerte so aus :')
        viz_task2(imgs)
    elif benutzereingabe == 3:
        print('Fuer die ersten 1000 Trainingsbilder, werden der Mittelwert und die 5 ersten Hauptkomponenten'
              'berechnet. 4 zufaellige Ziffern werden ausgesucht und auf diesen Komponenten projeziert.'
              'Das Ganze sieht so aus:')
        viz_hauptkomponenten(imgs)
        Four_rdProjections()
    elif benutzereingabe == 4:
        x = input('waehlen sie jetzt zwei Ziffer 0 bist 9 (getrennt durch ",", z.B 3,4)')
        y = x.split(',')
        if len(x) != 2:
            print('bitte 2 Ziffern waehlen')
            benutzereingabe = 4
        print('Sie haben, ' + x, 'gewaehlt. Es wird jetzt 1000 Trainingsbilder fuer die beide Ziffern'
              'gewaehlt und die Hauptkomponenten berechnet und eine Klassifizierung durch'
              'das K-mean Algorithmus wird durchgefuehrt. Dann waehlen wir 100 Test bilder und '
              'klassifizieren diese durch K-mean Algorithmus. Fuer das Training K-mean, werden 10'
              'Iterationen gemacht. Fuer das Test K-mean werden 100 Iterationen gemacht.'
              'Average waiting time: 60 seconds')
        print(y)
        Task_4(int(y[0]), int(y[1]))
    elif benutzereingabe == 0:
        print('Tschüss!')
    else:
        print('Falsche Eingabe!')
