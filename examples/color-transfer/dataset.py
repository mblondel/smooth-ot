# Author: Mathieu Blondel
#         Derek Lim
# License: BSD 3 clause

from matplotlib import image
import numpy as np
from sklearn.externals import joblib
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans

DATAPATH = '../data'

def color_quantize(img, n_colors, name, method):
    """ cluster all colors of image """
    shape = img.shape
    img = img.reshape(-1, 3)

    if method == "kmeans":
        km = KMeans(n_clusters=n_colors, n_init=1, max_iter=300).fit(img)
        centers = km.cluster_centers_
        labels = km.labels_
    elif method == "rand":
        rng = np.random.RandomState(0)
        ind = rng.permutation(img.shape[0])
        centers = img[ind[:n_colors]]

        D = euclidean_distances(centers, img, squared=True)
        labels = D.argmin(axis=0)
    else:
        raise ValueError("Invalid quantization method")

    out = "color-transfer/res/%s_%s_%d_colors.pkl" % (name, method, n_colors)
    joblib.dump((shape, centers, labels), out)
    print('Saving color quantization:', out)


def load_color_transfer(img1="comunion", img2="autumn", n_colors=256,
                        method="kmeans", transpose=False):

    if transpose:
        img1, img2 = img2, img1

    # Load quantized images.
    try: # load if already saved
        shape1, centers1, labels1 = \
            joblib.load("color-transfer/res/%s_%s_%d_colors.pkl" % (img1, method, n_colors))
    except:
        img = image.imread('%s/%s.jpg' % (DATAPATH, img1)).astype(np.float64) / 256
        color_quantize(img, n_colors, img1, method=method)
        shape1, centers1, labels1 = \
            joblib.load("color-transfer/res/%s_%s_%d_colors.pkl" % (img1, method, n_colors))

    try: # load if already saved
        shape2, centers2, labels2 = \
            joblib.load("color-transfer/res/%s_%s_%d_colors.pkl" % (img2, method, n_colors))
    except:
        img = image.imread('%s/%s.jpg' % (DATAPATH, img2)).astype(np.float64) / 256
        color_quantize(img, n_colors, img2, method=method)
        shape2, centers2, labels2 = \
            joblib.load("color-transfer/res/%s_%s_%d_colors.pkl" % (img2, method, n_colors))


    m = centers1.shape[0]
    n = centers2.shape[0]

    # Prepare histograms and cost matrix.
    hist1 = np.bincount(labels1, minlength=m).astype(np.float64)
    hist1 /= np.sum(hist1)

    hist2 = np.bincount(labels2, minlength=n).astype(np.float64)
    hist2 /= np.sum(hist2)

    # Remove elements with probability 0.
    hist1 += 1e-9
    hist1 /= np.sum(hist1)
    hist2 += 1e-9
    hist2 /= np.sum(hist2)

    # Sort centers and histograms.
    ind1 = np.argsort(hist1)[::-1]
    hist1 = hist1[ind1]
    centers1 = centers1[ind1]
    inv_map1 = dict((ind1[i], i) for i in range(len(ind1)))
    labels1 = np.array([inv_map1[l] for l in labels1])

    ind2 = np.argsort(hist2)[::-1]
    inv_ind2 = np.arange(len(hist2))[ind2]
    hist2 = hist2[ind2]
    centers2 = centers2[ind2]
    inv_map2 = dict((ind2[i], i) for i in range(len(ind2)))
    labels2 = np.array([inv_map2[l] for l in labels2])

    # Prepare cost matrix.
    C = euclidean_distances(centers1, centers2, squared=True)

    return hist1, hist2, C, centers1, centers2, labels1, labels2, shape1, shape2

