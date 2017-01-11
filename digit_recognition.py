import math
import pandas as pd
import numpy as np
import os
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KDTree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from skimage.feature import hog
from skimage import data, color, exposure, draw


DATA_DIR = os.path.join(os.path.expanduser('~'), 'mnist')

def fetch_data():
    return fetch_mldata('MNIST original', data_home=DATA_DIR)


def train_linear():
    sample = np.random.randint(0, len(X), 5000)
    X_sample = X[sample,:]
    X_hog = np.array([my_hog(reshape_image_vector(x)) for x in X_sample])
    y_sample = y[sample]
    X_train, X_test, y_train, y_test = (
        train_test_split(X_hog, y_sample, test_size=0.1, random_state=100)
    )
    model = LogisticRegression().fit(X_train, y_train)
    preds = model.predict(X_test)
    score = accuracy_score(y_test, preds)
    pairs = np.vstack([y_test, preds]).T
    mislabeled = pairs[pairs[:,0] != pairs[:,1]]

    df = pd.DataFrame(mislabeled)
    grouped = df.groupby(list(df.columns))
    error_counts = grouped.size().sort_values()

    seven_nine_errors = pd.DataFrame(pairs).apply((lambda r: np.logical_and(r[0] == 7, r[1] == 9)), axis=1)

    return model, score


def train_all_hogs(sample_size=10000, random_state=0):
    models = {}
    scores = {}
    X_sample, y_sample = take_sample(X, y, sample_size=sample_size)

    for C in np.logspace(-1, 0, 3):
        for num_orientations in (8,):#range(4, 14, 2):
            for ppc in ((3, 3),):
                model, score = train_single_hog(
                    X_sample,
                    y_sample,
                    num_orientations,
                    ppc,
                    random_state,
                    C
                )
                models[(num_orientations, ppc, C)] = model
                scores[(num_orientations, ppc, C)] = score

    return models, scores


def train_single_hog(X_sample, y_sample, num_orientations, ppc, random_state, C):
    hog_features = np.array([
        get_hog(reshape_image_vector(x), num_orientations, ppc)
        for x in X_sample
    ])
    X_train, X_test, y_train, y_test = (
        train_test_split(hog_features, y_sample, test_size=0.1, random_state=random_state)
    )
    model = LogisticRegression(penalty='l1', C=C).fit(X_train, y_train)
    preds = model.predict(X_test)
    score = accuracy_score(y_test, preds)
    return model, score


def take_sample(X, y, sample_size):
    sample = np.random.randint(0, len(X), sample_size)
    return X[sample,:], y[sample]


def visualize_coef_officially(coef, num_orientations, ppc):
    cx, cy = ppc
    sy, sx = (28, 28)
    n_cellsx = int(np.floor(sx // cx)) # number of cells in x
    n_cellsy = int(np.floor(sy // cy)) # number of cells in y
    radius = min(cx, cy) // 2 - 1
    orientations_arr = np.arange(num_orientations)
    dx_arr = radius * np.cos(orientations_arr / num_orientations * np.pi)
    dy_arr = radius * np.sin(orientations_arr / num_orientations * np.pi)
    fig, axes = plt.subplots(4, 3, figsize=(8, 4), sharex=True, sharey=True)

    vmax = abs(coef).max()
    vmin = -vmax

    for i, ax in enumerate(np.ravel(axes)[:10]):
        hog_vector = reshape_hog_vector(coef[i,:], num_orientations, ppc)
        hog_image = np.zeros((sy, sx), dtype=float)
        for x in range(n_cellsx):
            for y in range(n_cellsy):
                for o, dx, dy in zip(orientations_arr, dx_arr, dy_arr):
                    centre = tuple([y * cy + cy // 2, x * cx + cx // 2])
                    rr, cc = draw.line(int(centre[0] - dx),
                                       int(centre[1] + dy),
                                       int(centre[0] + dx),
                                       int(centre[1] - dy))
                    hog_image[rr, cc] += hog_vector[y, x, o]
        ax.set_title('Digit {0}'.format(i))
        ax.imshow(hog_image, cmap=plt.cm.gray, interpolation='none', vmin=vmin, vmax=vmax)

    plt.show()
    plt.close()


def visualize_coef(coef, num_orientations, ppc):
    reshaped = reshape_hog_vector(coef, num_orientations, ppc)
    dim = math.ceil(num_orientations / num_orientations**.5)
    fig, axes = plt.subplots(nrows=dim, ncols=dim, figsize=(6, 6))
    for x in range(dim):
        for y in range(dim):
            if x * dim + y < num_orientations:
                axes[x, y].imshow(reshaped[:, :, x * dim + y], cmap='gray')
    # normalized = 256 * reshaped / reshaped.max()
    # plt.imshow(reshaped[0,:,:], cmap='gray')
    plt.show()
    plt.close()


def aggregate_coef(all_coef, num_orientations, ppc):
    final = {}
    for digit in range(10):
        coef = all_coef[digit,:]
        reshaped = reshape_hog_vector(coef, num_orientations, ppc)
        final[digit] = {
            orientation: np.abs(reshaped[:,:, orientation]).mean() for orientation in range(num_orientations)
        }
    return pd.DataFrame(final)


def reshape_image_vector(v):
    return np.reshape(v, [28,28])


def reshape_hog_vector(hog_vector, num_orientations, ppc):
    return hog_vector.reshape(28 // ppc[1], 28 // ppc[0], num_orientations)


def plot_vector(v):
    reshaped = reshape_image_vector(v)
    plt.imshow(reshaped, cmap='gray')
    plt.show()


def get_hog(image, num_orientations, pixels_per_cell, feature_vector=True):
    return hog(image,
               orientations=num_orientations,
               pixels_per_cell=pixels_per_cell,
               cells_per_block=(1, 1),
               visualise=False,
               feature_vector=feature_vector)


if '__name__' == 'main':
    bunch = fetch_data()
    X = bunch.data
    y = bunch.target
    X_sample, y_sample = take_sample(X, y, 10000)
    num_orientations = 4
    ppc = (7, 7)
    random_state = 0
    C = .05
    model, score = train_single_hog(X_sample, y_sample, num_orientations, ppc, random_state, C)
    print('score is {0}'.format(score))
    visualize_coef_officially(model.coef_, num_orientations, ppc)
