# Author: Mathieu Blondel
#         Derek Lim
# License: BSD 3 clause


import argparse
import sys
from pathlib import Path
import os

import numpy as np
import matplotlib.pylab as plt
from matplotlib import patches as patches
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.externals import joblib
import ot

from smoothot.dual_solvers import solve_semi_dual, get_plan_from_semi_dual
from smoothot.dual_solvers import NegEntropy, SquaredL2
import dataset

# make needed directories if they do not already exist
root_dir = os.path.dirname(os.path.abspath(__file__))
Path(os.path.join(root_dir, 'images')).mkdir(exist_ok=True)
Path(os.path.join(root_dir, 'res')).mkdir(exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--n_colors', type=int, default=256, help='number of color clusters')
parser.add_argument('--method', type=str, default='l2_sd', help='OT method')
parser.add_argument('--gamma', type=float, default=1.0, help='regularization parameter')
parser.add_argument('--max_iter', type=int, default=1000)
parser.add_argument('--img1', type=str, default='comunion')
parser.add_argument('--img2', type=str, default='autumn')
args = parser.parse_args()

def map_img(T, img1, img2, weights1):
    """Transfer colors from img2 to img1"""
    return np.dot(T / weights1[:, np.newaxis], img2)

n_colors = args.n_colors
method = args.method
gamma = args.gamma
max_iter = args.max_iter
img1 = args.img1
img2 = args.img2
pair = img1+'-'+img2

print("images:", img1, '|', img2)
print("n_colors:", n_colors)
print("gamma:", gamma)
print("max_iter:", max_iter)
print()

hist1, hist2, C, centers1, centers2, labels1, labels2, shape1, shape2 = \
dataset.load_color_transfer(img1=img1, img2=img2, n_colors=n_colors,
                            transpose=False)
m = len(hist1)
n = len(hist2)

# Obtain transportation plan.
if method == "l2_sd":
    regul = SquaredL2(gamma=gamma)
    alpha = solve_semi_dual(hist1, hist2, C, regul, max_iter=max_iter, tol=1e-6)
    T = get_plan_from_semi_dual(alpha, hist2, C, regul)
    name = "Squared 2-norm"
elif method == "ent_sd":
    regul = NegEntropy(gamma=gamma)
    alpha = solve_semi_dual(hist1, hist2, C, regul, max_iter=max_iter, tol=1e-6)
    T = get_plan_from_semi_dual(alpha, hist2, C, regul)
    name = "Entropy"
elif method == "lp":
    T = ot.emd(hist1, hist2, C)
    name = "Unregularized"
else:
    raise ValueError("Invalid method")

sparsity = np.sum(T > 1e-10) / T.size
print("Sparsity:", sparsity)

T1 = np.sum(T, axis=1)
Tt1 = np.sum(T, axis=0)
err_a = hist1 - np.sum(T, axis=1)
err_b = hist2 - np.sum(T, axis=0)
print("Marginal a", np.dot(err_a, err_a))
print("Marginal b", np.dot(err_b, err_b))

#print(np.sum(T * C) + 0.5 / gamma * np.dot(err_a, err_a) + 0.5 / gamma * np.dot(err_b, err_b))
print('Objective value:', np.sum(T * C))

T_ = ot.emd(hist1, hist2, C)
print('Unregularized objective value:', np.sum(T_ * C))

img1 = centers1[labels1]
img2 = centers2[labels2]

centers1_mapped = map_img(T, centers1, centers2, T1)
img1_mapped = centers1_mapped[labels1]

centers2_mapped = map_img(T.T, centers2, centers1, Tt1)
img2_mapped = centers2_mapped[labels2]

fig = plt.figure(figsize=(6,6))

ax = fig.add_subplot(221)
ax.imshow(img1.reshape(shape1))
ax.axis("off")

ax = fig.add_subplot(222)
ax.imshow(img2.reshape(shape2))
ax.axis("off")

ax = fig.add_subplot(223)
ax.imshow(img1_mapped.reshape(shape1))
ax.axis("off")

ax = fig.add_subplot(224)
ax.imshow(img2_mapped.reshape(shape2))
ax.axis("off")

plt.tight_layout()

# plot original and transformed images
out = "%s/images/%s_%d_%s_%0.3e.jpg" % (root_dir, method, n_colors, pair, gamma)
plt.savefig(out)
print()
print('Saved image to:', out)

out = "%s/res/img_%s_%d_%s_%0.3e.pkl" % (root_dir, method, n_colors, pair, gamma)
tup = (img1_mapped.reshape(shape1), img2_mapped.reshape(shape2), sparsity)

joblib.dump(tup, out)
print('Saved pickle to:', out)

plt.show()

if n_colors <= 32:
    # plot transport plan and color histogram

    def draw_blocks(ax, T):
        # find contiguous chunks between coefficients
        for k, attn_row in enumerate(T):
            brk = np.diff(attn_row)
            brk = np.where(brk != 0)[0]
            brk = np.append(0, brk + 1)
            brk = np.append(brk, T.shape[0])

            right_border = True
            for s, t in zip(brk[:-1], brk[1:]):
                if attn_row[s:t].sum() == 0:
                    right_border = False
                    continue
                lines = [(s, k), (t, k), (t, k + 1), (s, k + 1)]
                lines = np.array(lines, dtype=np.float) - 0.5
                path = patches.Polygon(lines, facecolor='none', linewidth=1.5,
                                       alpha=1, joinstyle='round',
                                       closed=not right_border,
                                       edgecolor='#999999')
                ax.add_patch(path)
                right_border = True

    def draw_border(ax, T):
        lines = [(0, 0), (0, T.shape[1]), (T.shape[0], T.shape[1]), (T.shape[0], 0)]
        lines = np.array(lines) - 0.5
        path = patches.Polygon(lines, facecolor='none', linewidth=1.5,
                                   alpha=1, joinstyle='round',
                                   closed=True, edgecolor='#999999')
        scatter_axes.add_patch(path)


    plt.figure(figsize=(5,5))

    scatter_axes = plt.subplot2grid((3, 3), (1, 0), rowspan=2, colspan=2)
    x_hist_axes = plt.subplot2grid((3, 3), (0, 0), colspan=2,
                                   sharex=scatter_axes)

    scatter_axes.imshow(T, cmap=plt.cm.Reds, interpolation="nearest")
    scatter_axes.axis("off")

    scatter_axes.set_xlabel(name)

    draw_blocks(scatter_axes, T)
    draw_border(scatter_axes, T)

    bar_list = x_hist_axes.bar(np.arange(m), height=hist1,
                               width=(0.80 * n_colors) / m, edgecolor="#999999",
                               linewidth=1.0, color=centers1)
    x_hist_axes.axis("off")

    if method in ("l2_primal", "l2_sp"):
        y_hist_axes = plt.subplot2grid((3, 3), (1, 2), rowspan=2,
                                   sharey=scatter_axes)
        bar_list = y_hist_axes.barh(np.arange(n), width=hist2,
                                    height=(0.80 * n_colors) / n,
                                    edgecolor="#999999", linewidth=1.0, color=centers2)
        y_hist_axes.axis("off")

    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0, bottom=0.12)


    plt.show()
