# t-SNE visualization for feature embeddings.
#
# HCL extension by Tek Yaw Ng, Lotte van den Berg, Jason Tran (2026):
# "Hierarchical Contrastive Learning in Cross-Domain Aspect-Based Sentiment Classification"
# https://github.com/tekyawng/HCL-DAT-LCR-Rot-hop-plusplus/
#
# Adapted from Johan Verschoor (2025) for the thesis:
# "Enhancing Cross-Domain Aspect-Based Sentiment Analysis with Contrastive Learning"
#
# Erasmus University Rotterdam
# Master Econometrics and Management Science
# Business Analytics and Quantitative Marketing
#
# Generates and saves a t-SNE scatter plot, re-scaling the 2D embedding to lie within [-2,2].

import os
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.colors as mcolors

def plot_tsne(
    features,
    labels,
    sess=None,
    feed_dict=None,
    plot_title: str = "t-SNE Plot",
    save_dir: str = "tsne_plots"
):
    """
    Generates and saves a t-SNE scatter plot of the given features, colored by labels,
    then re-scales the 2D embedding to lie within [-2, 2] in both dimensions.
    Uses a discrete colormap for labels 0, 1, 2 (blue, green, red).

    :param features: np.ndarray or tf.Tensor
        2D array or Tensor of shape [batch_size, feature_dim].
    :param labels: np.ndarray or tf.Tensor
        1D/2D array or Tensor of shape [batch_size, ] or [batch_size, n_classes].
        If one-hot, it will be converted to single class index for coloring.
    :param sess: tf.Session or None
        If 'features' or 'labels' are Tensors, we need a Session to evaluate them.
    :param feed_dict: dict or None
        Optional feed dictionary if 'features' or 'labels' require placeholders for evaluation.
    :param plot_title: str
        The title of the t-SNE plot.
    :param save_dir: str
        The directory where the plot image is saved.
    """

    # Create output directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # 1) If 'features' or 'labels' is a TF Tensor, evaluate them using sess.run
    if isinstance(features, tf.Tensor):
        if sess is None:
            raise ValueError("Cannot evaluate 'features' Tensor because 'sess' is None.")
        features = sess.run(features, feed_dict=feed_dict)

    if isinstance(labels, tf.Tensor):
        if sess is None:
            raise ValueError("Cannot evaluate 'labels' Tensor because 'sess' is None.")
        labels = sess.run(labels, feed_dict=feed_dict)

    # 2) If labels is one-hot (or multi-class), convert to single class indices
    if len(labels.shape) > 1 and labels.shape[1] > 1:
        labels = np.argmax(labels, axis=1)

    # 3) Perform TSNE
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate='auto',
        init='pca',
        random_state=42
    )
    reduced_features = tsne.fit_transform(features)

    # 3a) Re-center & re-scale so that the final x,y lie in [-2, 2]
    x_min, y_min = reduced_features.min(axis=0)
    x_max, y_max = reduced_features.max(axis=0)
    range_x = x_max - x_min
    range_y = y_max - y_min

    # Avoid division by zero if data is constant along one dimension
    if range_x == 0:
        range_x = 1
    if range_y == 0:
        range_y = 1

    # Rescale each dimension to [-2, 2]
    rescaled_features = reduced_features.copy()
    rescaled_features[:, 0] = 4.0 * (reduced_features[:, 0] - x_min) / range_x - 2.0
    rescaled_features[:, 1] = 4.0 * (reduced_features[:, 1] - y_min) / range_y - 2.0

    # 3b) Define a discrete colormap for labels 0, 1, 2
    # - blue (label 0)
    # - green (label 1)
    # - red (label 2)
    discrete_cmap = mcolors.ListedColormap(["blue", "green", "black"])
    # Bins to separate each class label
    boundaries = [-0.5, 0.5, 1.5, 2.5]
    norm = mcolors.BoundaryNorm(boundaries, discrete_cmap.N)

    # 4) Create the scatter plot with the rescaled data
    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(
        rescaled_features[:, 0],
        rescaled_features[:, 1],
        c=labels,
        cmap=discrete_cmap,
        norm=norm,
        alpha=0.7
    )
    cbar = plt.colorbar(scatter, ticks=[0, 1, 2])
    cbar.set_label("Label")
    plt.title(plot_title)

    # 5) Save the figure to disk
    filename = f"{plot_title}.png"
    outpath = os.path.join(save_dir, filename)
    plt.savefig(outpath)
    plt.close()

    print(f"[plot_tsne] Saved t-SNE plot (rescaled to [-2,2]) to: {outpath}")
