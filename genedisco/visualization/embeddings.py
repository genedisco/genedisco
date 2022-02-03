"""
Copyright 2021 Patrick Schwab, Arash Mehrjou, GlaxoSmithKline plc; Andrew Jesson, University of Oxford; Ashkan Soleymani, MIT

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import glob
import os
import matplotlib
import numpy as np
import pandas as pd
from PIL import Image
import seaborn as sns
matplotlib.use('Agg')
matplotlib.style.use('ggplot')
import matplotlib.pyplot as plt
from typing import Optional, List
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA



_TSNE_LEARNING_RATE = 50.0 # Has significant effect on the resultant shape.
_NUM_PCA_COMPONENTS = 2 # This is the number of PCA component to keep not the
# the number of compoentns to plot.


def plot_tsne(data: np.ndarray,
              targets: Optional[np.ndarray] = None,
              colors: List[str] = None,
              saving_path: str = None):
    """Plot tsnee embedding of data.

    Takes data array and plot its tsnee visualization in 2D. The optional
    target (class/value) information can bve provided together with optional corresponding
    colors to draw every sample with its associated color.

    Args:
        data: An np.ndarray of shape (#datapoints, feature_size).
        targets: An np.array of integers with shape (#datapoints) that shows
            that shows thed datapoint corresponding to that index belongs to
            a certain class or continuous value.
        colors: A list of string colors that that are used for each class.
        saving_path; The full path (including file name) to save the plot.
    """
    
    tsne = TSNE(n_components=2, verbose=1, 
                perplexity=40, n_iter=300, learning_rate=_TSNE_LEARNING_RATE)
    tsne_results = tsne.fit_transform(data)
    df = pd.DataFrame(data=tsne_results,
                      columns=["tsne-2d-one", "tsne-2d-two"])
    if targets:
        df["y"] = targets
    plt.figure(figsize=(16,10))
    sns_plot = sns.scatterplot(
                    data=df,
                    x= "tsne-2d-one", y= "tsne-2d-two",
                    hue="y" if targets else None,
                    alpha=0.3
                    )
    if saving_path:
        sns_plot.figure.savefig(saving_path)


def plot_pca(data: np.ndarray,
             targets: Optional[np.ndarray] = None,
             colors: List[str] = None,
             saving_path: str = None):
    """Plot PCA embedding of data.

    Takes data array and plot its 2D projection on the leading two components
    found by PCA transformation. The optional target (class/value) information 
    can bve provided together with optional corresponding colors to draw every
    sample with its associated color.

    Args:
        data: An np.ndarray of shape (#datapoints, feature_size).
        targets: An np.array of integers with shape (#datapoints) that shows
            that shows thed datapoint corresponding to that index belongs to
            a certain class or continuous value.
        colors: A list of string colors that that are used for each class.
        saving_path; The full path (including file name) to save the plot.
    """
    
    pca = PCA(n_components=_NUM_PCA_COMPONENTS)
    pca_result = pca.fit_transform(data)
    df = pd.DataFrame(pca_result, columns=['pca-one', 'pca-two'])
    if targets is not None:
        df["y"] = targets
    plt.figure(figsize=(16,10))
    sns_plot = sns.scatterplot(
                    x="pca-one", y="pca-two",
                    hue="y" if targets is not None else None,
                    style="y" if targets is not None else None,
                    data=df,
                    alpha=0.3
                    )
    if saving_path:
        sns_plot.figure.savefig(saving_path)


def plot_pca_multiclass(data: np.ndarray,
             targets: Optional[np.ndarray] = None,
             colors: List[str] = None,
             saving_path: str = None):
    """Plot PCA embedding of data.

    Takes data array and plot its 2D projection on the leading two components
    found by PCA transformation. The optional target (class/value) information 
    can bve provided together with optional corresponding colors to draw every
    sample with its associated color.

    Args:
        data: An np.ndarray of shape (#datapoints, feature_size).
        targets: An np.array of integers with shape (#datapoints) that shows
            that shows thed datapoint corresponding to that index belongs to
            a certain class or continuous value.
        colors: A list of string colors that that are used for each class.
        saving_path; The full path (including file name) to save the plot.
    """
    _NUM_PCA_COMPONENTS = 2
    pca = PCA(n_components=_NUM_PCA_COMPONENTS)
    pca_result = pca.fit_transform(data)
    df = pd.DataFrame(pca_result, columns=['pca-one', 'pca-two'])
    if targets is not None:
        df["y"] = targets
    plt.figure(figsize=(16,10))
    sns_plot = sns.scatterplot(
                    x="pca-one", y="pca-two",
                    data=df.iloc[targets == 0],
                    alpha=0.3
                    )
    sns_plot = sns.scatterplot(
                x="pca-one", y="pca-two",
                data=df.iloc[targets == 1],
                alpha=1.0
                )
    if saving_path:
        sns_plot.figure.savefig(saving_path)


def make_gif(frame_folder, animated_filename="myGif.gif"):
    """make gif from the images in the frame_folder.
    
    The images in the frame folder must be indexed as *_{ind}.png"
    """
    frames = [Image.open(image) for image in sorted(glob.glob(f"{frame_folder}/*.png"))]
    frame_one = frames[0]
    frame_one.save(
        os.path.join(frame_folder, animated_filename), 
        format="GIF", 
        append_images=frames,
        save_all=True,
        duration=100, 
        loop=0)
