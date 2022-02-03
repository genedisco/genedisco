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
import numpy as np
from typing import List, AnyStr
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from slingpy import AbstractDataSource, AbstractBaseModel
from genedisco.active_learning_methods.acquisition_functions.base_acquisition_function import \
    BaseBatchAcquisitionFunction


class Kmeans(BaseBatchAcquisitionFunction):
    def __init__(self, representation="linear", n_init=10):
        """
            is embedding: Apply kmeans to embedding or raw data
            n_init: Specifies the number of kmeans run-throughs to use, wherein the one with the smallest inertia is
                selected for the selection phase
        """
        self.representation = representation
        self.n_init = n_init
        super(Kmeans, self).__init__()

    def __call__(self, dataset_x: AbstractDataSource, batch_size: int, available_indices: List[AnyStr],
                 last_selected_indices: List[AnyStr], last_model: AbstractBaseModel) -> List:
        if self.representation == 'linear':
            kmeans_dataset = last_model.get_embedding(dataset_x.subset(available_indices)).numpy()
        elif self.representation == 'raw':
            kmeans_dataset = np.squeeze(dataset_x.subset(available_indices), axis=1)
        else:
            raise ValueError("Representation must be one of 'linear', 'raw'")

        centers = self.kmeans_clustering(kmeans_dataset, batch_size)
        chosen = self.select_closest_to_centers(kmeans_dataset, centers)
        return [available_indices[idx] for idx in chosen]

    def kmeans_clustering(self, kmeans_dataset, num_centers):
        kmeans = KMeans(init='k-means++', n_init=self.n_init, n_clusters=num_centers).fit(kmeans_dataset)
        return kmeans.cluster_centers_

    def select_closest_to_centers(self, options, centers):
        dist_ctr = pairwise_distances(options, centers)
        min_dist_indices = np.argmin(dist_ctr, axis=0)

        return list(min_dist_indices)
