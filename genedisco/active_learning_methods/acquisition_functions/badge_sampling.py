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
import scipy.stats
import numpy as np
from typing import List, AnyStr
from sklearn.metrics import pairwise_distances
from slingpy import AbstractDataSource, AbstractBaseModel
from genedisco.active_learning_methods.acquisition_functions.base_acquisition_function import \
    BaseBatchAcquisitionFunction


class BadgeSampling(BaseBatchAcquisitionFunction):
    def __call__(self, dataset_x: AbstractDataSource, batch_size: int, available_indices: List[AnyStr],
                 last_selected_indices: List[AnyStr], last_model: AbstractBaseModel) -> List:
        gradient_embedding = last_model.get_gradient_embedding(dataset_x.subset(available_indices)).numpy()
        chosen = BadgeSampling.kmeans_initialise(gradient_embedding, batch_size)
        selected = [available_indices[idx] for idx in chosen]
        return selected

    @staticmethod
    def kmeans_initialise(gradient_embedding, k):
        ind = np.argmax([np.linalg.norm(s, 2) for s in gradient_embedding])
        mu = [gradient_embedding[ind]]
        indsAll = [ind]
        centInds = [0.] * len(gradient_embedding)
        cent = 0
        while len(mu) < k:
            if len(mu) == 1:
                D2 = pairwise_distances(gradient_embedding, mu).ravel().astype(float)
            else:
                newD = pairwise_distances(gradient_embedding, [mu[-1]]).ravel().astype(float)
                for i in range(len(gradient_embedding)):
                    if D2[i] > newD[i]:
                        centInds[i] = cent
                        D2[i] = newD[i]
            D2 = D2.ravel().astype(float)
            Ddist = (D2 ** 2) / sum(D2 ** 2)
            customDist = scipy.stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
            ind = customDist.rvs(size=1)[0]
            mu.append(gradient_embedding[ind])
            indsAll.append(ind)
            cent += 1
        gram = np.matmul(gradient_embedding[indsAll], gradient_embedding[indsAll].T)
        val, _ = np.linalg.eig(gram)
        val = np.abs(val)
        vgt = val[val > 1e-2]
        return indsAll
