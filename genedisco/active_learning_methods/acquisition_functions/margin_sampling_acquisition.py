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
from typing import AnyStr, List
from slingpy import AbstractDataSource
from slingpy.models.abstract_base_model import AbstractBaseModel
from genedisco.active_learning_methods.acquisition_functions.base_acquisition_function import \
    BaseBatchAcquisitionFunction


class MarginSamplingAcquisition(BaseBatchAcquisitionFunction):
    def __call__(self,
                 dataset_x: AbstractDataSource,
                 select_size: int,
                 available_indices: List[AnyStr],
                 last_selected_indices: List[AnyStr] = None,
                 model: AbstractBaseModel = None,
                 ) -> List:
        avail_dataset_x = dataset_x.subset(available_indices)
        model_pedictions = model.predict(avail_dataset_x, return_std_and_margin=True)

        if len(model_pedictions) != 3:
            raise TypeError("The provided model does not output margins.")

        # print("Hi! This is Margin Sampling!")
        pred_mean, pred_uncertainties, pred_margins = model_pedictions
        # print(pred_margins)

        if len(pred_mean) < select_size:
            raise ValueError("The number of query samples exceeds"
                             "the size of the available data.")

        numerical_selected_indices = np.flip(
            np.argsort(pred_margins)
        )[:select_size]
        selected_indices = [available_indices[i] for i in numerical_selected_indices]

        return selected_indices
