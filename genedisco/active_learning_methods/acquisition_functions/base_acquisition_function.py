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
from abc import abstractmethod
from typing import AnyStr, List
from slingpy import AbstractDataSource
from slingpy.models.abstract_base_model import AbstractBaseModel


class BaseBatchAcquisitionFunction(object):
    @abstractmethod
    def __call__(self,
                 dataset_x: AbstractDataSource,
                 batch_size: int,
                 available_indices: List[AnyStr],
                 last_selected_indices: List[AnyStr],
                 last_model: AbstractBaseModel,
                 ) -> List:
        """
        Nominate experiments for the next learning round.

        Args:
            dataset_x: The dataset containing all training samples.
            batch_size: Size of the batch to acquire.
            available_indices: The list of the indices (names) of the samples not
                chosen in the previous rounds.
            last_selected_indices: The set of indices selected in the prrevious
                cycle.
            last_model: The prediction model trained by labeled samples chosen so far.

        Returns:
            A list of indices (names) of the samples chosen for the next round.
        """
        raise NotImplementedError()
