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
from slingpy.data_access.data_sources.abstract_data_source import AbstractDataSource
from slingpy.models.tarfile_serialisation_base_model import TarfileSerialisationBaseModel


class AbstractMetaModel(TarfileSerialisationBaseModel):
    """An abstract base meta model consisting of multiple models."""
    def __init__(self):
        super(AbstractMetaModel, self).__init__()
    
    @abstractmethod
    def get_samples(self, x: AbstractDataSource, k: int = 1):
        """Generrate K sampleas [f_k(x) | k=1,2,...,K] for input x.
        
        Args:
            x: Input data of size (num_input_samples, input_dimension)
            k: The number of output values generated for each x by sampling k
                k models. k can be any intereget for Bayesian models. When
                self.base_models is a list of models, k cannot be larger than
                the length of that list. If None, make as many samples as 
                possible for ensemble models. None equals 1 for Bayesian models.
        Returns:
            y_samples: ndarray of shape (num_input_samples, k, output_dimension)
            """

        raise NotImplementedError()
    
    def get_model_prediction(self, data, return_std_and_margin):
        raise NotImplementedError
