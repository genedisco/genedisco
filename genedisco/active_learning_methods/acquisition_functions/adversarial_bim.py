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
import torch
import numpy as np
from typing import List, AnyStr
from slingpy import AbstractDataSource, AbstractBaseModel
from genedisco.active_learning_methods.acquisition_functions.base_acquisition_function import \
    BaseBatchAcquisitionFunction

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AdversarialBIM(BaseBatchAcquisitionFunction):
    def __init__(self, args=None):
        if args is None:
            args = {}

        if 'eps' in args:
            self.eps = args['eps']
        else:
            self.eps = 0.05

        if 'verbose' in args:
            self.verbose = args['verbose']
        else:
            self.verbose = True

        if 'stop_iterations_by_count' in args:
            self.stop_iterations_by_count = args['stop_iterations_by_count']
        else:
            self.stop_iterations_by_count = 1000

        if 'gamma' in args:
            self.gamma = args['gamma']
        else:
            self.gamma = 0.35

        if 'adversarial_sample_ratio' in args:
            self.adversarial_sample_ratio = args['adversarial_sample_ratio']
        else:
            self.adversarial_sample_ratio = 0.1

        super(AdversarialBIM, self).__init__()

    def __call__(self, dataset_x: AbstractDataSource, batch_size: int, available_indices: List[AnyStr],
                 last_selected_indices: List[AnyStr], last_model: AbstractBaseModel) -> List:
        dis = np.zeros(len(available_indices)) + np.inf
        data_pool = dataset_x.subset(available_indices)

        for i, index in enumerate(available_indices[:100]):
            x = torch.as_tensor(data_pool.subset([index]).get_data()).to(device)
            dis[i] = self.cal_dis(x, last_model)

        chosen = dis.argsort()[:batch_size]
        for x in np.sort(dis)[:batch_size]:
            print(x)
        return [available_indices[idx] for idx in chosen]

    def cal_dis(self, x, last_model):
        nx = x.detach()
        first_x = torch.clone(nx)

        nx.requires_grad_()
        eta = torch.zeros(nx.shape).to(device)
        iteration = 0

        while torch.linalg.norm(nx + eta - first_x) < self.gamma * torch.linalg.norm(first_x):

            if iteration >= self.stop_iterations_by_count:
                break

            out = torch.as_tensor(last_model.get_model_prediction(nx + eta, return_multiple_preds=True)[0])
            out = torch.squeeze(out)
            variance = torch.var(out)
            variance.backward()

            eta += self.eps * torch.sign(nx.grad.data)
            nx.grad.data.zero_()
            iteration += 1
        return (eta * eta).sum()
