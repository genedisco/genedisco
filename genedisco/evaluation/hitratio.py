"""
Copyright (C) 2022  Arash Mehrjou, GlaxoSmithKline plc

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions
 of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""
import os
import pickle
import numpy as np
from typing import Optional, AnyStr
from slingpy.evaluation.metrics.abstract_metric import AbstractMetric

class HitRatio(AbstractMetric):
    """
    A metric to measure the ratio of the top mover genes selected by the acquisition function.
    """
    
    def get_abbreviation(self) -> AnyStr:
        return "HR"
    
    @staticmethod
    def evaluate(top_movers_filepath:AnyStr, super_dir_to_cycle_dirs: AnyStr) -> np.ndarray:
    
        with open(top_movers_filepath, "rb") as f:
            top_mover_indices = pickle.load(f)
        top_mover_set = set(top_mover_indices)
        num_top_hits = len(top_mover_indices)
        num_AL_cycles = get_num_AL_cycles(super_dir_to_cycle_dirs)
        selected_indices_per_cycle = get_cumulative_selected_indices(
        super_dir_to_cycle_dirs)
        cumulative_top_hit_ratio = []
        for c in range(num_AL_cycles):
            selected_indices = selected_indices_per_cycle[c]            
            num_of_hits = num_top_hits - len(top_mover_set - set(selected_indices))
            cumulative_top_hit_ratio.append(num_of_hits/num_top_hits)
        return cumulative_top_hit_ratio[-1] # returns the top hit ratio of the current cycle
            

def get_cumulative_selected_indices(super_dir_to_cycle_dirs: AnyStr):
    """ Get a list of selected indiced at cycles of active learning.
    
        Args:
            super_dir_to_cycle_dirs: The dir in which the cycle dirs are saved.
            seed: The seed of the experiment.
            
        Return a concatenated list of the saved selected indices so far.
    """
    num_AL_cycles = get_num_AL_cycles(super_dir_to_cycle_dirs)
    selected_indices_per_cycles = []
    for c in range(num_AL_cycles):
        filename = os.path.join(super_dir_to_cycle_dirs, "cycle_" + str(c), "selected_indices.pickle")
        with open(filename, "rb") as f:
            selected_indices = pickle.load(f)
            # selected_indices = [x.decode("utf-8") for x in selected_indices] # Uncomment this line if the stored Gene names are byte strings.
            selected_indices_per_cycles.append(selected_indices)
    return selected_indices_per_cycles
    

def get_num_AL_cycles(super_dir_to_cycle_dirs: AnyStr):
    """Get the number of cycles stored in the provided dir.
    """
    all_subdirs = list(os.walk(super_dir_to_cycle_dirs))[0][1]
    cycle_subdirs = [folder_name for folder_name in all_subdirs if folder_name.startswith("cycle")]
    num_AL_cycles = len(cycle_subdirs)
    return num_AL_cycles