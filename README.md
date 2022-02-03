# GeneDisco: A benchmark for active learning in drug discovery

![Python version](https://img.shields.io/badge/Python-3.8-blue)
![Library version](https://img.shields.io/badge/Version-1.0.0-blue)

In vitro cellular experimentation with genetic interventions, using for example CRISPR technologies, is an essential 
step in early-stage drug discovery and target validation that serves to assess initial hypotheses about causal 
associations between biological mechanisms and disease pathologies. With billions of potential hypotheses to test, 
the experimental design space for in vitro genetic experiments is extremely vast, and the available experimental 
capacity - even at the largest research institutions in the world - pales in relation to the size of this biological 
hypothesis space. 

[GeneDisco (published at ICLR-22)](https://arxiv.org/abs/2110.11875) is a benchmark suite for evaluating active learning algorithms for experimental design in drug discovery. 
GeneDisco contains a curated set of multiple publicly available experimental data sets as well as open-source i
mplementations of state-of-the-art active learning policies for experimental design and exploration.

## GeneDisco ICLR-22 Challenge

Learn more about the GeneDisco challenge (with a total of £20'000 of prizes) for experimental design for optimally exploring the vast genetic intervention space [here](https://www.gsk.ai/genedisco-challenge/).

## Install

```bash
pip install genedisco
```

## Use

### How to Run the Full Benchmark Suite?

Experiments (all baselines, acquisition functions, input and target datasets, multiple seeds) included in GeneDisco can be executed sequentially for e.g. acquired batch size `64`, `8` cycles and a `bayesian_mlp` model using:
```bash
run_experiments \
  --cache_directory=/path/to/genedisco_cache  \
  --output_directory=/path/to/genedisco_output  \
  --acquisition_batch_size=64  \
  --num_active_learning_cycles=8  \
  --max_num_jobs=1
```
Results are written to the folder at `/path/to/genedisco_cache`, and processed datasets will be cached at `/path/to/genedisco_cache` (please replace both with your desired paths) for faster startup in future invocations.


Note that due to the number of experiments being run by the above command, we recommend execution on a compute cluster.<br/>
The GeneDisco codebase also supports execution on slurm compute clusters (the `slurm` command must be available on the executing node) using the following command and using dependencies in a Python virtualenv available at `/path/to/your/virtualenv` (please replace with your own virtualenv path):
```bash
run_experiments \
  --cache_directory=/path/to/genedisco_cache  \
  --output_directory=/path/to/genedisco_output  \
  --acquisition_batch_size=64  \
  --num_active_learning_cycles=8  \
  --schedule_on_slurm \
  --schedule_children_on_slurm \
  --remote_execution_virtualenv_path=/path/to/your/virtualenv
```

Other scheduling systems are currently not supported by default.

### How to Run A Single Isolated Experiment (One Learning Cycle)?

To run one active learning loop cycle, for example, with the `"topuncertain"` acquisition function, the `"achilles"` feature set and
the `"schmidt_2021_ifng"` task, execute the following command:
```bash
active_learning_loop  \
    --cache_directory=/path/to/genedisco/genedisco_cache \
    --output_directory=/path/to/genedisco/genedisco_output \
    --model_name="bayesian_mlp" \
    --acquisition_function_name="topuncertain" \
    --acquisition_batch_size=64 \
    --num_active_learning_cycles=8 \
    --feature_set_name="achilles" \
    --dataset_name="schmidt_2021_ifng" 
```


### How to Evaluate a Custom Acquisition Function?

To run a custom acquisition function, set `--acquisition_function_name="custom"` and `--acquisition_function_path` to the file path that contains your custom acquisition function.
```bash
active_learning_loop  \
    --cache_directory=/path/to/genedisco/genedisco_cache \
    --output_directory=/path/to/genedisco/genedisco_output \
    --model_name="bayesian_mlp" \
    --acquisition_function_name="custom" \
    --acquisition_function_path=/path/to/custom_acquisition_function.py \
    --acquisition_batch_size=64 \
    --num_active_learning_cycles=8 \
    --feature_set_name="achilles" \
    --dataset_name="schmidt_2021_ifng" 
```

...where `"/path/to/custom_acquisition_function.py"` contains code for your custom acquisition function corresponding to the [BaseBatchAcquisitionFunction interface](genedisco/active_learning_methods/acquisition_functions/base_acquisition_function.py), e.g.:

```python
import numpy as np
from typing import AnyStr, List
from slingpy import AbstractDataSource
from slingpy.models.abstract_base_model import AbstractBaseModel
from genedisco.active_learning_methods.acquisition_functions.base_acquisition_function import \
    BaseBatchAcquisitionFunction

class RandomBatchAcquisitionFunction(BaseBatchAcquisitionFunction):
    def __call__(self,
                 dataset_x: AbstractDataSource,
                 batch_size: int,
                 available_indices: List[AnyStr], 
                 last_selected_indices: List[AnyStr] = None, 
                 model: AbstractBaseModel = None,
                 temperature: float = 0.9,
                 ) -> List:
        selected = np.random.choice(available_indices, size=batch_size, replace=False)
        return selected
```
Note that the last class implementing `BaseBatchAcquisitionFunction` is loaded by GeneDisco if there are multiple valid acquisition functions present in the loaded file.

## Citation

Please consider citing, if you reference or use our methodology, code or results in your work:

    @inproceedings{mehrjou2022genedisco,
        title={{GeneDisco: A Benchmark for Experimental Design in Drug Discovery}},
        author={Mehrjou, Arash and Soleymani, Ashkan and Jesson, Andrew and Notin, Pascal and Gal, Yarin and Bauer, Stefan and Schwab, Patrick},
        booktitle={{International Conference on Learning Representations (ICLR)}},
        year={2022}
    }

### License

[License](LICENSE.txt)

### Authors

Patrick Schwab, GlaxoSmithKline plc<br/>
Arash Mehrjou, GlaxoSmithKline plc<br/>
Andrew Jesson, University of Oxford<br/>
Ashkan Soleymani, MIT

### Acknowledgements

PS and AM are employees and shareholders of GlaxoSmithKline plc.