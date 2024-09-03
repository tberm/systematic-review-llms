# Code for '*Automating Systematic Reviews: Generalisation to New Abstract Screening Tasks with Merged Expert LoRAs*'

This project explored the use of a fine-tuned LLM to screen research paper abstracts to be used in a systematic review. Experiments were carried out fine-tuning and evaluating models on datasets corresponding to completed systematic reviews. Each of these consisted of natural language inclusion criteria and a set of research paper titles and abstracts labelled as included or excluded. We focussed specifically on the ability of a model to generalise to new reviews in a zero-shot or few-shot setting, and experimented with training expert LoRA modules on particular reviews and combining several of these into one model. We found that this was an effective way to achieve positive transfer to new reviews, performing as well as multi-task learning on the same training data.

## Running the code

The project involved fine-tuning and evaluating an LLM, so most scripts are designed to run on a GPU cluster in a virtual machine. There are also several IPython notebooks that were run on a personal machine for other tasks such as preparing data and analysing the results.

GPU cluster scripts ran on Python 3.8.10 and requires the ``transformers``, ``wandb``, ``accelerate``, ``bitsandbytes``, ``peft``, ``datasets``, and ``flash-attn`` packages. The full list of packages that were installed via pip on the virtual machine can be found in the ``requirements-server.txt`` file.

The notebooks were run on Python 3.12.3 and require ``notebok``, ``matplotlib``, ``pandas``, ``numpy``, ``scipy``, ``scikit-learn`` and ``fasttext``. The full list of locally installed pip packages for the project can be found in the ``requirements-local.txt`` file.

## Project files

The two key scripts are ``finetune.py``, to fine-tune a model using LoRA (or IA3) and ``evaluate.py``, to evaluate an already-trained model.

Both scripts are fully configured through configuration YAML files, which are stored in the ``configs/`` directory. When running an experiment using one of these two scripts, setting the ``EXPERIMENT_NAME`` environment variable will cause the script to load its config from a file in the config directory with the name specified by the variable (plus a ``.yaml`` extension). If not set, the scripts will try to load the config from ``local.yaml``. One can also set ``SCRIPT_CONFIG`` variable as the path to a directory to override where configuration files are loaded from.

There is a second fine-tuning script, ``poly_finetune.py``, which is able to load several adapters into the [Polytropon](https://aclanthology.org/2023.eacl-main.49/) architecture and fine-tunes routing weights between the modules.

The above scripts all log their progress and results to Weights & Biases if this is configured in the given config file.

An example configuration file can be found at ``configs/example.yaml``.

``run_common.py``, ``models.py``, ``lora_plus.py``, ``prompt_utils.py`` and ``global_config.py`` provide additional functionality on which the fine-tuning and evaluation scripts depend.

Other utility scripts used as part of the project:

* ``compare_loras.py`` is used to calculate cosine similarities between the parameters of pre-trained LoRA modules.
* ``cross_validate.py`` runs a batch of several experiments following a k-fold cross validation approach.
* ``run_many_adapter_combos.py`` also batches many experiments, testing different combinations of adapters merged together. 
* ``get_criteria_embedding.py`` is used to extract LLM representations of review criteria.

Finally, the notebooks directory contains:

* ``review_data_clean.ipynb``, which contains the code used to process the data CSV files provided to me, including removing unwanted items and creating splits as described in the project report.
* ``cross_review_evals.ipynb``, ``results_analysis.ipynb``, ``trunc_data.ipynb``, which contain code to create figures and analyse data for the report.
* ``nb_funcs.py``, a module of helper functions used by the above.