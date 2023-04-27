
<p align="center">
<img src="./site/assets/logo.png" alt="TABLET logo" width="600"/><img src="./site/assets/ucilogo.png" alt="UCI NLP Logo" width="150">
</p>

# TABLET: Learning From Instructions For Tabular Prediction With Language Models 

Welcome to the TABLET github! The goal of this project is to benchmark progress on instruction learning for tabular prediction. 
Hopefully, we can create models that solve tabular prediction tasks using instructions and few labeled examples.

## Links 
- Check out the project [website](https://dylanslacks.website/Tablet) 🖥️
- Read the TABLET [paper](temp) ✨
- Play with the TABLET [demo](https://nlp.ics.uci.edu/tablet/) 🚀

## Overview

While many prediction problems require the use of tabular data, often, gathering sufficient training data can be a challenge task, due to costs or privacy issues. Large language models (LLMs) offer considerable world knowledge due to their pre-training and could help improve sample efficiency for these problems. Still, these models are often not completely aligned with many tabular prediction tasks because of model biases from pre-training and lack of information about the task, hurting their performance in the zero and few shot settings.

What if we could use task instructions to help bridge this gap? That’s where TABLET comes in. TABLET is a living benchmark of tabular datasets annotated with task instructions for evaluating how well LLMs utilize instructions for improving performance on tabular prediction tasks.

### What is TABLET?

TABLET is a living benchmark of tabular prediction tasks annotated with instructions. TABLET provides the tools to evaluate models on current tasks and contribute new tasks. The goal is to help researchers develop techniques that improve the sample efficieny of LLMs on tabular prediction.

## Citation

If TABLET is useful to your work, please cite us.

```latex
@article{tabletSlack23,
         Author = {Dylan Slack and Sameer Singh},
         Title = {TABLET: Learning From Instructions For Tabular Data},
         Year = {2023},
         journal = {arXiv},
}
```

## Installation

### Getting the data

To download the data, clone [the github repository](https://github.com/dylan-slack/Tablet).

```shell
git clone https://github.com/dylan-slack/Tablet.git
```

Once this completes, the data is stored in this path.

```shell
Tablet/data/benchmark
```

### Installing TABLET

Please use `Python>=3.9`. Because of a quirk in one of the packages, please do not use `Python=3.9.7`. Also, ensure you have `pip>=23.0.1`.

```shell
conda create -n tablet python=3.9.6
conda activate tablet
pip install --upgrade pip
```

If you want to install the tablet package from source, navigate into the TABLET package directory and install.

```shell
cd Tablet
python3 -m pip install -e .
```

Otherwise, you can install from PyPI with pip. [Note: not released yet]

```shell
pip install tablet-benchmark
```

### Completing the benchmark

Unfortunately, some naturally occurring instructions come from sources that are not permissively licensed and do not
permit hosting elsewhere. We provide a guide for collecting these instructions in
```shell
Tablet/fill_missing_instructions.py
```
Once this is completed, you can run 
```shell
python fill_missing_instructions.py
```
and the instructions will be added to the benchmark data.

## Evaluate

The TABLET package offers several useful features for evaluating performance of LLMs + instructions on tabular datasets.
TABLET provides code to evaluate arbitrary huggingface models on tasks and also provides tools to simply get the HuggingFace
dataset for a particular task so you can perform whatever evaluation you want.

### Task Storage

First, let's look at how the task datasets are stored in TABLET. All the tasks are stored in

```shell
Tablet/data/benchmark/performance
```

For example, the Adult task is store at 

```shell
Tablet/data/benchmark/performance/Adult
```

Within this directory, there are different directories for each instruction annotation for the Adult task. For example,
let's look at one of the prototypes generated instructions. This instruction is stored at

```shell
Tablet/data/benchmark/performance/Adult/prototypes-synthetic-performance-0
```

Instructions collected through other sources have different paths. The rulesets generated instructions all have the directory name

```shell
ruleset-synthetic-performance-*
```

And the naturally occurring instructions have

```shell
prototypes-naturallanguage-performance-*
```

Note, the usage of prototypes here is just to retain formatting consistency with the other directory names.

Within each directory, there are four files

```shell
../test.csv
../test.json
../train.csv
../train.json
```

These are the training and testing sets, stored both in their tabular formats (the .csv's) and their natural language
formats (the .json) files. Within the json files, there are each prompt component, like the header, data point serialization,
and instruction.

### Getting a HuggingFace Dataset for a Task

Here's how to use the TABLET package to get a Huggingface dataset for a particular task. Let's say we want to get one of
the Adult and Whooping Cough datasets at these locations
```shell
Tablet/data/benchmark/performance/Adult/prototypes-synthetic-performance-0
Tablet/data/benchmark/performance/A37/prototypes-synthetic-performance-0
```
We can get the test datasets as follows
```python
from Tablet import evaluate

benchmark_path = "./data/benchmark/performance/"
tasks = ['A37/prototypes-synthetic-performance-0',
         'Adult/prototypes-synthetic-performance-0']
evaluator = evaluate.Evaluator(benchmark_path=benchmark_path,
                               tasks_to_run=tasks,
                               encoding_format="flan",
                               k_shot=0)
whooping_cough, adult = evaluator.get_test_hf_datasets()
```
We can specify `k_shot` here to control how many k_shot instances are sampled from the training data and included into
    the prompts. Then, we can access the Adult test data and labels as
```python
test_data, ground_truth_labels = adult['text'], adult['label'] 
```

### Evaluating Performance on a Task

We can also directly evaluate performance on tasks. For instance, evaluating 2-shot Flan-T5 small performance on Adult
with prototypes generated instructions with 3 seeds is as follows
```python
from Tablet import evaluate

benchmark_path = "./data/benchmark/performance/"
tasks = ['Adult/prototypes-synthetic-performance-0']
evaluator = evaluate.Evaluator(benchmark_path=benchmark_path,
                               tasks_to_run=tasks,
                               encoding_format="flan",
                               results_file="my_cool_results.txt",
                               k_shot=2)
evaluator.run_eval(how_many=3)
```
The results will be appended to `my_cool_results.txt`.

## Contribute

In order to build models that can align themselves with tabular prediction problems extremely well from just instructions and
perhaps a few examples, we need many tasks. These are useful for evaluating how well we're doing and could be useful
for future supervision.

TABLET makes it easy to create new tasks by writing instructions or generating them with GPT-3 for new datasets. Here's 
how you do it.

### Creating a new task

You must have the training and testing for your task stored in pandas df's. Then, you can call `Tablet.create`. This
function will take care of creating the task for the naturally occuring instructions you provide and will also generate
instructions using GPT-3, if you would like.

```python
from Tablet import create

create.create_task(train_x,
                   eval_x,
                   train_y,
                   eval_y,
                   name=my_data_set_name,
                   header="Predict xyz.",
                   nl_instruction="Generally, people papers are grad students.",
                   categorical_columns=names_of_categorical_columns,
                   num=index_of_task,
                   num_gpt3_revisions=10,
                   openai_key_path=path_to_open_ai_key,
                   save_loc="./data/benchmark")
```

Here, `train_x` and `eval_x` are the train and test splits. Similarly, `train_y` and `eval_y` are the label columns.
This function also accepts the name of the task (e.g., things like `Adult` or `Wine`), the header describing the high level goal of the task, and the natural langauge instructions--this is the `nl_instructions` argument. You must also specify the names of the categorical columns.
The `num` argument is the index the task with this naturally occurring instruction will be stored under (e.g., `prototypes-naturallanguage-performance-{num}`).

Further, If you wish to generate instructions with GPT-3, you will need to provide an OpenAI key in a file and give the location of this
file to the `openai_key_path` argument and specify how many instructions for the prototypes and rulesets templates you
wish to create with `num_gpt3_revisions`.

### Submitting a task

To include your awesome new task, please make sure the task's files are under
```shell
./data/benchmark/performance/my_new_task
```
and submit a pull request.

Please also include a short readmd.md in folder describing the goal of the task and the license the data and instructions are
under. For instance, something like this is ideal:
```markdown
Task: Predict how many sheep someone will need to count before they fall asleep.
Data License: Apache 2.0
Instruction License: MIT
```
We'll review it and add it to the benchmark. If you would like your name & website added to the lists of tasks [on the homepage](https://dylanslacks.website/Tablet),
please mention this in the pull request as well.
