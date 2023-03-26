---
layout: page
title: Evaluate
permalink: /evaluate/
---

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
