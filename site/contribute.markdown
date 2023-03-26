---
layout: page
title: Contribute
permalink: /contribute/
---

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
