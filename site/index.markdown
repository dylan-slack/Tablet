---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: home
---


<p align="center">
  <img height="160" src="assets/logo.png" alt="TABLET Logo">
</p>

<center>
{% include button.html button_name="Paper" button_link="https://arxiv.org/abs/2304.13188" button_class="primary" %}&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{% include button.html button_name="Code" button_link="https://github.com/dylan-slack/Tablet" button_class="primary" %}
</center>

## Overview

While many prediction problems require the use of tabular data, often, gathering sufficient training data can be a challenge task, due to costs or privacy issues.
Large language models (LLMs) offer considerable world knowledge due to their pre-training and can help improve sample efficiency for these problems.
Still, these models are often not completely aligned with many tabular prediction tasks because of model biases from pre-training and lack of information about the task, hurting their performance in the zero and few shot settings.
Task _instructions_ are ideal for bridging this gap, because they offer LLMs context about the task. That's where **TABLET** comes in. 
TABLET is a living benchmark of tabular datasets annotated with task instructions for evaluating how well LLMs utilize
instructions for improving performance.

TABLET contains 20 diverse classification tasks.
Each task is annotated with multiple instructions. We collected these instructions from naturally occurring sources and generated them with GPT-3, and they vary in their
phrasing, granularity, and technicality.

The results [in our paper](https://arxiv.org/abs/2304.13188) demonstrate instructions are highly useful for improving LLM performance on tabular prediction. Still, LLMs underperform fully supervised models fit on all the training data, and we analyze several of the shortcomings of LLMs which contribute to this. Hopefully, we develop models that close this gap! 


## What can you do with TABLET?

1. **Evaluate**. TABLET provides the tools to collate tabular instances into prompts for LLMs. TABLET also makes it easy to evaluate performance across different instructions.
2. **Compare**. TABLET enables users to compare performance across LLMs in zero and few-shot settings, or against fully supervised models like XGBoost that have access to _all_ the training data.
3. **Contribute**. TABLET provides a simple API for creating new prediction tasks from tabular datasets. TABLET supports both instructions written by users and can also generate instructions for tasks.

## Getting Started

- [Paper](https://arxiv.org/abs/2304.13188): Read our evaluation on TABLET 📝
- [Demo](https://dylanslacks.website/Tablet/demo/): Explore LLM predictions on TABLET 🕵️
- [Install](https://dylanslacks.website/Tablet/install/): Install TABLET 💾
- [Evaluate](https://dylanslacks.website/Tablet/evaluate/): Follow a tutorial on how to evaluate an LLM on TABLET 💯
- [Contribute](https://dylanslacks.website/Tablet/contribute/): Follow a tutorial on how to contribute a new task to TABLET ✏️

## Tasks

Here are the current tasks in TABLET, a short description, the creator, and the number of each type of instruction. Contribute more tasks to TABLET by following the instructions in [contribute](contribute), and I will add the task and your name and website as the person who contributed it (if you want).

<iframe class="airtable-embed" src="https://airtable.com/embed/shr42xpzNOKcUmXse?backgroundColor=blue&viewControls=on" frameborder="0" onmousewheel="" width="100%" height="533" style="background: transparent; border: 1px solid #ccc;"></iframe>

## Citation

```latex
@article{tabletSlack23,
Author = {Dylan Slack and Sameer Singh},
Title = {TABLET: Learning From Instructions For Tabular Dataset Tasks},
Year = {2023},
journal = {arXiv},
}
```

## Authors

<style>
    .img-container {
        border-radius: 25px;
        background: #ff9797;
        text-align: center;
        border: 1px solid black;
        padding: 10px;
        display: inline-block;
        margin: 10px;
    }
</style>

<center>
    <div class="img-container">
        <a href="https://dylanslacks.website">
            <img src="https://dylanslacks.website/images/me.jpeg" height="150px" alt="Dylan Slack profile photo.">
        </a>
        <div class="caption">Dylan Slack</div>
    </div>
    <div class="img-container">
        <a href="http://sameersingh.org">
            <img src="http://sameersingh.org/img/face/mr-singh-face.jpg" height="150px" alt="Sameer Singh profile photo.">
        </a>
        <div class="caption">Sameer Singh</div>
    </div>
</center>
