---
layout: page
title: Install
permalink: /install/
---

Please use `Python>=3.9.6`.

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

If you want to install the tablet package from source, navigate into the TABLET package directory and install.

```shell
cd Tablet
python setup.py install
```

Otherwise, you can install from PyPI with pip.

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
