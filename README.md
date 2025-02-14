# Time Series Metrics


## Contents
- [Install](#install)
- [Usage](#usage)
- [Metrics](#metrics)


## Install
You can pip install the repo using:
```bash
pip install "git+https://github.com/bxz9200/bd-ts-metrics"
```



## Usage
Data must be in ".csv" and **must comply** the format defined in [Input format](https://www.notion.so/betterdataai/TS-V1-10de183a10414c668cd46db59ce95495?pvs=4#7f02e54cde0b41f2b70adb52b511d1fe).

After installation, follow the code below for evaluation:
```python
from bdtsmetrics import bd_ts_metrics

# Define the path of config and data files
config = "PATH/config.yaml" # both yaml and json files are supported
real_data = "PATH/real_data.csv"
syn_data = "PATH/synthetic_data.csv"

# Run evaluation
my_metrics = bd_ts_metrics.tsMetrics(config=config, real_data=real_data, syn_data=syn_data)
my_metrics.evaluate()
```
**Report.html** will be generated in your local folder and detailed results will be stored in folder **./result**. 

Refer to **config/config.yaml** for information of the config file. To run the evaluation correctly, make sure you correctly define the **'seq_len'** and **'non_ts_cols'** in the config file.


## Metrics
Metrics details are listed [here](https://www.notion.so/betterdataai/TS-V1-10de183a10414c668cd46db59ce95495?pvs=4#bcc5d6544efe46f18d3d12722994669f).

Refer to this [paper](https://arxiv.org/pdf/2309.03755) for knowledge and education.

Refer to [TSGBench](https://github.com/YihaoAng/TSGBench.git) for deatiled config settings and instructions.

## Acknowledgement
This repo is developed based on [TSGBench](https://github.com/YihaoAng/TSGBench.git).
