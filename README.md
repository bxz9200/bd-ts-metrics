# Time Series Metrics


## Contents
- [Install](#install)
- [Usage](#usage)


## Install
### Pip installation:
You can pip install the repo using:
```bash
pip install "git+https://github.com/bxz9200/bd-ts-metrics"
```


### Clone this repo:
Or install by cloning this repo:
1. Clone this repository and navigate to time-series-synthetic folder
```bash
git clone https://github.com/betterdataai/ts-metrics.git
cd ts-metrics
```

2. Install Package
```Shell
conda create -n ts-metrics
conda activate ts-metrics
pip install --upgrade pip
pip install -r requirements.txt
```


## Usage
### Pip installation:
Data must be in ".csv" and **must comply** the format defined [here](https://www.notion.so/betterdataai/TS-V1-10de183a10414c668cd46db59ce95495?pvs=4#7f02e54cde0b41f2b70adb52b511d1fe).

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

### Clone this repo:
Data must be in ".csv". Rows are time steps and columns are features/channels/series (whatever you call it). --sql stands for the sequence length of one single time-series data.
All results will be stored in the result folder.
```
python evaluate.py --config /PATH/config.ymal --rdp /PATH/real_data.csv --sdp /PATH/synthetic_data.csv --sql Seqence_Length
```

Metrics details are listed [here](https://www.notion.so/betterdataai/TS-V1-10de183a10414c668cd46db59ce95495?pvs=4#bcc5d6544efe46f18d3d12722994669f).

Refer to this [paper](https://arxiv.org/pdf/2309.03755) for knowledge and education.

Refer to [TSGBench](https://github.com/YihaoAng/TSGBench.git) for deatiled config settings and instructions.

## Acknowledgement
This repo is developed based on [TSGBench](https://github.com/YihaoAng/TSGBench.git).
