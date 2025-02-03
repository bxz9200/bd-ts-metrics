# Time Series Metrics


## Contents
- [Install](#install)
- [Evaluation](#evaluation)

## Install

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


## Evaluation
Data must be in ".csv". Rows are time steps and columns are features/channels/series (whatever you call it). --sql stands for the sequence length of one single time-series data.
```
python main.py --rdp /PATH/real_data.csv --sdp /PATH/synthetic_data.csv --sql Seqence_Length
```

Metrics details are listed [here](https://www.notion.so/betterdataai/TS-V1-10de183a10414c668cd46db59ce95495?pvs=4#bcc5d6544efe46f18d3d12722994669f).

Refer to this [paper](https://arxiv.org/pdf/2309.03755) for knowledge and education.

Refer to [TSGBench](https://github.com/YihaoAng/TSGBench.git) for deatiled config settings and instructions.

## Acknowledgement
This repo is developed based on [TSGBench](https://github.com/YihaoAng/TSGBench.git)
