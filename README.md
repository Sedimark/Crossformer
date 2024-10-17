# Surrey_AI

This code implements the deep learning based approach for multivariate time series data forecasting.

## Project Structure

The project directory is structured as follows:

```
Surrey_AI/
├── crossformer
│   ├── __init__.py
│   ├── data_tools
│   │     ├── __init__.py
│   │     ├── data_interface.py     # data interface
│   ├── model   
│   │     ├── __init__.py
│   │     ├── crossformer.py        # crossformer interface
│   │     ├── layers
│   │     │    ├── __init__.py
│   │     │    ├── attention.py
│   │     │    ├── decoder.py
│   │     │    ├── embedding.py
│   │     │    ├── encoder.py
│   ├── utils
│   │     ├── __init__.py
│   │     ├── metrics.py    
│   │     ├── tools.py              # training tools
├── broker.csv                      # data accessed from broker
├── cfg.json                        # configs
├── run.py                          # demo script
├── .gitignore                      # Git ignore file
├── README.md                       # project README file
└── LICENSE                         # license
```

## Features

- Data preprocessing
- Utility functions for data handling
- Model training and evaluation
- Unit tests for ensuring code quality

## Clone the project
You can clone this project from the Github.
```bash
git clone git@github.com:Sedimark/Surrey_AI.git
```

## Environment
Highly recommend to use conda for environment control. Python 3.8 is used during implementation. The following commands are provided to set up the environment with replacing *myenv* by your own environment name:

```bash
cd Surrey_AI
conda create --name myenv python=3.8
conda activate myenv
pip install -e .
```

## Demo
The scrip **run.py** is used for demonstration. It provides the template of how to use the package - crossformer.


