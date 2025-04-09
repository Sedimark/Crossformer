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
├── crossformer
│   ├── all_weather_values.csv      # all weather data (values-only)
│   ├── broker_values.csv           # part data accessed from broker (values only)
│   ├── broker.csv                  # part data accessed from broker
│   ├── WeatherInforamtion.csv      # all data accessed from broker
├── cfg.json                        # configs
├── pyproject.toml                  # project dependencies
├── main.py                         # The main file
├── setup.py                        # setup script
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
Highly recommend to use conda for environment control. **Python 3.8** is used during implementation and the **later** versions can work as well. The following commands are provided to set up the environment with replacing *myenv* by your own environment name:

```bash
cd Surrey_AI
conda create --name myenv python=3.8
conda activate myenv
```

## Install the package
The package can be installed using the following command:

```bash
cd Surrey_AI
pip install -e .
```

## Demo
The scrip **main.py** is used for demonstration. It provides the template of how to use the package - crossformer. Also, it displays the core function of the AI asset for model fitting and prediction.

Assume that the environment has been activated, the UI platform for visualize AI model training is MLFlow. To turn on the mlflow server, please follow the command below, before you start your model's training.

```bash
mlflow server --host 127.0.0.1 --port 5000
```

After starting the mlflow, you can run the demo using **main.py**.

```bash
python main.py
```




