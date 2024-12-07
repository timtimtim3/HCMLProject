# Human Centered Machine Learning

## Getting started

Create python 3.10 environment:

```
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```
pip install -r requirements.txt
```

## Train model

```
python src/train.py --model mlp --dataset mnist
```

## Get influences
Note that the arguments for this script are the same as for train.py

```
python src/get_influences.py --model mlp --dataset mnist
```

## Analyse

```
python src/plot.py --data ..
```

```
python src/get_op_pp.py --data ...
```

## Snellius

Install enviroment;
run job
