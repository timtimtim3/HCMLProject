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
python src/train.py --model ... --dataset ...
```

## Get influences

```
python src/get_influences.py --model ... --dataset ... --checkpoint_path
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
