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
python src/train.py --model mlp --dataset mnist --label_noise 0.1
```

## Get baseline
Note that the arguments for this script are the same as for train.py

```
python src/calculate_baseline.py --model mlp --dataset mnist --epochs 10 --label_noise 0.1
```

## Get influences
Note that the arguments for this script are the same as for train.py

```
python src/get_influences.py --model mlp --dataset mnist --epochs 10 --label_noise 0.1
```

## Get evaluation on noise prediction
Note that the arguments for this script are the same as for train.py, this also saves a plot of the noise prediction

```
python src/evaluate_noise_prediction.py --model mlp --dataset mnist --label_noises 0.1 0.2 0.3 0.4
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
