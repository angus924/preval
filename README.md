# PreVal

<div align="center"><b><i>Prevalidated ridge regression is a highly-efficient drop-in replacement<br>for logistic regression for high-dimensional data</b></i></div>

[arXiv:](https://arxiv.org/) (preprint)

> <div align="justify">Logistic regression is a ubiquitous method for probabilistic classification. However, the effectiveness of logistic regression depends upon careful and relatively computationally expensive tuning, especially for the regularisation hyperparameter, and especially in the context of high-dimensional data. We present a prevalidated ridge regression model that closely matches logistic regression in terms of classification error and log-loss, particularly for high-dimensional data, while being significantly more computationally efficient and having effectively no hyperparameters beyond regularisation. We scale the coefficients of the model so as to minimise log-loss for a set of prevalidated predictions derived from the estimated leave-one-out cross-validation error. This exploits quantities already computed in the course of fitting the ridge regression model in order to find the scaling parameter with nominal additional computational expense.</div>

```bibtex
@article{dempster_etal_2024,
  author = {Dempster, Angus and Webb, Geoffrey I and Schmidt, Daniel F},
  title  = {Prevalidated ridge regression is a highly-efficient drop-in replacement
  for logistic regression for high-dimensional data},
  year   = {2024},
}
```

## Code

### [`preval.py`](./code/preval.py)

## Example

```python
from preval import PreVal
from sklearn.preprocessing import StandardScaler

[...] # load training data {X_tr [np.float32], Y_tr}
[...] # load test data {X_te [np.float32]}

scaler = StandardScaler()
X_tr = scaler.fit_transform(X_tr)
X_te = scaler.transform(X_te)

model = PreVal()
model.fit(X_tr, Y_tr)

model.predict_proba(X_te) # probabilities
model.predict(X_te)       # class predictions
```

## Requirements

* Python
* NumPy

## Results

* **tabular data** (*OpenML*)
    * [original features](./results/openml_original.csv)
    * [original features + interactions](./results/openml_interactions.csv)
* [**microarray data**](./results/cumida.csv) (*CuMiDa*)
* **image data**
    * *MNIST*
        * [original](./results/mnist_original.csv)
        * [projection](./results/mnist_projection.csv)
    * *Fashion-MNIST*
        * [original](./results/fashion-mnist_original.csv)
        * [projection](./results/fashion-mnist_projection.csv)
* [**time series data**](./results/ucr.csv) (UCR)