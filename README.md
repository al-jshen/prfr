# prfr

Probabilistic random forest regressor: random forest model that accounts for errors in predictors and yields calibrated probabilistic predictions.

## Installation

```bash
pip install prfr
```

## Example usage

```python
import numpy as np
from prfr import ProbabilisticRandomForestRegressor, split_arrays

x_obs = np.random.uniform(0., 10., size=10000).reshape(-1, 1)
x_err = np.random.exponential(1., size=10000).reshape(-1, 1)
y_obs = np.random.normal(x_obs, x_err).reshape(-1) * 2. + 1.

train_arrays, test_arrays, valid_arrays = split_arrays(x_obs, x_err, y_obs, test_size=0.2, valid_size=0.2)
x_train, x_err_train, y_train = train_arrays
x_test, x_err_test, y_test = test_arrays
x_valid, x_err_valid, y_valid = valid_arrays

model = ProbabilisticRandomForestRegressor(n_estimators=250, n_jobs=-1)
model.fit(x_train, y_train, eX=x_err_train)
model.calibrate(x_valid, y_valid, eX=x_err_valid)
model.fit_bias(x_valid, y_valid, eX=x_err_valid)

pred = model.predict(x_test, eX=x_err_test)
pred_bounds = np.quantile(pred, [0.16, 0.84], axis=1)
pred_mean = np.mean(pred, axis=1)

print(pred.shape)
```
