# prfr

Probabilistic random forest regressor: random forest model that accounts for errors in predictors and labels, yields calibrated probabilistic predictions, and corrects for bias.

## Installation

```bash
pip install prfr
```

OR

```bash
pip install git+https://github.com/al-jshen/prfr
```

## Example usage

```python
import numpy as np
from prfr import ProbabilisticRandomForestRegressor, split_arrays

x_obs = np.random.uniform(0., 10., size=10000).reshape(-1, 1)
x_err = np.random.exponential(1., size=10000).reshape(-1, 1)
y_obs = np.random.normal(x_obs, x_err).reshape(-1, 1) * 2. + 1.
y_err = np.ones_like(y_obs)

train, test, valid = split_arrays(x_obs, y_obs, x_err, y_err, test_size=0.2, valid_size=0.2)

model = ProbabilisticRandomForestRegressor(n_estimators=250, n_jobs=-1)
model.fit(train[0], train[1], eX=train[2], eY=train[3])
model.fit_bias(valid[0], valid[1], eX=valid[2])
model.calibrate(valid[0], valid[1], eX=valid[2])

pred = model.predict(test[0], eX=test[2])
pred_qtls = np.quantile(pred, [0.16, 0.5, 0.84], axis=-1)

print(pred.shape)
```
