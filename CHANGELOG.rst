0.2.4 (2022-11-23)
++++++++++++++++++
- fix masking in calibration with `simple=True`
- add `check_calibration` function
- make `calibrate` a standalone function

0.2.3 (2022-11-23)
++++++++++++++++++
- add minimum iterations to calibration optimizer

0.2.2 (2022-11-13)
++++++++++++++++++
- remove `numba` dependency

0.2.1 (2022-11-13)
++++++++++++++++++
- No-JAX version of the code is now available
- add tolerance arg to stop JAX calibration earlier based on running variance

0.1.11, 0.2.0 (2022-11-08)
++++++++++++++++++
- Rewrite calibration in JAX

0.1.10 (2022-11-02)
++++++++++++++++++
- Incorporate label noise into calibration using `noisyopt`

0.1.9 (2022-11-02)
++++++++++++++++++
- Parallelize calibration

0.1.8 (2022-11-02)
++++++++++++++++++
- New calibration method using regularized quantile matching

0.1.6, 0.1.7 (2022-10-31)
++++++++++++++++++
- Fix bug with calibration application

0.1.5 (2022-06-28)
++++++++++++++++++
- Various bug fixes

0.1.4 (2022-06-22)
++++++++++++++++++
- Add option to get biases out of predict function

0.1.3 (2022-06-22)
++++++++++++++++++
- Add inverse sum of variance sample weighting when errors on labels are provided
- Add labels to progress bars
- Prefer "threads" backend for joblib

0.1.2 (2022-06-21)
++++++++++++++++++
- Scale labels internally

0.1.1 (2022-06-21)
++++++++++++++++++
- Support multiple outputs

0.1.0 (2022-06-11)
++++++++++++++++++
- Initial release
