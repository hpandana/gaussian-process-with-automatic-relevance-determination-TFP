# gaussian-process-with-automatic-relevance-determination-TFP

An implementation of GP regression with automatic relevance determination (ARD) in `tensorflow_probability` (TFP). Code is [here](tfp/src/GPR.py).

Dataset was taken from sklearn `load_diabetes` toy [dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html#sklearn.datasets.load_diabetes).

TFP implementation is [here](tfp). Plots of the length_scale posterior are in this [ipynb](tfp/02_plot_length_scale_ard.ipynb). Stan implementation is [here](stan), and its plots of the length scale posterior are in this [ipynb](stan/03_plot_length_scale_ard.ipynb).