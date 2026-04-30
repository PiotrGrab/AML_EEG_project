from moabb.datasets import BNCI2014_001
from moabb.paradigms import LeftRightImagery
from sklearn.preprocessing import LabelEncoder
import numpy as np

dataset = BNCI2014_001()

paradigm = LeftRightImagery(
    fmin=8,
    fmax=30,
    resample=128
)

X, y, meta = paradigm.get_data(dataset, subjects=[1])

# normalize
mean = X.mean(axis=2, keepdims=True)
std = X.std(axis=2, keepdims=True) + 1e-8
X = (X - mean) / std

X = np.clip(X, -5, 5)

# labels
y = LabelEncoder().fit_transform(y)

print(X.shape, y.shape)
