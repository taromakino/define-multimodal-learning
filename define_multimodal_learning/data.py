import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def sigmoid(x, beta):
    return 1 / (1 + np.exp(-(x - beta)))


def normalize(x_train, x_val, x_test):
    x_mean, x_sd = x_train.mean(axis=0), x_train.std(axis=0)
    x_train = (x_train - x_mean) / x_sd
    x_val = (x_val - x_mean) / x_sd
    x_test = (x_test - x_mean) / x_sd
    return x_train, x_val, x_test


def to_torch(*arrs):
    out = [torch.tensor(arr)[:, None] if len(arr.shape)== 1 else torch.tensor(arr) for arr in arrs]
    if len(out) == 1:
        return out[0]
    else:
        return out


def make_dataloader(data_tuple, batch_size, n_workers, is_train):
    return DataLoader(TensorDataset(*data_tuple), shuffle=is_train, batch_size=batch_size, num_workers=n_workers,
        pin_memory=True, persistent_workers=True)


def make_matched_data(rng, n_examples, beta):
    cov = np.array([
        [1.00, 0.10, 0.90],
        [0.10, 1.00, 0.90],
        [0.90, 0.90, 1.00],
    ])
    x_noise = rng.multivariate_normal(np.zeros(3), cov, n_examples)

    y = rng.binomial(1, 0.5, n_examples)
    x0 = 0.5 * y + x_noise[:, 0]
    x1 = -0.5 * y + x_noise[:, 1]
    xp = y + x_noise[:, 2]
    x = np.c_[x0, x1, xp]

    x_magnitude = np.abs(x.sum(axis=1))
    x_magnitude = (x_magnitude - x_magnitude.mean()) / x_magnitude.std()
    m_prob = sigmoid(x_magnitude, beta)
    m = rng.binomial(1, m_prob)
    matched_idxs = np.where(m == 1)
    x, y = x[matched_idxs], y[matched_idxs]
    return x.astype("float32"), y.astype("float32")


def make_data(seed, n_examples, beta, batch_size, n_workers):
    n_total = sum(n_examples)
    rng = np.random.RandomState(seed)
    x_total, y_total = [], []
    count = 0
    while count < n_total:
        x, y = make_matched_data(rng, n_total, beta)
        count += len(x)
        x_total.append(x)
        y_total.append(y)
    x_total = np.concatenate(x_total)[:n_total]
    y_total = np.concatenate(y_total)[:n_total]

    n_train, n_val, n_test = n_examples
    x_train, y_train = x_total[:n_train], y_total[:n_train]
    x_val, y_val = x_total[n_train:n_train+n_val], y_total[n_train:n_train+n_val]
    x_test, y_test = x_total[n_train+n_val:], y_total[n_train+n_val:]

    x_train, x_val, x_test = to_torch(*normalize(x_train, x_val, x_test))
    y_train, y_val, y_test = to_torch(y_train, y_val, y_test)
    data_train = make_dataloader((x_train, y_train), batch_size, n_workers, True)
    data_val = make_dataloader((x_val, y_val), batch_size, n_workers, False)
    data_test = make_dataloader((x_test, y_test), batch_size, n_workers, False)
    return data_train, data_val, data_test