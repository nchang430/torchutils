from functools import wraps
import multiprocessing as mp

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import cross_validate, GridSearchCV
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


class _ArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __getitem__(self, index):
        return tuple(array[index] for array in self.arrays)

    def __len__(self):
        return self.arrays[0].shape[0]


class TorchDeviceManager:

    _use_cuda = False
    _cuda_device_sems = dict()
    _cpu_device = torch.device("cpu")

    @staticmethod
    def init_cuda(gpus=None, max_jobs_per_gpu=1):
        if not torch.cuda.is_available():
            raise RuntimeError("cuda unavailable")
        if gpus is None:
            gpus = range(torch.cuda.device_count())

        TorchDeviceManager._use_cuda = True
        cuda_sems = TorchDeviceManager._cuda_device_sems
        mp_manager = mp.Manager()

        for gpu in gpus:
            device_name = f"cuda:{gpu}"
            cuda_sems[device_name] = mp_manager.Semaphore(max_jobs_per_gpu)

    @staticmethod
    def acquire_device(device=None):
        if not TorchDeviceManager._use_cuda:
            return TorchDeviceManager._cpu_device

        cuda_sems = TorchDeviceManager._cuda_device_sems
        if device is None:
            got_device = False
            for (device_name, device_sem) in cuda_sems.items():
                got_device = device_sem.acquire(blocking=False)
                if got_device:
                    break
            if not got_device:
                device_name, device_sem = next(iter(cuda_sems.items()))
                device_sem.acquire()
            device = torch.device(device_name)
        else:
            cuda_sems[str(device)].acquire()

        return device

    @staticmethod
    def release_device(device):
        if TorchDeviceManager._use_cuda:
            TorchDeviceManager._cuda_device_sems[str(device)].release()


class TorchEstimator(BaseEstimator):
    def __init__(
        self,
        model_cls,
        model_args=None,
        loss_fun=F.mse_loss,
        optim_cls=optim.Adam,
        lr=1,
        optim_args=None,
        lr_sched_cls=None,
        lr_sched_args=None,
        train_epochs=1,
        train_batch_size=32,
        train_shuffle=True,
        train_drop_last_batch=False,
        test_batch_size=None,
    ):
        self.model = None
        self.model_cls = model_cls
        self.model_args = model_args
        self.loss_fun = loss_fun
        self.optim_cls = optim_cls
        self.lr = lr
        self.optim_args = optim_args
        self.lr_sched_cls = lr_sched_cls
        self.lr_sched_args = lr_sched_args
        self.train_epochs = train_epochs
        self.train_batch_size = train_batch_size
        self.train_shuffle = train_shuffle
        self.train_drop_last_batch = train_drop_last_batch
        self.test_batch_size = test_batch_size or train_batch_size
        self.device = None

    def fit(self, X, y):
        if self.model_args is None:
            self.model_args = dict()
        if self.optim_args is None:
            self.optim_args = dict()
        if self.lr_sched_args is None:
            self.lr_sched_args = dict()

        self.device = TorchDeviceManager.acquire_device()
        print(f"training {self} on {self.device}")

        self.model = self.model_cls(**self.model_args).to(self.device)
        optzr = self.optim_cls(
            self.model.parameters(), self.lr, **self.optim_args
        )
        if self.lr_sched_cls is not None:
            lr_sched = self.lr_sched_cls(optzr, **self.lr_sched_args)
        else:
            lr_sched = None

        train_dataset = _ArrayDataset(X, y)
        train_loader = DataLoader(
            train_dataset,
            self.train_batch_size,
            self.train_shuffle,
            pin_memory=True,
            drop_last=self.train_drop_last_batch,
        )

        for _ in range(self.train_epochs):
            if lr_sched is not None:
                lr_sched.step()

            for X_batch, y_batch in iter(train_loader):
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                yhat_batch = self.model(X_batch)
                loss = self.loss_fun(yhat_batch, y_batch)
                optzr.zero_grad()
                loss.backward()
                optzr.step()

        TorchDeviceManager.release_device(self.device)

    def predict(self, X):
        if self.model is None:
            raise RuntimeError("model not trained")

        test_dataset = _ArrayDataset(X)
        test_loader = DataLoader(
            test_dataset, self.test_batch_size, pin_memory=True
        )
        yhat = []

        TorchDeviceManager.acquire_device(self.device)
        for X_batch in iter(test_loader):
            X_batch = X_batch[0].to(self.device)
            yhat_batch = self.model(X_batch)
            yhat.append(yhat_batch.detach().cpu().numpy())
        TorchDeviceManager.release_device(self.device)

        yhat = np.vstack(yhat)
        return yhat


class FCNet(nn.Module):
    def __init__(self, h_dim):
        super().__init__()
        self.fc1 = nn.Linear(8, h_dim)
        self.fc2 = nn.Linear(h_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CNNet(nn.Module):
    def __init__(self):
        super().__init__()
        p = 28
        self.conv1 = nn.Conv2d(3, 8, 5)
        self.pool1 = nn.MaxPool2d(2)
        p = (p - 4) // 2
        self.conv2 = nn.Conv2d(8, 16, 5)
        self.pool2 = nn.MaxPool2d(2)
        p = (p - 4) // 2
        n_fc = p * p * 16
        self.fc = nn.Linear(n_fc, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x


def main():
    if torch.cuda.is_available():
        TorchDeviceManager.init_cuda()

    # Regression
    base_model = TorchEstimator(FCNet)
    param_grid = dict(
        model_args=[dict(h_dim=4), dict(h_dim=8), dict(h_dim=16)],
        lr=[0.1, 0.01, 0.001],
        train_epochs=[1, 5, 10],
    )
    model_grid = GridSearchCV(
        base_model,
        param_grid,
        scoring="neg_mean_squared_error",
        iid=True,
        cv=5,
        refit=True,
        error_score="raise",
    )

    X = np.random.randn(100, 8).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)

    cv_score = cross_validate(
        model_grid,
        X,
        y,
        scoring="neg_mean_squared_error",
        cv=3,
        n_jobs=4,
        verbose=True,
        error_score="raise",
    )
    print(cv_score)

    # Classification
    def _clf_score_wrapper(score_fun):
        @wraps(score_fun)
        def _wrapper(y_true, y_pred, *args, **kwargs):
            y_pred = y_pred.argmax(axis=1)
            return score_fun(y_true, y_pred, *args, **kwargs)

        return _wrapper

    score_fun = make_scorer(_clf_score_wrapper(accuracy_score))

    base_model = TorchEstimator(CNNet, loss_fun=F.cross_entropy)
    param_grid = dict(lr=[0.1, 0.01, 0.001], train_epochs=[1, 5, 10])
    model_grid = GridSearchCV(
        base_model, param_grid, scoring=score_fun, iid=True, cv=5, refit=True
    )

    X = np.random.rand(100, 3, 28, 28).astype(np.float32)
    y = np.random.randint(0, 10, 100)

    cv_score = cross_validate(
        model_grid, X, y, scoring=score_fun, cv=3, n_jobs=4, verbose=True
    )
    print(cv_score)


if __name__ == "__main__":
    main()
