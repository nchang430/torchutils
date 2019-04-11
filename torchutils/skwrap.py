from functools import wraps
import multiprocessing as mp

import numpy as np
import uuid
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


class LogGridSearchCV(GridSearchCV):
    def fit(self, X, y=None, groups=None, **fit_params):
        self.refit = False
        super().fit(X, y, groups, **fit_params)
        results = self.cv_results_
        refit_metric = "score"

        self.best_index_ = results["rank_test_%s" % refit_metric].argmin()
        self.best_params_ = results["params"][self.best_index_]
        self.best_score_ = results["mean_test_%s" % refit_metric][self.best_index_]

        self.best_estimator_ = clone(self.estimator).set_params(**self.best_params_)
        refit_start_time = time.time()
        if y is not None:
            self.best_estimator_.fit(X, y, log=True, **fit_params)
        else:
            self.best_estimator_.fit(X, log=True, **fit_params)
        refit_end_time = time.time()
        self.refit_time_ = refit_end_time - refit_start_time
        self.refit = True

        return self


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
        key_model_args=None,
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
        tb_dir=None,
        model_path=None,
    ):
        self.model = None
        self.model_cls = model_cls
        self.key_model_args = key_model_args
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
        self.tb_dir = tb_dir
        self.model_path = model_path
        if not hasattr(self, "model_id"):
            self.model_id = str(uuid.uuid4())[:13]

    def fit(self, X, y, log=False):
        self.device = TorchDeviceManager.acquire_device()
        print(f"training {self} on {self.device}")
        # log final model training tb and tb setup
        tb_writer = None
        if log:
            print(f"training {self} on {self.device}")
            try:
                tb_dir = path.Path(self.tb_dir) / self.model_id
                tb_dir.mkdir()
                tb_writer = SummaryWriter(str(tb_dir), purge_step=0, max_queue=0)
                log.debug(f"Tensorboard writer initialized for {self.tb_dir}")
                repo = git.Repo(search_parent_directories=True)
                sha = repo.head.object.hexsha
                with open(str(tb_dir) + f"/{self.model_id}_args.txt", "w") as f:
                    f.write(f"Git Commit: {sha}\n\nModel Parameters:\n")
                    for k, v in self.__dict__.items():
                        f.write(f"\t{str(k)}: {str(v)}\n")
            except OSError as e:
                log.critical(f"Cannot access tensorboard directory: {e}")
                sys.exit(1)

        if self.model_args is None:
            self.model_args = dict()
        if self.optim_args is None:
            self.optim_args = dict()
        if self.lr_sched_args is None:
            self.lr_sched_args = dict()

        self.model = self.model_cls(**self.key_model_args, **self.model_args).to(
            self.device
        )
        optzr = self.optim_cls(self.model.parameters(), self.lr, **self.optim_args)
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
            mean_ep_loss = 0
            epoch_pbar = trange(len(train_loader), desc=f"Epoch {ep}", disable=not log)

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

                # update loss on tensor progress bar
                loss = loss.item()
                mean_ep_loss += loss / len(train_loader)
                epoch_pbar.set_postfix(loss=loss)
                epoch_pbar.update(1)
            # training acc on all training data for current epoch
            if log and tb_writer is not None:
                y_pred = self.predict(X, acquire_device=False)
                y_pred = y_pred.argmax(axis=1)
                train_acc = accuracy_score(y, y_pred)
                tb_writer.add_scalar(f"loss", mean_ep_loss, ep)
                tb_writer.add_scalar(f"train_acc", train_acc, ep)

            epoch_pbar.close()

            # saving model
        if log:
            print("Saving model")
            model_path = self.model_path / f"{self.model_id}.pth"
            torch.save(self.model.state_dict(), model_path)

        TorchDeviceManager.release_device(self.device)

    def predict(self, X, acquire_device=True):
        if self.model is None:
            raise RuntimeError("model not trained")

        test_dataset = _ArrayDataset(X)
        test_loader = DataLoader(test_dataset, self.test_batch_size, pin_memory=True)
        yhat = []

        if acquire_device:
            TorchDeviceManager.acquire_device(self.device)
        for X_batch in iter(test_loader):
            X_batch = X_batch[0].to(self.device)
            yhat_batch = self.model(X_batch)
            yhat.append(yhat_batch.detach().cpu().numpy())
        if acquire_device:
            TorchDeviceManager.release_device(self.device)

        yhat = np.vstack(yhat)
        return yhat


class FCNet2(nn.Module):
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


class FCNet(nn.Module):
    def __init__(self, n_channels, n_output, layer_sizes, nonlin=F.relu):
        super().__init__()
        layer_sizes = [n_channels] + layer_sizes + [n_output]
        self.layers = []
        for i, (l, lnext) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            fc = nn.Linear(l, lnext)
            self.layers.append(fc)
            setattr(self, f"fc{i}", fc)
        self.nonlin = nonlin

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            layer = self.layers[i]
            x = self.nonlin(layer(x))
        last_layer = self.layers[-1]
        x = last_layer(x)
        return x
