"""test_ridge.py: tests for ridge module."""

from functools import wraps
import unittest

from sklearn.datasets import load_linnerud
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold, KFold
import torch
import torch.nn.functional as F

from torchutils.ridge import MultiRidge, RidgeCVEstimator


MSE_SCORING = lambda y, yhat: -F.mse_loss(yhat, y)


class TestInterfaceBase(unittest.TestCase):

    """Base class for interface tests."""

    N_TR_SAMPLES = 200
    N_TE_SAMPLES = 100
    N_FEATURES = 500
    N_TARGETS = 25
    LS = torch.tensor([1.0, 2.0, 3.0, 5.0, 10.0, 100.0])

    def _gen_tr_XY(self):
        X = torch.rand(self.N_TR_SAMPLES, self.N_FEATURES)
        Y = torch.rand(self.N_TR_SAMPLES, self.N_TARGETS)
        return X, Y

    def _gen_te_XY(self):
        X = torch.rand(self.N_TE_SAMPLES, self.N_FEATURES)
        Y = torch.rand(self.N_TE_SAMPLES, self.N_TARGETS)
        return X, Y

    def setUp(self):
        self.X_tr, self.Y_tr = self._gen_tr_XY()
        self.X_te, self.Y_te = self._gen_te_XY()


class TestFunctionBase(unittest.TestCase):

    """Base class for functionality tests."""

    N_TR_SAMPLES = 15
    N_TE_SAMPLES = 5
    N_FEATURES = 3
    N_TARGETS = 3
    LS = torch.logspace(-5, 5, 100)
    ASSERT_PLACES = 3

    def _gen_tr_XY(self):
        X, Y = map(torch.from_numpy, load_linnerud(return_X_y=True))
        return X[: self.N_TR_SAMPLES], Y[: self.N_TR_SAMPLES]

    def _gen_te_XY(self):
        X, Y = map(torch.from_numpy, load_linnerud(return_X_y=True))
        return X[self.N_TR_SAMPLES :], Y[self.N_TR_SAMPLES :]

    def _compare_preds(self, y1, y2):
        self.assertEqual(y1.shape, y2.shape)
        self.assertEqual(y1.dtype, y2.dtype)
        for y1_i, y2_i in zip(y1.flatten(), y2.flatten()):
            self.assertAlmostEqual(
                y1_i.item(), y2_i.item(), places=self.ASSERT_PLACES
            )

    def setUp(self):
        self.X_tr, self.Y_tr = self._gen_tr_XY()
        self.X_te, self.Y_te = self._gen_te_XY()


def cuda_test(testcase):
    """Wrapper for cuda tests."""

    def cuda_decorate(f):
        @wraps(f)
        def wrapper(self, *args, **kwargs):
            f(self, *args, **kwargs)
            self.X_tr = self.X_tr.cuda()
            self.Y_tr = self.Y_tr.cuda()
            self.X_te = self.X_te.cuda()
            self.Y_te = self.Y_te.cuda()

        return wrapper

    testcase.setUp = cuda_decorate(testcase.setUp)
    return unittest.skipUnless(torch.cuda.is_available(), "cuda unavailable")(
        testcase
    )


class TestMultiRidgeInterface(TestInterfaceBase):

    """Test inputs/outputs of MultiRidge methods."""

    def setUp(self):
        super().setUp()
        self.multi_ridge = MultiRidge(self.LS)

    def test_get_prediction_scores_return_value(self):
        self.multi_ridge.fit(self.X_tr, self.Y_tr)
        scores = self.multi_ridge.get_prediction_scores(
            self.X_te, self.Y_te, MSE_SCORING
        )
        self.assertIsInstance(scores, torch.Tensor)
        self.assertIs(scores.dtype, self.X_te.dtype)
        self.assertListEqual(list(scores.shape), [self.N_TARGETS, len(self.LS)])

    def test_predict_single_return_value(self):
        self.multi_ridge.fit(self.X_tr, self.Y_tr)
        Yhat_te = self.multi_ridge.predict_single(
            self.X_te, [0] * self.N_TARGETS
        )
        self.assertIsInstance(Yhat_te, torch.Tensor)
        self.assertIs(Yhat_te.dtype, self.X_te.dtype)
        self.assertListEqual(
            list(Yhat_te.shape), [self.N_TE_SAMPLES, self.N_TARGETS]
        )

    def test_fit_returns_self(self):
        X, Y = self._gen_tr_XY()
        ret = self.multi_ridge.fit(X, Y)
        self.assertIs(ret, self.multi_ridge)


@cuda_test
class TestMultiRidgeInterfaceCuda(TestMultiRidgeInterface):

    pass


class TestMultiRidgeFunction(TestFunctionBase):

    """Test MultiRidge by comparing with sklearn."""

    def _fit_predict(self, ridge_mr, ridge_sk, scale):
        ridge_mr.fit(self.X_tr, self.Y_tr)

        X_tr_np, X_te_np = self.X_tr.cpu().numpy(), self.X_te.cpu().numpy()
        X_tr_np = X_tr_np - ridge_mr.Xm.cpu().numpy()
        X_te_np = X_te_np - ridge_mr.Xm.cpu().numpy()
        if scale:
            X_tr_np = X_tr_np / ridge_mr.Xs.cpu().numpy()
            X_te_np = X_te_np / ridge_mr.Xs.cpu().numpy()

        ridge_sk.fit(X_tr_np, self.Y_tr.cpu().numpy())

        Yhat_te_mr = ridge_mr.predict_single(self.X_te, [0] * self.N_TARGETS)
        Yhat_te_sk = ridge_sk.predict(X_te_np)
        return Yhat_te_mr, torch.from_numpy(Yhat_te_sk)

    def test_predictions(self, scale=False):
        for l in self.LS:
            ridge_mr = MultiRidge([l], scale_X=scale)
            ridge_sk = Ridge(l, solver="svd")
            Yhat_te_mr, Yhat_te_sk = self._fit_predict(
                ridge_mr, ridge_sk, scale
            )
            self._compare_preds(Yhat_te_mr, Yhat_te_sk)

    def test_predictions_scaled(self):
        self.test_predictions(scale=True)

    def test_prediction_scores(self, ridge_mr=None):
        if ridge_mr is None:
            ridge_mr = MultiRidge(self.LS, scale_X=False)
        ridge_mr.fit(self.X_tr, self.Y_tr)
        scores_mr = ridge_mr.get_prediction_scores(
            self.X_te, self.Y_te, MSE_SCORING
        )
        for i, l in enumerate(self.LS):
            ridge_sk = Ridge(l).fit(
                self.X_tr.cpu().numpy(), self.Y_tr.cpu().numpy()
            )
            Yhat_te_sk = ridge_sk.predict(self.X_te.cpu().numpy())
            for j, yhat_te_sk_j in enumerate(Yhat_te_sk.T):
                score_ij_mr = scores_mr[j, i].item()
                score_ij_sk = -1 * mean_squared_error(
                    self.Y_te[:, j].cpu().numpy(), yhat_te_sk_j
                )
                self.assertAlmostEqual(score_ij_mr, score_ij_sk, delta=1e-1)

    def test_fit_called_multiple_times(self):
        ridge_mr = MultiRidge(self.LS, scale_X=False).fit(self.X_tr, self.Y_tr)
        self.test_prediction_scores(ridge_mr)
        ridge_mr.fit(self.X_te, self.Y_te)
        self.test_prediction_scores(ridge_mr)


@cuda_test
class TestMultiRidgeFunctionCuda(TestMultiRidgeFunction):
    pass


class TestRidgeCVEstimatorInterface(TestInterfaceBase):

    """Test inputs/ouputs of RidgeCVEstimator methods."""

    CV = KFold(5)

    def setUp(self):
        super().setUp()
        self.ridge_cve = RidgeCVEstimator(self.LS, self.CV, MSE_SCORING)

    def test_init_input_validation(self):
        with self.assertRaises(AttributeError):
            RidgeCVEstimator(self.LS.unsqueeze(0), self.CV, MSE_SCORING)
        with self.assertRaises(AttributeError):
            RidgeCVEstimator(self.LS.cpu().numpy(), self.CV, MSE_SCORING)
        with self.assertRaises(AttributeError):
            RidgeCVEstimator(self.LS.int(), self.CV, MSE_SCORING)

    def test_fit_input_wrong_shape(self):
        with self.assertRaises(AttributeError):
            self.ridge_cve.fit(self.X_tr[:, 0], self.Y_tr)
        with self.assertRaises(AttributeError):
            self.ridge_cve.fit(self.X_tr, self.Y_tr[:, 0])

    def test_fit_input_shape_mismatch(self):
        X = torch.rand(self.N_TR_SAMPLES, self.N_FEATURES)
        Y = torch.rand(self.N_TARGETS, self.N_TARGETS)
        with self.assertRaises(AttributeError):
            self.ridge_cve.fit(X, Y)

    def test_fit_input_dtype_mismatch(self):
        self.Y_tr = self.Y_tr.double()
        with self.assertRaises(AttributeError):
            self.ridge_cve.fit(self.X_tr, self.Y_tr)

    def test_fit_returns_self(self):
        ret = self.ridge_cve.fit(self.X_tr, self.Y_tr)
        self.assertIs(ret, self.ridge_cve)

    def test_predict_fails_without_fit(self):
        with self.assertRaises(RuntimeError):
            self.ridge_cve.predict(self.X_te)

    def test_predict_return_value(self):
        self.ridge_cve.fit(self.X_tr, self.Y_tr)
        Yhat_te = self.ridge_cve.predict(self.X_te)
        self.assertListEqual(
            list(Yhat_te.shape), [self.N_TE_SAMPLES, self.N_TARGETS]
        )

    def test_predict_fails_with_wrong_dtype(self):
        self.ridge_cve.fit(self.X_tr, self.Y_tr)
        X_te, _ = self._gen_te_XY()
        X_te = X_te.double()
        with self.assertRaises(RuntimeError):
            self.ridge_cve.predict(X_te)


@cuda_test
class TestRidgeCVEstimatorInterfaceCuda(TestRidgeCVEstimatorInterface):
    pass


class TestRidgeCVEstimatorFunction(TestFunctionBase):

    """Test RidgeCVEstimator by comparing with sklearn."""

    def _fit_predict(self, ridge_cve, ridge_sk, groups=None, single=False):
        Y_tr = self.Y_tr[:, :1] if single else self.Y_tr
        ridge_cve.fit(self.X_tr, Y_tr, groups)
        ridge_sk.fit(self.X_tr.cpu().numpy(), Y_tr.cpu().numpy())
        Yhat_te_cve = ridge_cve.predict(self.X_te)
        Yhat_te_sk = ridge_sk.predict(self.X_te.cpu().numpy())
        return Yhat_te_cve, torch.from_numpy(Yhat_te_sk)

    def test_single_target_kfold(self):
        cv = KFold(5)
        ridge_cve = RidgeCVEstimator(self.LS, cv, MSE_SCORING, scale_X=False)
        ridge_sk = RidgeCV(
            self.LS, cv=cv, scoring="neg_mean_squared_error", gcv_mode="svd"
        )

        Yhat_te_cve, Yhat_te_sk = self._fit_predict(
            ridge_cve, ridge_sk, single=True
        )
        self._compare_preds(Yhat_te_cve, Yhat_te_sk)
        self.assertAlmostEqual(
            self.LS[ridge_cve.best_l_idxs[0]].item(),
            ridge_sk.alpha_,
            places=self.ASSERT_PLACES,
        )

    def test_single_target_grouped_kfold(self):
        cv = GroupKFold(4)
        groups = [0] * 3 + [1] * 4 + [2] * 2 + [3] * 6
        ridge_cve = RidgeCVEstimator(self.LS, cv, MSE_SCORING, scale_X=False)
        ridge_sk = RidgeCV(
            self.LS,
            cv=list(cv.split(self.X_tr, groups=groups)),
            scoring="neg_mean_squared_error",
            gcv_mode="svd",
        )
        Yhat_te_cve, Yhat_te_sk = self._fit_predict(
            ridge_cve, ridge_sk, groups, single=True
        )
        self._compare_preds(Yhat_te_cve, Yhat_te_sk)
        self.assertAlmostEqual(
            self.LS[ridge_cve.best_l_idxs[0]].item(),
            ridge_sk.alpha_,
            places=self.ASSERT_PLACES,
        )

    def test_single_l_kfold(self):
        cv = KFold(5)
        for l in self.LS:
            ridge_cve = RidgeCVEstimator(
                torch.tensor([l]), cv, MSE_SCORING, scale_X=False
            )
            ridge_sk = RidgeCV(
                [l], cv=cv, scoring="neg_mean_squared_error", gcv_mode="svd"
            )
            Yhat_te_cve, Yhat_te_sk = self._fit_predict(ridge_cve, ridge_sk)
            self._compare_preds(Yhat_te_cve, Yhat_te_sk)


@cuda_test
class TestRidgeCVEstimatorFunctionCuda(TestRidgeCVEstimatorFunction):

    pass
