import numpy as np
from scipy.stats import special_ortho_group

class FunScaler:
    """Testing scaling invariance and monoton f-transformations.

    If the `FunScaler` is applied after the observer is added, an invariant
    algorithm creates the very same BBOB trace.

    For testing scaling invariance we have to set::

        x0 /= scaler
        sigma0 /= scaler

    When ``x0 == 0``, which is a good choice on the bbob functions, the former
    doesn't change ``x0``.

    Examples::

    >>> import numpy as np
    >>> from fun_scaler import FunScaler
    >>> def fun0(x): return sum([xi**2 for xi in x]) - 2

    >>> x = np.asarray([.1,.2,.3])

    >>> fun = FunScaler(fun0, scaler=1e3)
    >>> assert fun(x) == fun0(1e3 * x)

    >>> fun = FunScaler(fun0, expo=2)
    >>> assert fun(x) == np.sign(fun0(x)) * np.abs(fun0(x))**2

    """
    def __init__(self, fun, scaler=1, expo=1):
        self._fun = fun
        self._scaler = scaler
        self._expo = expo
    def __getattr__(self, name):
        return getattr(self._fun, name)
    def __call__(self, x):
        val = self._fun(self._scaler * np.asarray(x))
        return np.sign(val) * np.abs(val)**self._expo


class FunTransform:
    def __init__(
        self,
        fun,
        scaler: float = 1.0,
        expo: float = 1.0,
        scaling_vec: float | np.ndarray = 1.0,
        x_offset: float | np.ndarray = 0.0,
        rotation: bool = False,
        rotation_seed: int = 42,
    ):
        self._fun = fun
        self._scaler = scaler
        self._expo = expo
        self._scaling_vec = scaling_vec
        self._x_offset = x_offset
        self._rotation = rotation
        self._rotation_seed = rotation_seed

    @property
    def scaler(self):
        return self._scaler

    @property
    def expo(self):
        return self._expo

    def transform(self, x):
        x = self._rotate(np.asarray(x))
        return (self._scaler * self._scaling_vec * x) + self._x_offset

    def transform_inverse(self, z):
        x = (np.asarray(z) - self._x_offset) / self._scaler / self._scaling_vec
        return self._rotate_inverse(x)

    def get_isotropic_cov_matrix(self, dim: int):
        d = np.asarray(self._scaling_vec, dtype=float)
        if d.ndim == 0:
            d = np.full(dim, float(d))
        R = self._get_rot_matrix(dim) if self._rotation else np.identity(dim)
        return (R.T * d**-2) @ R

    def _rotate(self, x):
        if not self._rotation:
            return x
        R = self._get_rot_matrix(dim=len(x))
        return R.dot(x)

    def _rotate_inverse(self, x):
        if not self._rotation:
            return x
        R = self._get_rot_matrix(dim=len(x))
        return R.T.dot(x)

    def _get_rot_matrix(self, dim=None) -> np.ndarray:
        if getattr(self, "_rot_matrix", None) is None or self._rot_matrix.shape[0] != dim:
            self._rot_matrix = special_ortho_group.rvs(dim=dim, random_state=self._rotation_seed)
        return self._rot_matrix

    def __getattr__(self, name):
        return getattr(self._fun, name)

    def __call__(self, x):
        val = self._fun(self.transform(x))
        return np.sign(val) * np.abs(val) ** self._expo
