import numpy as np

from skopt.learning import GaussianProcessRegressor, RandomForestRegressor
from skopt.learning.gaussian_process.kernels import Matern, WhiteKernel

from scipy.optimize import fmin_l_bfgs_b

from .acq import *


def transform(x, space):
  return (x - space[None, :, 0]) / (space[:, 1] - space[:, 0])[None, :]

def reverse_transform(x, space):
  return x * (space[:, 1] - space[:, 0])[None, :] + space[None, :, 0]


def gpbo_cycle(ndim, space, target_f, n_iters=10, acq_function=ei, model=None, n_multi_start=100):
  space = np.array(space)

  if model is None:
    kernel = WhiteKernel(0.001, noise_level_bounds=[1.0e-5, 1.0e-3]) + \
             Matern(1.0, nu=1.5, length_scale_bounds=[1.0e-3, 1.0e+3])

    model = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=False, noise=None,
        n_restarts_optimizer=2
    )

  known_points = []
  known_values = []
  cost = []

  for i in range(n_iters):
    acq = acq_function(model, known_points, known_values)

    candidates = []
    for _ in range(n_multi_start):
      x0 = np.random.uniform(size=(ndim,))

      x, f, _ = fmin_l_bfgs_b(
        maxiter=1000,
        func=acq,
        x0=x0,
        approx_grad=False,
        bounds=[(0, 1)] * ndim
      )

      candidates.append((x, f))

    best = np.argmin([f for x, f in candidates])
    suggestion, _ = candidates[best]
    suggestion = reverse_transform(suggestion.reshape(1, -1), space)[0, :]

    point_cost, observed = target_f(suggestion)

    known_points.append(suggestion)
    known_values.append(observed)
    cost.append(point_cost)

    model.fit(
      transform(np.array(known_points), space),
      np.array(known_values)
    )

    yield model, acq, space, known_points, known_values, cost

def rfbo_cycle(ndim, space, target_f, n_iters=10, acq_function=ei, n_samples=int(1.0e+5), model=None):
  space = np.array(space)

  if model is None:
    model = RandomForestRegressor(n_estimators=200, n_jobs=20, min_variance=1.0e-3, random_state=1234)

  known_points = []
  known_values = []
  cost = []

  for i in range(n_iters):
    acq = acq_function(model, known_points, known_values)

    candidates = np.random.uniform(size=(n_samples, ndim,))
    f = acq(candidates)

    best = np.argmin(f)
    suggestion = reverse_transform(candidates[best].reshape(1, -1), space)[0, :]

    point_cost, observed = target_f(suggestion)

    known_points.append(suggestion)
    known_values.append(observed)
    cost.append(point_cost)

    model.fit(
      transform(np.array(known_points), space),
      np.array(known_values)
    )

    yield model, acq, space, known_points, known_values, cost