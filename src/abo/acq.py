import numpy as np
from skopt.acquisition import gaussian_ei

__all__ = [
  'ei_grad', 'ei_with_prior_grad',
  'ei', 'ei_with_prior'
]

def ei_grad(model, known_points, known_values):
  if len(known_values) > 0:
    y_opt = np.min(known_values)
  else:
    y_opt = 0.0

  def neg_ei(x):
    if len(known_values) > 0:
      a, grad = gaussian_ei(x.reshape(1, -1), model=model, y_opt=y_opt, return_grad=True)
      return -a, -grad
    else:
      return -1.0, np.zeros(x.shape[0])

  return neg_ei

def ei_with_prior_grad(prev_model, threshold=0):
  from scipy.special import ndtr

  def ei_w_p(model, known_points, known_values):
    if len(known_values) > 0:
      y_opt = np.min(known_values)
    else:
      y_opt = 0.0

    def acq_f(x):
      if len(known_values) > 0:
        acq, acq_grad = gaussian_ei(x.reshape(1, -1), model=model, y_opt=y_opt, return_grad=True)
      else:
        acq, acq_grad = 1.0, np.zeros(x.shape[0])

      prev_mean, prev_std, prev_mean_grad, prev_std_grad = \
        prev_model.predict(x.reshape(1, -1), return_std=True, return_mean_grad=True, return_std_grad=True)

      phi = ndtr((threshold - prev_mean) / prev_std)

      dphi_dz = 1.0 / prev_std / np.sqrt(2 * np.pi) * np.exp(-(prev_mean - threshold) ** 2 / 2.0 / prev_std ** 2)
      dz_dx = -prev_mean_grad / prev_std + prev_mean / (prev_std ** 2) * prev_std_grad

      dphi_dx = dphi_dz * dz_dx

      full_grad = acq_grad * phi + acq * dphi_dx

      return -acq * phi, -full_grad

    return acq_f

  return ei_w_p

def ei_with_prior(prev_model, threshold=0):
  from scipy.special import ndtr

  def ei_w_p(model, known_points, known_values):
    if len(known_values) > 0:
      y_opt = np.min(known_values)
    else:
      y_opt = 0.0

    def acq_f(x):
      if len(known_values) > 0:
        acq = gaussian_ei(x, model=model, y_opt=y_opt)
      else:
        acq = np.zeros(x.shape[0])

      prev_mean, prev_std = prev_model.predict(x, return_std=True)

      phi = ndtr((threshold - prev_mean) / prev_std)

      return -acq * phi

    return acq_f

  return ei_w_p

def ei(model, known_points, known_values):
  if len(known_values) > 0:
    y_opt = np.min(known_values)
  else:
    y_opt = 0.0

  def neg_ei(x):
    if len(known_values) > 0:
      a = gaussian_ei(x, model=model, y_opt=y_opt)
      return -a
    else:
      return -1.0

  return neg_ei