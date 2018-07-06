import numpy as np
import matplotlib.pyplot as plt


from .bo import transform, reverse_transform

def cum_min(xs):
  mins = [xs[0]]
  for i in range(1, len(xs)):
    mins.append(
      min(mins[-1], xs[i])
    )
  return np.array(mins)

def plot_convergance(title, iteration, known_values, cost):
  from IPython import display
  display.clear_output(wait=True)

  plt.figure(figsize=(12, 6))
  plt.plot(np.cumsum(cost), cum_min(known_values), color='green', label='min')
  plt.scatter(np.cumsum(cost), cum_min(known_values), marker='o', color='green')
  plt.scatter(np.cumsum(cost), known_values, marker='o', color='blue', label='known')
  plt.legend(fontsize=18)
  plt.savefig("%s_%04d.png" % (title, iteration))
  plt.show()

def plot_bo(title, iteration, model, acq, space, known_points, known_values, cost, compare_to=None):
  model_xs = np.linspace(0, 1, num=100).reshape(-1, 1)
  xs = reverse_transform(model_xs, space)[:, 0]

  mean, std = model.predict(model_xs, return_std=True)

  ncols = 2 if compare_to is None else 3

  plt.subplots(nrows=1, ncols=ncols, figsize=(ncols * 6, 4))

  plt.subplot(1, ncols, 1)
  plt.title('%s iteration %d: gaussian process' % (title, iteration))
  plt.plot(xs, mean, color='green')
  plt.fill_between(xs, mean - std, mean + std, alpha=0.25, color='green')

  plt.scatter(np.array(known_points), np.array(known_values), marker='x', color='black')

  plt.subplot(1, ncols, 2)
  plt.title('%s iteration %d: acquisition function' % (title, iteration))
  acq_values = np.array([acq(x)[0] for x in model_xs])
  plt.plot(xs, -acq_values)

  if compare_to is not None:
    plt.subplot(1, ncols, 3)
    plt.title('%s iteration %d: alternative acquisition function' % (title, iteration))
    acq_values = np.array([compare_to(x)[0] for x in model_xs])
    plt.plot(xs, -acq_values)

  plt.savefig("%s_%04d.png" % (title, iteration))
  plt.show()