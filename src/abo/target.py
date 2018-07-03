import numpy as np

from scipy.stats import wasserstein_distance

from .pythia import get_data

def get_losses(discriminator, x1, x2):
  p1 = discriminator.predict_proba(x1)[:, 1]
  ls1 = -np.log(p1 + 1.0e-6)

  p2 = discriminator.predict_proba(x2)[:, 0]
  ls2 = -np.log(p2 + 1.0e-6)

  return ls1, ls2

def train(discriminator_factory, X_true_train, X_gen_train):
  discriminator = discriminator_factory()

  dataset = np.concatenate([X_true_train, X_gen_train])
  labels = np.concatenate([
    np.ones(X_true_train.shape[0]),
    np.zeros(X_gen_train.shape[0])
  ])
  weights = np.concatenate([
    np.ones(X_true_train.shape[0]) / X_true_train.shape[0],
    np.ones(X_gen_train.shape[0]) / X_gen_train.shape[0]
  ]) * 2 * np.min([X_gen_train.shape[0], X_true_train.shape[0]])

  discriminator.fit(dataset, labels, sample_weight=weights)

  return discriminator, get_losses(discriminator, X_true_train, X_gen_train)

def validate(discriminator, X_true_val, X_gen_val):
  return get_losses(discriminator, X_true_val, X_gen_val)

def expand(X, N, params, param_names):
  return np.vstack([ X, get_data(n_samples=N, params=params, param_names=param_names) ])

def estimate_wassershtein(pos1, neg1, pos2, neg2):
  s1 = np.concatenate([pos1, neg1], axis=0)
  w1 = np.concatenate([
    np.ones(pos1.shape[0]) / pos1.shape[0],
    np.ones(neg1.shape[0]) / neg1.shape[0]
  ], axis=0)

  s2 = np.concatenate([pos2, neg2], axis=0)
  w2 = np.concatenate([
    np.ones(pos2.shape[0]) / pos2.shape[0],
    np.ones(neg2.shape[0]) / neg2.shape[0]
  ], axis=0)

  return wasserstein_distance(s1, s2, w1, w2)


def jensen_shannon(
  params, param_names, discriminator_factory, X_true_train, X_true_val,
  N_init=128, N_step=64, train_delta=2, plot=False
):

  current_size = N_init
  X_gen_train = get_data(n_samples=N_init, params=params)
  X_gen_val = get_data(n_samples=4 * N_init, params=params)

  losses_train_gen_history = []
  losses_train_true_history = []

  losses_val_gen_history = []
  losses_val_true_history = []

  kl_train_test_history = []
  kl_train_train_history = []

  def draw():
    import matplotlib.pyplot as plt
    from IPython import display

    display.clear_output(wait=True)
    plt.figure(figsize=(9, 6))

    x = np.arange(len(losses_train_true_history))

    m = np.array([
      0.5 * np.mean(x_pos) + 0.5 * np.mean(x_neg)
      for x_pos, x_neg in zip(losses_train_true_history, losses_train_gen_history)
    ])
    plt.plot(x, m, color='blue', label='train loss')

    m = np.array([
      0.5 * np.mean(x_pos) + 0.5 * np.mean(x_neg)
      for x_pos, x_neg in zip(losses_val_true_history, losses_val_gen_history)
    ])
    plt.plot(x, m, color='green', label='test loss')

    plt.plot(x, np.zeros(x.shape[0]), '--', color='black')

    plt.plot(
      x, np.minimum(np.log(2), np.array(kl_train_test_history)),
      color='red', label='W(train, test)'
    )

    plt.plot(
      x, np.minimum(np.log(2), np.array(kl_train_train_history)),
      color='orange', label='W(train, train)'
    )

    plt.legend(loc='upper right')
    plt.show()

  iteration = 0
  while True:
    discriminator, (l_pos, l_neg) = train(discriminator_factory, X_true_train, X_gen_train)
    losses_train_true_history.append(l_pos)
    losses_train_gen_history.append(l_neg)

    l_pos, l_neg = validate(discriminator, X_true_val, X_gen_val)
    losses_val_true_history.append(l_pos)
    losses_val_gen_history.append(l_neg)

    if len(losses_train_gen_history) < train_delta:
      kl_train_train_history.append(0)
    else:
      kl_train_train_history.append(estimate_wassershtein(
        losses_train_true_history[-1], losses_train_gen_history[-1],
        losses_train_true_history[-train_delta], losses_train_gen_history[-train_delta]
      ))

    kl_train_test_history.append(estimate_wassershtein(
      losses_val_true_history[-1], losses_val_gen_history[-1],
      losses_train_true_history[-1], losses_train_gen_history[-1]
    ))

    if plot:
      draw()
      print('Train-train [%.2e], train-test [%.2e]' % (
        kl_train_train_history[-1], kl_train_test_history[-1]
      ))

    if kl_train_test_history[-1] < 1.0e-2 and kl_train_train_history[-1] < 1.0e-2 and iteration >= train_delta:
      break
    else:
      current_size += N_step
      X_gen_train = expand(X_gen_train, N_step, params, param_names)
      X_gen_val = expand(X_gen_val, N_step, params, param_names)

    iteration += 1

  return current_size, 1 - (
          0.5 * np.mean(losses_train_true_history[-1]) + 0.5 * np.mean(losses_train_gen_history[-1])
  ) / np.log(2)