import numpy as np

import pythiamill as pm

fixed_options = [
  ### telling pythia to be quiet.
  'Print:quiet = on',
  'Init:showProcesses = off',
  'Init:showMultipartonInteractions = off',
  'Init:showChangedSettings = off',
  'Init:showChangedParticleData = off',
  'Next:numberCount=0',
  'Next:numberShowInfo=0',
  'Next:numberShowEvent=0',
  'Stat:showProcessLevel=off',
  'Stat:showErrors=off',

  ### seeting default parameters to Monash values
  "Tune:ee = 7",
  "Beams:idA = 11",
  "Beams:idB = -11",
  "Beams:eCM = 91.2",
  "WeakSingleBoson:ffbar2gmZ = on",
  "23:onMode = off",
  "23:onIfMatch = 1 -1",
  "23:onIfMatch = 2 -2",
  "23:onIfMatch = 3 -3",
  "23:onIfMatch = 4 -4",
  "23:onIfMatch = 5 -5",
]

param_names20 = [
  "TimeShower:alphaSvalue",
  "TimeShower:pTmin",
  "TimeShower:pTminChgQ",

  "StringPT:sigma",
  "StringZ:bLund",
  "StringZ:aExtraSQuark",
  "StringZ:aExtraDiquark",
  "StringZ:rFactC",
  "StringZ:rFactB",

  "StringFlav:probStoUD",
  "StringFlav:probQQtoQ",
  "StringFlav:probSQtoQQ",
  "StringFlav:probQQ1toQQ0",
  "StringFlav:mesonUDvector",
  "StringFlav:mesonSvector",
  "StringFlav:mesonCvector",
  "StringFlav:mesonBvector",
  "StringFlav:etaSup",
  "StringFlav:etaPrimeSup",
  "StringFlav:decupletSup"
]

space20 = np.array([
  (0.06, 0.25),
  (0.1, 2.0),
  (0.1, 2.0),

  (0.2, 1.0),
  (0.0, 1.0),
  (0.0, 2.0),
  (0.0, 2.0),
  (0.0, 2.0),
  (0.0, 2.0),

  (0.0, 1.0),
  (0.0, 1.0),
  (0.0, 1.0),
  (0.0, 1.0),
  (0.0, 1.0),
  (0.0, 1.0),
  (0.0, 1.0),
  (0.0, 3.0),
  (0.0, 3.0),
  (0.0, 3.0),
  (0.0, 3.0)
])

monash20 = np.array([
  0.1365,
  0.5,
  0.5,

  0.98,
  0.335,
  0,
  0.97,
  1.32,
  0.885,

  0.217,
  0.081,
  0.915,
  0.0275,
  0.6,
  0.12,
  1,
  0.5,
  0.55,
  0.88,
  2.2
])

param_names1 = param_names20[:3]
space1 = space20[:3]
monash1 = monash20[:3]

param_names2 = param_names20[3:9]
space2 = space20[3:9]
monash2 = monash20[3:9]

param_names3 = param_names20[9:]
space3 = space20[9:]
monash3 = monash20[9:]

def get_mill(params=monash20, param_names=param_names20, n_workers=8, batch_size=128, seed=123):
  options = fixed_options + ["%s=%lf" % (k, v) for k, v in zip(param_names, params)]

  ### TuneMC detector provides the same features used in TuneMC paper
  detector = pm.utils.TuneMCDetector()

  mill = pm.PythiaMill(
    detector, options, batch_size=batch_size,
    cache_size=n_workers * 2, n_workers=n_workers, seed=seed
  )

  return mill

def sample_mill(mill, n_samples=2**16):
  batches = []
  while sum([ b.shape[0] for b in batches ]) < n_samples:
    batches.append(mill.sample())

  return np.vstack(batches)[:n_samples]


def get_data(n_samples=2**16, params=monash20, param_names=param_names20, n_workers=8, batch_size=128, seed=123):
  mill = get_mill(params, param_names, n_workers, batch_size, seed)
  result = sample_mill(mill, n_samples)
  mill.terminate()
  return result