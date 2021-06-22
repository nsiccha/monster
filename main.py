import itertools
import numpy as np
import matplotlib.pyplot as plt
from excmdstanpy import *
import logging
import public_data
import private_data
import plotting
from setup import *

logging.basicConfig(level=logging.WARNING)
cmdstan_paths = [
    '/home/niko/cmdstan'
]
cmdstanpy.set_cmdstan_path(cmdstan_paths[-1])

float_formatter = "{:.4g}".format
np.set_printoptions(formatter={'float_kind':float_formatter})

model_path = f'stan/monster.stan'

def estimate_work(data):
    return 1+data['likelihood']*np.size(data['experiments'])*data['no_sub_steps']

model = StanModel(
    stan_file=model_path,
    params=[
        'unit_log_population_eM',
        'unit_log_population_eS',
        'unit_log_person_params',
        'noise'
    ],
    estimate_work=estimate_work
)

measured_params = public_data.measured_params
exposures = public_data.exposures
raw_measurements = private_data.raw_measurements
weights = private_data.weights

no_conditioned_persons = 2
weights[no_conditioned_persons:] = 0

no_persons, no_measured_params = measured_params.shape
no_persons, no_experiments, no_measurements, _ = raw_measurements.shape

min_no_sub_steps = 1
max_no_sub_steps = 128
no_sub_steps_progression = list(geometric_progression(
    min_no_sub_steps, max_no_sub_steps
))

refinement_data = [
    dict(no_sub_steps=no_sub_steps)
    for no_sub_steps in no_sub_steps_progression[1:]
]

fneff_goal = .99#.99
divergence_goal = 1#0
no_fit_sub_steps = no_sub_steps_progression[0]
no_sim_sub_steps = -12


std_trunc = 1
pop_trunc = 0
person_trunc = 10
noise_scale = .1


param_labels = public_data.param_labels
no_latent_params = public_data.no_latent_params

base_data = dict(
    no_persons=no_persons,
    no_measured_params=no_measured_params,
    measured_params=measured_params,
    no_experiments=no_experiments,
    exposures=exposures,
    no_measurements=no_measurements,
    experiments=raw_measurements,
    weights=weights,
    no_latent_params=no_latent_params,
    noise_scale=noise_scale,
    no_sub_steps=no_fit_sub_steps,
    no_sim_sub_steps=no_sim_sub_steps,
)

prior_data = dict(
    base_data,
    likelihood=0,
    **public_data.get_base_data(
        public_data.prior_population_parameters, std_trunc, pop_trunc, person_trunc
    ),
)
prior_fit = model.sample(prior_data, **sample_kwargs)

fit_data = dict(
    prior_data,
    likelihood=1
)
incremental_data = [prior_data] + [
    dict(
        fit_data,
        no_measurements=i,
        experiments=raw_measurements[:,:,:i],
        weights=weights[:,:,:i]
    )
    for i in geometric_progression(2,no_measurements)
]

fig = None
def callback(i, fit, **kwargs):
    global fig
    tprint(fit.short_diagnosis)
    fit_idx = len(fit.fit_sequence)
    fig = plotting.plot_fit(
        fit, path=f'figs/incremental/{fit_idx:03d}.png', fig=fig,
        overlay=fit_idx
    )

incremental_fit = model.isample(
    incremental_data,
    warmup=dict(
        callback=callback,
        refine=(fneff_goal, refinement_data),
    ),
    **sample_kwargs
).eliminate_divergences(divergence_goal, callback)
regular_fit = model.sample(
    incremental_fit.sample_data, **sample_kwargs
)
prior_fit = model.sample(prior_data, **sample_kwargs)
incremental_fig = plotting.plot_fit(
    prior_fit, prefix='prior', path='figs/prior.png'
)
plotting.plot_fit(
    incremental_fit, fig=incremental_fig, path='figs/incremental.png',
    prefix='incremental warmup'
)
regular_fig = plotting.plot_fit(prior_fit, prefix='prior')
plotting.plot_fit(
    regular_fit, fig=regular_fig, path='figs/regular.png',
    prefix='regular warmup'
)
