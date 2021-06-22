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

model_path = f'stan/variance_monster.stan'

def estimate_work(data):
    return 1

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

no_persons = 6
no_plotted_persons = min(6, no_persons)
divergence_goal = 1#0


std_trunc = 1
pop_trunc = 0#np.inf
person_trunc = 10
noise_scale = .1

param_labels = public_data.param_labels
no_latent_params = public_data.no_latent_params

base_data = dict(
    no_persons=no_persons,
    no_latent_params=no_latent_params,
    noise_scale=noise_scale,
    observed_states=np.zeros((no_persons, no_latent_params)),
    no_experiments=0
)

prior_data = dict(
    base_data,
    likelihood=0,
    **public_data.get_base_data(
        public_data.prior_population_parameters, std_trunc, pop_trunc, person_trunc
    ),
)
posterior_data = dict(
    base_data,
    likelihood=0,
    **public_data.get_base_data(
        public_data.posterior_population_parameters, std_trunc, pop_trunc, person_trunc
    ),
)
prior_fit = model.sample(prior_data, **sample_kwargs)
posterior_fit = model.sample(posterior_data, **sample_kwargs)
prior_fig = plotting.plot_fit(
    prior_fit, prefix='prior', no_plotted_persons=no_plotted_persons,
    path=f'figs/sbc/variance/prior.png',
)
plotting.plot_fit(
    posterior_fit, fig=prior_fig,
    prefix='posterior', no_plotted_persons=no_plotted_persons,
    path=f'figs/sbc/variance/posterior.png',
)
for idx in range(10):
    fit_data = dict(
        posterior_fit.sbc_data(idx),
        **public_data.get_base_data(
            public_data.prior_population_parameters, std_trunc, pop_trunc, person_trunc
        ),
    )
    regular_fit = model.sample(fit_data, **sample_kwargs)
    best_population_parameters = np.array([
        regular_fit.true_series.filter(regex=f'^population_eM'),
        regular_fit.true_series.filter(regex=f'^population_eS')**np.sqrt(1/no_persons),
        regular_fit.true_series.filter(regex=f'^population_eS'),
        regular_fit.true_series.filter(regex=f'^population_eS')*0+no_persons+2,
        public_data.prior_population_parameters[:,-1]
    ]).T
    best_data = dict(
        prior_data,
        **public_data.get_base_data(
            best_population_parameters, std_trunc, pop_trunc, person_trunc
        )
    )
    best_fit = model.sample(best_data, **sample_kwargs)
    regular_fig = plotting.plot_fit(
        prior_fit,
        prefix='prior', no_plotted_persons=no_plotted_persons
    )
    plotting.plot_fit(
        regular_fit, fig=regular_fig,
        prefix='regular warmup', no_plotted_persons=no_plotted_persons
    )
    plotting.plot_fit(
        best_fit, fig=regular_fig, path=f'figs/sbc/variance/{idx}.png',
        prefix='best prior fit', no_plotted_persons=no_plotted_persons
    )
