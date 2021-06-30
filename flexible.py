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

model_path = f'stan/flexible_monster.stan'

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
model_base = f'figs/{model.name}'
fig = base = title = None

def callback(i, fit, **kwargs):
    global fig
    global base
    global title
    tprint(fit.short_diagnosis)
    fit_idx = len(fit.fit_sequence)
    if fit_idx == 1: fig = None
    subdir = ('serial' if fit.no_chains == 1 else 'parallel') + '_incremental'
    fit.dump(f'cfg/{title}/{subdir}/{fit_idx:03d}')
    # if std_nu < 4: return
    fig = plotting.plot_fit(
        fit, path=f'figs/{title}/{subdir}/{fit_idx:03d}.png', fig=fig,
        overlay=fit_idx
    )

measured_params = public_data.measured_params
exposures = public_data.exposures
raw_measurements = private_data.raw_measurements
weights = private_data.weights

no_conditioned_persons = 6
weights[no_conditioned_persons:] = 0

no_persons, no_measured_params = measured_params.shape
no_persons, no_experiments, no_measurements, _ = raw_measurements.shape

min_no_sub_steps = 16
max_no_sub_steps = 128
no_sub_steps_progression = list(geometric_progression(
    min_no_sub_steps, max_no_sub_steps
))

refinement_data = [
    dict(no_sub_steps=no_sub_steps)
    for no_sub_steps in no_sub_steps_progression[1:]
]

fneff_goal = .99#.99
divergence_goal = .01#0
no_fit_sub_steps = no_sub_steps_progression[0]
no_sim_sub_steps = -12


std_trunc = np.inf
# std_nu = 8
enforce_constraints = True
pop_trunc = np.inf
person_trunc = np.inf
noise_scale = .1

param_labels = public_data.param_labels
no_latent_params = public_data.no_latent_params

for std_nu, enforce_constraints in itertools.product([8,4,3,2], [1]):
    # title = f'persons={no_conditioned_persons}_sd={std_trunc}_nu={std_nu}_hard={enforce_constraints}'
    title = f'nu={std_nu}'
    tprint.extra = [title]
    # base = f'{model_base}/{title}'
    fig_base = f'figs/{title}'
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
        enforce_constraints=enforce_constraints,
        include_jacobian=1,
    )

    prior_data = dict(
        base_data,
        likelihood=0,
        **public_data.get_base_data(
            public_data.prior_population_parameters, std_trunc, pop_trunc, person_trunc,
            nu=std_nu
        ),
    )

    paper_posterior_data = dict(
        base_data,
        likelihood=0,
        **public_data.get_base_data(
            public_data.posterior_population_parameters, std_trunc, pop_trunc, person_trunc,
            nu=std_nu
        ),
    )
    reg_prior_data = dict(
        prior_data,
        population_eS_nu=np.where(
            prior_data['population_eS_nu'] > 2,
            prior_data['population_eS_nu'],
            3
        )
    )
    prior_fit = model.sample(reg_prior_data, **sample_kwargs)
    prior_fit.label = 'prior'
    incremental_fig = plotting.plot_fit(
        prior_fit, path=f'{fig_base}/prior.png',
        force=True,
        title=title
    )
    if std_nu > 2:
        paper_posterior_fit = model.sample(paper_posterior_data, **sample_kwargs)
        paper_posterior_fit.label = 'paper posterior'
        plotting.plot_fit(
            [prior_fit, paper_posterior_fit], path=f'{fig_base}/paper_posterior.png',
            force=True,
            title=title
        )

    fit_data = dict(
        prior_data,
        likelihood=1
    )
    incremental_data = [reg_prior_data] + [
        dict(
            fit_data,
            no_measurements=i,
            experiments=raw_measurements[:,:,:i],
            weights=weights[:,:,:i]
        )
        for i in geometric_progression(2,no_measurements)
    ]

    tprint('Starting adaptive incremental fit')
    incremental_fit = model.isample(
        incremental_data,
        warmup=dict(
            callback=callback,
            refine=(fneff_goal, refinement_data),
        ),
        **sample_kwargs
    )#.eliminate_divergences(divergence_goal, callback, **sample_kwargs)
    incremental_fit.label = 'parallel incremental warmup'
    # if fig is not None:
    #     plt.close(fig)
    plotting.plot_fit(
        incremental_fit, fig=incremental_fig, path=f'{fig_base}/parallel_incremental.png',
        # prefix='incremental warmup',
        force=True
    )

    # tprint('Starting ADVI')
    # advi = advi_fit = model.variational(
    #     dict(incremental_fit.sample_data, include_jacobian=0),
    #     **advi_kwargs
    # )
    # advi.lw_unconstrained_params = advi.lw_params.copy()
    # advi.lw_unconstrained_params['noise.1'] = np.log(advi.lw_unconstrained_params['noise.1'])
    # advi.lw_unconstrained_params['noise.2'] = np.log(advi.lw_unconstrained_params['noise.2'])
    # plotting.plot_fit(
    #     advi_fit, fig=incremental_fig, path=f'{base}/advi.png',
    #     prefix='ADVI',
    #     force=True,
    # )

    for n in [1]:#,4,8]:
        tprint(f'Starting {n}-adaptive incremental fit')
        single_incremental_fit = model.isample(
            # incremental_data,
            incremental_fit.sequence_data[:-1],
            warmup=dict(
                callback=callback,
                # refine=(fneff_goal, refinement_data),
            ),
            **dict(sample_kwargs, chains=n)
        )#.eliminate_divergences(divergence_goal, callback, **sample_kwargs)
        single_incremental_fit.label = 'serial incremental warmup'
        plotting.plot_fit(
            single_incremental_fit, fig=incremental_fig,
            # force=True
        )

        lk = {
            f'serial regular warmup': dict(),
            # f'{n}-HMC(init,metric)+diagonal': dict(
            #     inits=incremental_fit.last_draws.iloc[:n],
            #     metric=np.diag(incremental_fit.global_metric)
            # ),
            # 'ADVI(init,metric)+diagonal': dict(
            #     inits=advi_fit.last_draws.iloc[:1],
            #     metric=np.diag(advi_fit.global_metric)
            # ),
            # 'HMC(init)+diagonal': dict(
            #     inits=incremental_fit.last_draws.iloc[:1],
            # ),
            # 'ADVI(init)+diagonal': dict(
            #     inits=advi_fit.last_draws.iloc[:1],
            # ),
        }
        for label, kwargs in lk.items():
            tprint(f'Starting {label} fit')
            pdir_fit = model.sample(
                incremental_fit.sample_data,
                **dict(sample_kwargs, chains=n, **kwargs)
            )
            pdir_fit.label = label
            plotting.plot_fit(
                pdir_fit, fig=incremental_fig, path=f'{fig_base}/comparison.png',
                force=True,
            )

            plotting.plot_fit(
                [prior_fit, pdir_fit], path=f'{fig_base}/serial_regular.png',
                force=True,
                title=title
            )

        plotting.plot_fit(
            [prior_fit, single_incremental_fit], path=f'{fig_base}/serial_incremental.png',
            force=True,
            title=title
        )
    plt.close('all')
    #
    # tprint('Starting perfectly initialized dense regular fit')
    # pder_fit = model.sample(
    #     incremental_fit.sample_data,
    #     inits=incremental_fit.last_draws.iloc[:1],
    #     metric=incremental_fit.global_metric,
    #     **single_sample_kwargs
    # )
    # plotting.plot_fit(
    #     pdir_fit, fig=incremental_fig, path=f'{base}/pder.png',
    #     prefix='Perfectly-initialized-dense-regular',
    #     force=True,
    # )
    tprint('Done.')
