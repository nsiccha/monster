no_samples = 2000
no_chains = 6
sample_kwargs = dict(
    chains=no_chains, seed=3,
    show_progress=True, refresh=1,
    iter_sampling=no_samples//no_chains,
    cache_dir='out',
    trycatch=True
)
advi_kwargs = dict(
    seed=sample_kwargs['seed'],
    cache_dir='out',
    output_samples=no_samples,
    require_converged=False
)

single_sample_kwargs = dict(
    sample_kwargs,
    chains=1
)
