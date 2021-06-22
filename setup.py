no_samples = 2000
no_chains = 6
sample_kwargs = dict(
    chains=no_chains, seed=3,
    show_progress=True, refresh=1,
    iter_sampling=no_samples//no_chains,
    cache_dir='out',
    trycatch=True
)
