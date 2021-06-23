In the `figs` folder you will find large figures([constrained](https://github.com/nsiccha/monster/tree/master/figs/constrained_monster/incremental.png), [unconstrained](https://github.com/nsiccha/monster/tree/master/figs/unconstrained_monster/incremental.png)) representing a fit of the Monster model. I'll try to keep this readme as brief as possible.

I rebuilt the models ([constrained](https://github.com/nsiccha/monster/tree/master/stan/constrained_monster.stan), [unconstrained](https://github.com/nsiccha/monster/tree/master/stan/unconstrained_monster.stan))
- (mostly) according to the specification in [the original paper](https://stat.columbia.edu/~gelman/research/published/toxicology.pdf),
- from which I also took the [publicly available data](https://github.com/nsiccha/monster/tree/master/public_data.py),
- taking `PPM_per_mg_per_l` from the [MCSim documentation example](https://www.gnu.org/software/mcsim/mcsim.html#perc_002emodel) which differs from the value given in the paper,
- using (not publicly available?) raw measurement data as provided by Charles (not included in the repository),

I added
- an upper truncation for the population (geometric) standard deviation and
- a lower and upper truncation for the personwise parameters.

I kept
- the lower and upper truncation for the population (geometric) means.

I potentially did not implement faithfully
- the "sum-to-one" constraints. Compare samples from the prior with the bounds (grey vertical dashed lines in the population mean plots) at [constrained priors](https://github.com/nsiccha/monster/tree/master/figs/constrained_monster/prior.png) and [unconstrained priors](https://github.com/nsiccha/monster/tree/master/figs/unconstrained_monster/prior.png)

I don't *think* the sum-to-one constraints should impact the fit *very* much, though `Fwp` looks a bit weird.

I did/checked
- [prior predictive checks](https://github.com/nsiccha/monster/tree/master/figs/constrained_monster/prior.png),
- [paper-posterior predictive checks](https://github.com/nsiccha/monster/tree/master/figs/constrained_monster/posterior.png),
- [posterior predictive checks](https://github.com/nsiccha/monster/tree/master/figs/constrained_monster/incremental.png)
- after Bob's suggestion a poor man's [SBC](https://github.com/nsiccha/monster/tree/master/figs/sbc/monster/0.png), for which the fits don't look too bad (red vertical lines are the true parameter values and are sampled from the paper-posteriors)
All predictive checks and the generated data used for SBC used the built-in BDF solver with high precision (rel_tol=1e-12, abs_tol=1e-26).

Warmup+sampling (while conditioning only on the data from the first two participants) took [3+3 minutes](https://github.com/nsiccha/monster/tree/master/figs/constrained_monster/incremental.png) for "my" incremental warmup, which includes an automatic adaptation of the ODE solution precision , while it took [24+6 minutes](https://github.com/nsiccha/monster/tree/master/figs/constrained_monster/regular.png) using Stan's regular warmup, with the ODE solution precision provided by my warmup (not included in runtime). Neff is slightly higher for my warmup, but overall comparable. There were no divergences. Rhat with 6x333 samples is 1.014/1.023. E-BFMI is a bit low, around 0.5.

My warmup consists of repeatedly rerunning the first two adaptation windows with few iterations and pooling draws to estimate the covariance across chains, while
- first simultaneously increasing the number of data points included and
- then simultaneously increasing the precision of the ODE solver until we think lp__ has converged.
Plots of intermediate fits can be found in https://github.com/nsiccha/monster/tree/master/figs/constrained_monster/incremental.

Neither for this partial fit nor for fits including all of the data was I able to recover the marginal posteriors as provided in the original paper (black vertical lines for all figures). Some parameters agreed relatively well, others not at all. Most of my marginal posteriors were considerably wider than the ones from the paper.

I do not include in the repository
- the raw measurements, as they are potentially private,
- the code/framework for my incremental warmup, as it is kind of a mess.
All computations were run on my local machine (6/12 core 2020 Dell-notebook).

I think the model should be correct, or at least as close to correct as necessary. In the best case (for me), everything but the "sum-to-one" constraints are correct, and I can successfully fit the original Monster model roughly within an hour on my local machine. In the worst case, there is a bug somewhere, but even then I can fit a model which qualitatively behaves very similarly to the original Monster model, and after an eventual bugfix I *should* be able to fit the original Monster model.   

Renormalization to satisfy sum-to-X constraints happens at two levels:
- At the population level and
- at the person level.

Meaning, for the blood flows (parameters 2-5) which are constrained to sum to one,
both the *population geometric means* get [renormalized](https://github.com/nsiccha/monster/blob/7c71f0a0d7390ca389459c7d86da3fba10f1da38/stan/monster.stan#L252) such that they sum to one
and the *personwise parameters* get [renormalized](https://github.com/nsiccha/monster/blob/7c71f0a0d7390ca389459c7d86da3fba10f1da38/stan/monster.stan#L260) as well.
The same is done for the relative volumes of the compartments.

To prescribe the same priors exactly as in the [original paper](http://www.stat.columbia.edu/~gelman/bayescomputation/GelmanBoisJIang1996.pdf), it looks like
we would have to adjust the prior means and SDs as described e.g. in http://stat.columbia.edu/~gelman/research/published/moments.pdf

A better approach appears to be to enforce the sum-to-X constraints
by reducing the number of parameters by one for each constraint,
as described by @bob-carpenter (personal communication).
This is implemented in a [tentative model](https://github.com/nsiccha/monster/blob/master/stan/hard_unconstrained_monster.stan) and a sample outputs are [with SD truncation but nu=2](https://github.com/nsiccha/monster/blob/master/figs/hard_unconstrained_monster_sd%3D1/incremental.png) or [without any truncation but nu=4](https://github.com/nsiccha/monster/blob/master/figs/hard_unconstrained_monster_sd%3Dinf_nu%3D4/incremental.png).
