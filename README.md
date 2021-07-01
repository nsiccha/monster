# The Monster model revisited using [Stan](https://mc-stan.org/)

The so called Monster model is a
[hierarchical](https://en.wikipedia.org/wiki/Bayesian_hierarchical_modeling#Hierarchical_models)
[physiologically based pharmacokinetic](https://en.wikipedia.org/wiki/Physiologically_based_pharmacokinetic_modelling) model describing
the evolution and measurement of a carcinogenic in the human body, named after
the first author (A.C. Monster) of the
[paper](https://link.springer.com/content/pdf/10.1007/BF00377784.pdf)
which inspired the "original" Bayesian model developed by
[Gelman et al.](http://www.stat.columbia.edu/~gelman/bayescomputation/GelmanBoisJIang1996.pdf).
This model has originally been fit using [GNU MCSim](https://www.gnu.org/software/mcsim/)
implementing "a variant of the [Gibbs sampler](https://en.wikipedia.org/wiki/Gibbs_sampling)" (Gelman et al.).
While the authors report imperfect convergence diagnostics
("Sqrt(Rhat) values were reduced to all lie below 1.2", section 3.1, ibid.),
they appear confident in their results and provide population and personwise
posterior means and standard deviations (Table 1, ibid.).

However, attempts to fit the Monster model using Stan, which implements
among other things [Hamiltonian Monte Carlo](https://en.wikipedia.org/wiki/Hamiltonian_Monte_Carlo)
with the associated [No U-Turn Sampler](https://jmlr.org/papers/volume15/hoffman14a/hoffman14a.pdf),
have so far been unsuccessful. Here we will very briefly summarize the steps needed
to fit the Monster model efficiently and reliably using Stan and report differences
between the fit obtained using Stan and the fit reported in the original paper.

## Methods

We believe we
[rebuilt the original model](stan/flexible_monster.stan)
almost 100% as in the original paper (there were some inconsistencies across sources).
For this we combined
* the [original paper](http://www.stat.columbia.edu/~gelman/bayescomputation/GelmanBoisJIang1996.pdf)
which is somewhat sparse on details,
* a [revamped version(?)](https://stat.columbia.edu/~gelman/research/published/toxicology.pdf)
with more authors and more explicit implementation details,
* an [MCSim documentation example](https://www.gnu.org/software/mcsim/mcsim.html#perc_002emodel)
which provides a different `PPM_per_mg_per_l` than the two papers, but one that makes
the prior [predictive](https://docs.pymc.io/notebooks/posterior_predictive.html) [checks](https://avehtari.github.io/masterclass/slides_ppc.pdf) look much more [reasonable](figs/nu=8/paper_posterior.png),
* raw measurement data provided by Charles.

For fitting the model we employed our own custom warm-up, but verified with the default warm-up.

### The modified Monster model

Changes to the original model which **were not necessary** for "convergence" but "feel" better
in the context of HMC include
* removing the hard bounds on the population means and
* removing the overparametrizations due to softly enforcing various sum-to-X constraints (thanks Bob!).

The only change to the original model which **was necessary** to fit the model without divergences
was
* to tighten the priors on the population geometric standard deviations (GSD).

We believe that due to an inherent [non-identifiability](https://en.wikipedia.org/wiki/Identifiability)
of some pairs of parameters
(the behavior of the underlying ordinary differential equation depends *only*
on pairwise *products* of these parameters)
we get very little information on the GSDs of these parameters.
For the priors proposed in the paper ([scaled_inv_chi_square](https://mc-stan.org/docs/2_22/functions-reference/scaled-inverse-chi-square-distribution.html) with nu=2)
we cannot even sample from the prior without issues, and with little to no information
on some GSDs from the data, we expect this problems to carry over to the posterior.
In fact, even for many of the parameters which *do not* suffer from this non-identifiability,
we do not obtain tight bounds on personwise parameter values or on population means and GSDs.  

For the model formulation a ~~centered~~ ~~(no, non-centered!)~~
"unconstrained-and-on-the-unit-scale" parametrization was chosen for the population
means and personwise parameters (meaning a manual [non-centered](https://mc-stan.org/docs/2_18/stan-users-guide/reparameterization-section.html) parametrization on the unit scale, i.e. mu_raw ~ std_normal() and param_raw ~ std_normal()).

### Incremental and adaptive warm-up

While the Monster model does not appear to suffer from spurious
[multimodality](https://www.yulingyao.com/blog/2019/stacking/) such as
other ordinary differential equation (ODE) models
(e.g. the [planetary motion problem](https://mc-stan.org/users/documentation/case-studies/planetary_motion/planetary_motion.html)), it still suffers from the regular difficulties
intrinsic to ODE models. There are several tuning parameters, which
* if chosen too conservatively (high precision) slows down computation considerably,
* if chosen too aggressively (low precision) may frustrate the sampler by introducing
"spurious" divergences.

While one wants to avoid these spurious divergences at all cost, one also does not want to
have to wait hours to days for the results, if not necessary!
On top of efficiency and divergence concerns,
choosing an ODE precision that is too low may unbeknownst to the user introduce bias into the estimate!

*However*, a priori we do not know which ODE configuration is just right, yielding
unbiased estimates and not derailing the HMC sampler, while also efficiently yielding results.
In principle we could draw a lot of samples from the prior, simulate the ODEs
with incrementally increasing precision, select some domain specific convergence
threshold after which we deem the solution(s) accurate enough and select an ODE
configuration just past this point (as done e.g. [here](https://users.aalto.fi/~timonej3/case_study_num.html#32_Generating_test_data)).
This however is potentially inefficient or inaccurate, simply because the precision
requirements across the prior may be completely different than across the posterior.
For the [planetary motion problem](https://mc-stan.org/users/documentation/case-studies/planetary_motion/planetary_motion.html),
accurately simulating the "typical set" of the *prior* would require a much *higher precision*
than is necessary for the specific posterior due to the data discussed,
while for models such as the Monster model certain *posteriors* could require a
much *higher precision* than is necessary to simulate the "typical set" of the prior.

Luckily we have an answer to automatically, adaptively and efficiently select
the "best" ODE configuration. It is essentially [this workflow](https://users.aalto.fi/~timonej3/case_study_num.html#2_Workflow) but embedded into an incremental and adaptive warm-up
which allows the reconfiguration (of e.g. ODE solver configurations) **during warm-up**.

Bob has (rightfully) pointed out similarities of my warm-up procedure to
[Sequential Monte Carlo methods (SMC)](https://en.wikipedia.org/wiki/Particle_filter).
In fact, just as SMC, our *incremental* warm-up procedures relies on the user identifying a
"good" sequence of data updates, preferrably starting at "no data" (only prior information)
and ending at "all of the data" (full posterior). With these data updates provided,
the incremental warm-up procedure proceeds as follows:

#### Incremental and parallelizable warm-up

For each data update
* For each chain, perform the first two warm-up phases with a *single* metric adaptation window.
* Compute the global covariance from the pooled metric adaptation draws across chains.
* Use the last draws from each chain as starting point for the next dataset, the (pooled) covariance as the metric, and (currently) the across-chain-mean of the very last timestep as the new timestep.
* Add whatever reconfiguration you deem necessary to the next data-update(s). This is where the automatic ODE reconfiguration can be plugged in.

Before sampling, Stan's last warm-up phase may be necessary (currently I use a replacement).

A "good" sequence of data updates allows chains to use starting points and metrics
from previous data updates to **efficiently** explore the current intermediate posterior.
To minimize the total computational cost, data updates which incrementally *double*
the cost of each leapfrog iteration appear to be ideal. This generally allows us to *skip
the costly early phases of Stan's default warm-up, where the metric is not well-adapted and
the average treedepth is high*. In the best case (as in the Monster model), average
treedepths are only high during early stages of our warm-up, which due to the
exponentially increasing computational cost per leapfrog iteration contributes very
little in terms of total computational costs. *Pooling of draws across chains to
estimate the covariance* allows us to **parallelize the warm-up**, with the current
parallelization bottleneck being the repeated first warm-up phase. *However*,
the first warm-up phase might be able to be eliminated completely or shortened by
using importance sampling ~~(not tested yet)~~. Due to the pooling across chains we
currently get away with 100 final metric adaptation iterations per chain
(using 6 parallel chains). As a side effect of doing things incrementally,
we also tend to avoid spurious modes.

##### Short addendum: importance resampling to skip repeating phase I

With the posteriors appearing unimodal, at first glance it appears to be safe
to use importance resampling to get new initial points for the chains which
are closer to or in the typical set of the next (intermediate) posterior,
thereby allowing us to shorten or skip the repeated phase I. This *appears* to improve the
scalability of the algorithm somewhat. **This should not be done if the posterior
is multimodal, as we may lose a mode this way**, at least if we first pool draws across
chains and then resample. Resampling "within chains" should still be safe, even
for multimodal posteriors.

#### Adaptive warm-up

Adaptively tuning the ODE solver configurations is just one special case of reconfiguration
that is possible using the above warm-up procedure.
In our [implementation of the Monster model](stan/flexible_monster.stan)
we can solve the personwise ODEs either using a custom ODE solver relying
on a [Strang splitting](https://en.wikipedia.org/wiki/Strang_splitting)
or using the [built-in](https://mc-stan.org/docs/2_24/functions-reference/functions-ode-solver.html)
adaptive numerical solver provided by [CVODES](https://computing.llnl.gov/projects/sundials/cvodes)
using the [backward differentiation formula (BDF)](https://en.wikipedia.org/wiki/Backward_differentiation_formula). We can switch between the two options and tune them using a `data`
argument.

While the built-in ODE solver appears to more efficiently provide high precision solutions,
it does not appear to work *at all* if the precision is too low. Our custom ODE
solver is roughly equivalent in terms of computational cost for moderate precision solutions,
while allowing us to go to arbitrarily low precision without ever derailing the
HMC sampler by introducing "spurious" divergences but still yielding *qualitatively*
correct solutions. The custom ODE solver is tuneable
via a `data` argument determining the number of steps performed, with the
computational cost scaling linearly with this argument. For the built-in ODE solver
the configuration-cost relationship is unclear a priori, except that it is monotonic.

Currently, the adaptation starts with a very cheap, very low precision configuration and proceeds as follows:
* If we are at the final data update (i.e. don't adapt the configuration beforehand),
recompute the (log) posterior density for the N draws from the current metric
adaptation window, *but with a higher precision*.
* Compute importance weights and Neff.
* If Neff/N < threshold rerun warm-up phases I+II reinitialized as discussed above and repeat,
else start sampling with the current metric and initial points (and a custom timestep adaptation).

Currently "higher precision" means using double the number of ODE steps and
equivalently using double the computational resources per leapfrog step.
The current threshold is set at a very conservative 99% to ensure convergence.
*However*, the threshold could potentially be relaxed if we want to importance sample after
approximate HMC sampling (as done e.g. [here](https://users.aalto.fi/~timonej3/case_study_num.html#2_Workflow)). Due to the computational cost again
increasing exponentially, early adaptation windows tend to contribute very little
to the final computational cost.

# Results

We can efficiently fit the full Monster model with all diagnostics looking good for nu>=3.
For nu=2 the divergences do not appear to be removable by lowering the step size.
Our custom warm-up procedure not only automatically provides the "ideal" ODE configuration,
it also is considerably faster and computationally more efficient
than Stan's regular warm-up.

Below we discuss a single case in detail, but all other cases are similar.

## The case nu=8

The parallel version of my warm-up (employing 6 chains) outperforms
the regular warm-up (also employing 6 parallel chains) in terms of wall time
by a factor of more than 12 with "better" diagnostics and higher Neff.
However, as Bob has pointed out,
neither my warm-up nor the regular warm-up can run at peak computational efficiency
with 6 chains in parallel on my local hardware
(a 2020 Dell Precision 5500 running an Intel(R) Core(TM) i7-10750H CPU @ 2.60GHz with 6 cores).
A fairer comparison (neglecting the parallelizability of my warm-up) thus runs just a
single chain (for both warm-up procedures),
which can then exploit computational and memory bandwidth resources
at its fullest. For this setup we get for warm-up+sampling wall times

* 45m+19m (my serial warm-up),
* 3h+54m (Stan's warm-up)

and for "effective" total number of final leapfrog steps per sample

* 204 (my serial warm-up)
* 733 (Stan's warm-up)

with better diagnostics and Neff for my warm-up.

For my warm-up we have to estimate how many "cheaper" leapfrog steps are equivalent
to one "final" leapfrog step for my warm-up. The cost scales both with the amount
of data included and with the number of ODE steps. The estimation used appears to
be accurate enough, with wall time and effective total number of final leapfrog steps
per sample correlating nicely.

For sake of completeness we also include the same metrics for the parallel version
of my warm-up with the same number of final samples per chain:

* warm-up+sampling wall times: 24m+33m
* effective total number of final leapfrog steps per sample: 100

During sampling, the mean number of leapfrog steps were

* 56 (my serial warm-up),
* 166 (Stan's warm-up),
* 57 (my parallel warm-up).

Thus, 1-19/33 = 42%, appears to be a good estimate of the loss of computational
efficiency for the parallel run due to the limitations of my local hardware
and we could optimistically expect a further reduction of warm-up wall time by 42%
on a machine/cluster on which all chains could run unperturbed.
This would currently give us a speedup of 45/(24*58%) = 323% with 6 chains,
which is not ideal but appears acceptable. Other parallelization overhead such
as communication should be negligible, as the computations are very data efficient
(many FLOPs per Mb of data/communication) for the later, more expensive stages of warm-up.

### Excessively large figures with too much information

For each setting and method large figures can be found under [figs/nu=x](figs/)
visualizing the different fits and including diagnostics in the lower left corner.
[figs/nu=x/method.png](figs/nu=8/) plots the prior fit and the fit obtained using
the respective method, while [figs/nu=x/comparison.png](figs/nu=8/comparison.png)
includes all fits.

In the figures you can see, with different colors representing different fits,

* in the first row an estimate of the cumulative work performed,
* in the second row a trace plot of the number of leapfrog steps per iteration,
* in the third row a trace plot of `lp__`,
* in the fourth row histograms of the population means (left) and
mean predicted vs observed states per experiment and measurement type (right),
* in the fifth row histograms of the population GSDs (left) and
of the two noise parameters and `lp__` and `energy__` and  
* starting from the sixth row with one person per row
histograms of the personwise parameter values (left)
and predictive checks (right).

For the *population means*, vertical lines indicate
prior (grey) and
paper posterior (black)
means (solid) and
population means +/- 3 population SDs (dashed).
For the *population GSDs* vertical lines indicate
prior (grey) and
paper posterior (black)
estimates of the population GSDs.
For the *personwise parameter values*, grey lines indicate the same thing
as for the population means, while black lines indicate
*personwise paper posterior means +/- 3 personwise SDs*.


For ease of access we link the comparison figure for the case nu=8:
![nu=8](figs/nu=8/comparison.png?raw=true "nu=8")

### Why you should/should not trust my results

Reasons not to trust my results:

* I know nothing about pharmacology,
* I know very little about Bayesian inference,
* there may still be a bug in my model, e.g. I might accidentally solve the wrong ODE,
* this is my first hierarchical model and by extension my first hierarchical ODE model,
* my warm-up procedure has **not** been extensively tested,
* my results do not agree with the results reported in the paper:
  * **all population-variance-parameters and parameter variances are much higher than reported**,
  * VPR does not move much (and in the wrong direction),
  * Fwp, Fpp, VMI and KMI do not move enough (but in the right direction),
  * Ff, Fl do almost not move at all,
  * Vwp, Vpp, Vl, Pwp, Ppp, Pf and Pl seem to move to the "right" location (but retain a high variance),
  * Pba very confidently appears to "overshoot",
* not all diagnostics are always perfect (Some Rhats are > 1.01, even if barely so),
* relatedly, I reimplemented the (split) Rhat and E-BFMI computation myself,
meaning there might potentially be a bug somewhere.

Possible reasons to trust my results:

* prior predictive checks and paper-posterior predictive checks [look good](figs/nu=8/paper_posterior.png),
* my-posterior predictive checks [look great](figs/nu=8/parallel_incremental.png),
* posterior predictive checks **taking the paper-posterior as prior and then fitting**
[look similarly great, if maybe slightly worse,](figs/nu=8/posterior_resampling_parallel_incremental.png)
and appear to recover the reported personwise parameters reasonably well,
* the fits obtained using my warm-up look
[indistinguishable in the eye-norm](figs/nu=8/incremental_vs_regular.png)
from fits obtained using Stan's regular warm-up,
* for the predictive checks, Stan's built in ODE solver with a high-precision configuration
has been used, while for fitting I used my custom ODE solver and both appear to agree with one another,
* apparently, the variant of the Gibbs sampler used in the paper is prone to getting stuck or
not properly exploring high-dimensional posteriors with correlations,
* all diagnostics look fine for my fits, while the diagnostics reported in the
paper are less than ideal and less exhaustive (no concept of divergences, no E-BFMI),
* my reimplementation of the diagnostics so far have always agreed closely with the
ones computed by `CmdStan`.

### Data and model availability

The Stan model can be found at [stan/flexible_monster.stan](stan/flexible_monster.stan).
The final fits can be found at [cfg/nu=x/method.csv](cfg/nu=8/) and
the final data files and configurations can be found at [cfg/nu=x/method_*.json](cfg/nu=8/),

* where x is 2, 3, 4 or 8 (once everything has finished running),
* method is either of `serial_regular`, `serial_incremental`, `parallel_incremental` or `resampling_parallel_incremental` and
* `*` is either of `data`, `init_*`, `kwargs` or `metric`.

In the subfolders [cfg/nu=x/method](cfg/nu=8/) the same files can be found for all
intermediate data updates.

### Code availability

Everything but the secret sauce is included in this repository.
The secret sauce is a hot and continuously shifting mess, but is available upon request.
