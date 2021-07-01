import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import public_data
from excmdstanpy import *
import shutil


possible_priors = [
    public_data.posterior_population_parameters,
    public_data.prior_population_parameters,
]

def plot_fit(fit, **kwargs):
    if isinstance(fit, list):
        fig = None
        for ifit in fit[:-1]:
            fig = plot_fit(ifit, **dict(kwargs, fig=fig, path=None))
        return plot_fit(fit[-1], **dict(kwargs, fig=fig))
    force = kwargs.pop('force', False)
    # return
    no_persons = fit.sample_data['no_persons']
    no_experiments = fit.sample_data['no_experiments']
    no_latent_params = fit.sample_data['no_latent_params']
    true_params = fit.sample_data.get('true_params', None)
    fig = kwargs.pop('fig', None)
    no_fits = 0 if fig is None else len(fig.fits)
    init = kwargs.pop('init', fig is None)
    overlay = kwargs.pop('overlay', False)
    overlay_alpha = kwargs.pop('overlay_alpha', .25)
    no_lines = kwargs.pop('no_lines', 6)
    colors = kwargs.pop('colors', sns.color_palette('husl', no_lines))
    color = kwargs['color'] = kwargs.get('color', colors[no_fits % no_lines])
    prefix = kwargs.pop('prefix', getattr(fit, 'label', None))
    path = kwargs.pop('path', None)
    title = kwargs.pop('title', None)

    no_mrows = 3
    no_rows = no_mrows+2+kwargs.pop('no_plotted_persons', no_persons)
    no_cols = (2*no_experiments+no_latent_params)
    no_lines = 6
    text_height = no_lines/3
    figsize = (4*no_cols, text_height+4*no_rows)
    top = (figsize[1] - 1)/figsize[1]
    bot = text_height / figsize[1]
    dy = bot/(no_lines+1)
    if init:
        fig, all_axes = plt.subplots(
            no_rows, no_cols,
            figsize=figsize, squeeze=False,
            sharex=False, sharey=False
        )
        fig.fits = []
        fig.top_axes = []
        fit.param_axes = None
        plt.tight_layout(pad=3, rect=(0,bot,1,1))

        for i in range(no_mrows):
            gs = all_axes[i,0].get_gridspec()
            for ax in all_axes[i]:
                ax.remove()
                tax = fig.add_subplot(gs[i,:])
                tax.set(xlabel='transition')
            fig.top_axes.append(tax)
        wax, tax, lax, gax, cax = fig.top_axes + [None,None]
        # wax.set(ylabel='estimated cumulative work')
        tax.set(yscale='log', ylabel='n_leapfrog')
        lax.set(ylabel='lp__')
        # gax.set(yscale='log', ylabel='sorted covariance eigenvalues')
        # cax.set(ylabel='correlation coefficients', ylim=[-1, 1])
        axes = fig.param_axes = all_axes[no_mrows:]
        for ax, label in zip(fig.param_axes[0], public_data.param_labels):
            ax.set(title=label)
        for ax in axes[1, -2*no_experiments:-no_experiments]:
            ax.set(xscale='log')
        for i in range(no_latent_params):
            for k, color in enumerate(['black', 'grey']):
                pp = possible_priors[k]
                eM, eS = pp[i, :2]
                truncation = pp[i, -1]
                ft = truncation
                xlim = eM * (eS ** np.array([-ft, ft]))
                pxlim = eM * (eS ** np.array([-ft-1, ft+1]))
                for j, ax in enumerate(axes[:, i]):
                    if j != 1:
                        ax.set(xscale='log')
                    if j == 0:
                        if k == 1:
                            # print(public_data.param_labels[i], eM, pxlim)
                            ax.set(xlim=pxlim)
                            for ax_ in axes[2:, i]:
                                ax_.set(xlim=pxlim)
                    elif j == 1:
                        trunc = np.array(fit.sample_data['std_truncation'])[:, i]
                        trunc[1] = min(1, trunc[1])
                        ax.set(xlim=np.exp(np.exp(
                            np.log(np.log(possible_priors[1][i,2])) +
                            trunc
                        )))
                        ax.axvline(pp[i, 2], color=color, zorder=100)
                        continue
                    else:
                        if k == 0:
                            eM,eS = public_data.posterior_person_parameters[i,:,j-2]
                            xlim = eM * (eS ** np.array([-ft, ft]))
                    for val in xlim.tolist():
                        ax.axvline(val, color=color, linestyle='--', zorder=100)
                    ax.axvline(eM, color=color, zorder=100)
        # if true_params is not None:
        #
        #     for ax, val in zip(axes.flat, true_params):
        #         ax.axvline(val, color='red', zorder=1000)


    color = kwargs['color']
    wax, tax, lax, gax, cax = fig.top_axes + [None,None]
    axes = fig.param_axes
    fig.fits.append(fit)
    x = .33 * (no_fits // no_lines)
    y = bot - ((no_fits % no_lines) * dy + dy)
    if prefix is None:
        prefix = f'datum-{no_fits}'
        # data_update = fit.scalar_data_update
        # if data_update:
        #     prefix += f' ({data_update})'
    summ = prefix + ': ' + fit.short_diagnosis.replace('\n', ' | ')
    print(summ)

    if title is not None:
        fig.text(
            0, bot, str(title),
            ha='left', va='center',
            color='black'
        )

    fig.text(
        x, y, summ,
        ha='left', va='center',
        color=color
    )

    if overlay and not init:
        zorder = 1 + int(overlay)
        for ax in axes.flat:
            xlim = x0, x1 = ax.get_xlim()
            # ylim = y0, y1 = ax.get_ylim()
            ax.axvspan(*xlim, color='white', alpha=overlay_alpha, zorder=zorder)
            # ax.fill_between(
            #     [x0,x1],[y0,y0],[y1,y1], color='white', alpha=overlay_alpha,
            #     zorder=zorder
            # )
            ax.set(xlim=xlim)
        kwargs['zorder'] = zorder

    for i in range(no_latent_params):
        tprint('param', i)
        cols = [f'population_eM.{i+1}', f'population_eS.{i+1}'] + [
            f'person_params.{j+1}.{i+1}' for j in range(no_persons)
        ]
        axes[0,i].get_shared_x_axes().join(axes[0, i], *axes[2:, i])
        for ax, col in zip(axes[:, i], cols):
            # if col != cols[1]:
            #     ax.set(xscale='log')

            xlim = ax.get_xlim()
            if col.startswith('population_eS'):
                bins = np.linspace(*xlim)
                log_scale = False
            else:
                bins = np.linspace(*np.log(xlim))/np.log(10)
                log_scale = 10
            # print(col, xlim, np.quantile(fit.lw_df[col], [0,1]), bins)
            fit.plot_hist(
                ax, col, label=f'{fit.lw_rhat(col):.2f}',
                # binrange=np.quantile(fit.lw_df[col], [0.05, .95]),
                log_scale=log_scale,
                bins=bins,
                stat='probability',
                **kwargs
            )
            ax.set(xlim=xlim)
            l = ax.legend(loc='best', title='Rhat')
            if overlay:
                l.set_zorder(1000)

    measurement_names = [
        'venous blood concentration',
        'exhaled air concentration'
    ]
    for experiment in range(no_experiments):
        for j in range(2):
            acol = axes[:, no_latent_params+2*experiment+j]
            ppm_exposure = public_data.ppm_exposures[experiment]
            exposure = public_data.exposures[experiment]
            acol[2].set(title=f'Ex {experiment} \
({ppm_exposure} PPM = {exposure:.2f} mg/l)\n{measurement_names[j]}')
            ax = acol[0]
            ax.set(
                xlabel='observed', ylabel='predicted',
                xscale='log', yscale='log'
            )
            if not fit.sample_data['likelihood']: continue
            ax = acol[1]
            col = f'noise.{j+1}'
            if experiment > 0:
                col = ['lp__', 'energy__'][j]
                # if j > 0:
                #     continue
                    # col = 'recomputed_lp'
                    # fit.lw_df[col] = fit.recompute_log_prob_grad(
                    #     refdata, cache_dir='out'
                    # ).lp__.to_numpy()
                if col not in fit.lw_df.columns: continue
            ax.set(xlim=np.quantile(fit.lw_df[col], [0,1]))
            fit.plot_hist(
                ax, col, label=f'{fit.lw_rhat(col):.2f}',
                stat='probability',
                **kwargs
            )
            l = ax.legend(loc='best', title='Rhat')
            if overlay:
                l.set_zorder(1000)

    if no_experiments:
        axes[2,no_latent_params].get_shared_x_axes().join(*axes[2:, no_latent_params])
        axes[2,no_latent_params].get_shared_y_axes().join(*axes[2:, no_latent_params])

    if no_experiments:
        meas_limits = np.nanquantile(
            fit.sample_data['experiments'], [0, 1],
            axis=(0, 2)
        )

    for i, row in enumerate(axes[2:, no_latent_params:]):
        tprint('person', i)
        for experiment in range(no_experiments):
            exposure = public_data.exposures[experiment]
            for j, ax in enumerate(row[2*experiment:][:2]):
                meas = fit.sample_data['experiments'][i,experiment]
                times = meas[:,0]
                ax.set(yscale='log')
                ppattern = f'^predicted_states\.{i+1}\.{experiment+1}\..+\.{j+1}$'
                fit.plot_fan(
                    ax, ppattern, x=times,
                    qs=[.05, .25, .5, .75, .95],
                    **kwargs
                )
                if init:
                    ax.axhline(exposure, color='grey', alpha=.5, zorder=100)
                    ax.plot(times, meas[:,1+j], '.', color='black', zorder=100)
                ax = axes[0, no_latent_params+2*experiment+j]
                mlim = meas_limits[:, experiment, 1+j]
                pmlim = [.9,1.1]*mlim
                ax.set(xlim=pmlim, ylim=pmlim)
                ax.plot(mlim, mlim, color='black')
                if not fit.sample_data['likelihood']: continue
                ax.plot(meas[:,1+j], fit[ppattern].mean(), 'x', **kwargs)


    gx = [0]
    # gvals = []
    # cvals = []
    # lfit = None
    for tfit in fit.fit_sequence:
        draw_idx = tfit.sequence_stop
        gx.append(draw_idx)
        # gvals.append(tfit.global_metric_eigenvalues)
        # cvals.append(tfit.correlation_coefficients)
        # lfit = tfit
    # gvals.insert(0, 1+0*gvals[0])
    # cvals.insert(0, 0*cvals[0])


    if 'n_leapfrog__' in fit.lw_df.columns:
        fit.plot_work(wax, colors=color)
        fit.plot_trace(tax, 'n_leapfrog__', colors=color)
        fit.plot_trace(lax, 'lp__', colors=color)
        lax.set(ylim=np.quantile(fit['lp__'], [0,1]))
    # gax.plot(gx, gvals, **kwargs)
    # cax.plot(gx, cvals, **kwargs)
    for ax in fig.top_axes:
        if ax is None: continue
        ax.set(xlim=(gx[0], max(gx[-1], ax.get_xlim()[1])))
        fit.plot_updates(ax, color=color)


    if path is not None and (not os.path.exists(path) or force):
        dirname = os.path.dirname(path)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname)
        fig.savefig(path)
        shutil.copy(path, 'current_fig.png')

    return fig
