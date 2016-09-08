"""Plotting functions related to dalecv2.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.mlab as mlab
import matplotlib.gridspec as gridspec
import mod_class as mc
import seaborn as sns

# ------------------------------------------------------------------------------
# Plot observation time series
# ------------------------------------------------------------------------------


def plot_drive_dat(dat, dC):
    """Plots specified driving data from data class, dC. Takes an
    observation string, a dataClass (dC) and a start and finish point.
    """
    sns.set_context(rc={'lines.linewidth': 0.8, 'lines.markersize': 6})
    fig, ax = plt.subplots(nrows=1, ncols=1)
    drive_dict = {'t_mean': dC.t_mean, 't_max': dC.t_max, 't_min': dC.t_min,
                  'I': dC.I,}
    palette = sns.color_palette("colorblind", 11)

    ax.plot(dC.dates, drive_dict[dat], color=palette[0])
    ax.set_xlabel('Year')
    ax.set_ylabel(dat)
    plt.gcf().autofmt_xdate()
    return ax, fig


def plot_ob_dict(ob, dC):
    """Plots a specified observation from the observation dictionary in a data class, dC. Takes an
    observation string, a dataClass (dC).
    """
    sns.set_context(rc={'lines.linewidth': 0.8, 'lines.markersize': 6})
    fig, ax = plt.subplots(nrows=1, ncols=1)

    palette = sns.color_palette("colorblind", 11)

    ax.plot(dC.dates, dC.ob_dict[ob], 'o', color=palette[0])
    ax.set_xlabel('Year')
    ax.set_ylabel(ob)
    plt.gcf().autofmt_xdate()
    return ax, fig


def plot_obs(ob, pvals, dC):
    """Plots a specified observation using obs eqn in mod_class module. Takes an
    observation string, a dataClass (dC).
    """
    sns.set_context(rc={'lines.linewidth': .8, 'lines.markersize': 6})
    fig, ax = plt.subplots(nrows=1, ncols=1)
    m = mc.DalecModel(dC)
    mod_lst = m.mod_list(pvals)
    obs_lst = m.oblist(ob, mod_lst)

    palette = sns.color_palette("colorblind", 11)

    ax.plot(dC.dates, obs_lst, color=palette[0])
    ax.set_xlabel('Year')
    ax.set_ylabel(ob)
    plt.gcf().autofmt_xdate()
    return ax, fig


# ------------------------------------------------------------------------------
# Assimilation plotting functions
# ------------------------------------------------------------------------------


def plot_4dvar(ob, dC, xb=None, xa=None, erbars=1, awindl=None, obdict_a=None):
    """Plots a model predicted observation value for two initial states (xb and xa)
    and also the actual observations taken of the physical quantity. Takes a ob
    string, two initial states (xb,xa), a dataClass and a start and finish
    time step.
    """
    sns.set_context(rc={'lines.linewidth':.8, 'lines.markersize':6})
    fig, ax = plt.subplots(nrows=1, ncols=1)
    m = mc.DalecModel(dC)
    palette = sns.color_palette("colorblind", 11)
    if xb != None:
        mod_lst = m.mod_list(xb)
        obs_lst = m.oblist(ob, mod_lst)
        ax.plot(dC.dates, obs_lst, color=palette[0])
    if xa != None:
        mod_lst = m.mod_list(xa)
        obs_lst = m.oblist(ob, mod_lst)
        ax.plot(dC.dates, obs_lst, color=palette[1])

    ob_dict = dC.ob_dict
    ob_err_dict = dC.ob_err_dict
    if ob in ob_dict.keys():
        if erbars == True:
            ax.errorbar(dC.dates, ob_dict[ob], yerr=ob_err_dict[ob],
                         fmt='o', label=ob+'_o', color=palette[2], alpha=0.7)
        else:
            ax.plot(dC.dates, ob_dict[ob], 'o', label=ob+'_o', color=palette[2])
    if obdict_a != None:
        ax.plt.plot(dC.dates, obdict_a[ob], 'o')

    if awindl != None:
        ax.axvline(x=dC.dates[awindl], color='k', ls='dashed')

    ax.set_xlabel('Year')
    ax.set_ylabel(ob)
    plt.gcf().autofmt_xdate()

    return ax, fig


def plotscatterobs(ob, pvals, dC, awindl, bfa='a'):
    """Plots scatter plot of obs vs model predicted values. Takes an initial
    parameter set, a dataClass (must have only desired ob for comparison
    specified in dC), assimilation window length and whether a comparison of
    background 'b', forecast 'f' or analysis 'a' is desired.
    """
    sns.set_context('poster', font_scale=1.5, rc={'lines.linewidth': 1., 'lines.markersize': 6.})
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    sns.set_style('ticks')
    palette = sns.color_palette("colorblind", 11)
    m = mc.DalecModel(dC)
    mod_lst = m.mod_list(pvals)
    obs_lst = m.oblist(ob, mod_lst)
    y_obs = dC.ob_dict[ob]
    plt_ob_lst = (y_obs/y_obs)*obs_lst
    one_one = np.arange(int(min(min(y_obs[np.isnan(y_obs) != True]), min(plt_ob_lst[np.isnan(y_obs) != True]))),
                        int(max(max(y_obs[np.isnan(y_obs) != True]), max(plt_ob_lst[np.isnan(y_obs) != True]))))
    plt.plot(one_one, one_one, color=palette[0])
    if bfa == 'b' or bfa == 'a':
        ax.plot(y_obs[0:awindl], plt_ob_lst[0:awindl], 'o', color=palette[1])
        error = np.sqrt(np.nansum((y_obs[0:awindl] - plt_ob_lst[0:awindl])**2)/len(y_obs[0:awindl]))
        yhx = np.nanmean(y_obs[0:awindl] - plt_ob_lst[0:awindl])
    elif bfa == 'f':
        ax.plot(y_obs[awindl:], plt_ob_lst[awindl:], 'o', color=palette[1])
        error = np.sqrt(np.nansum((y_obs[awindl:] - plt_ob_lst[awindl:])**2)/len(y_obs[awindl:]))
        yhx = np.nanmean(y_obs[awindl:] - plt_ob_lst[awindl:])
    else:
        raise Exception('Please check function input for bfa variable')
    plt.xlabel(ob.upper()+r' observations (g C m$^{-2}$ day$^{-1}$)')
    plt.ylabel(ob.upper()+' model (g C m$^{-2}$ day$^{-1}$)')
    #plt.title(bfa+'_error=%f, mean(y-hx)=%f' %(error,yhx))
    print bfa+'_error=%f, mean(y-hx)=%f' %(error, yhx)
    #plt.xlim((-20, 15))
    #plt.ylim((-20, 15))
    return ax, fig


# ------------------------------------------------------------------------------
# Twin experiment plot fns
# ------------------------------------------------------------------------------


def plot_scatter_twin(ob, pvals, dC, awindl, bfa='a'):
    """Plots scatter plot of obs vs model predicted values. Takes an initial
    parameter set, a dataClass (must have only desired ob for comparison
    specified in dC), assimilation window length and whether a comparison of
    background 'b', forecast 'f' or analysis 'a' is desired.
    """
    sns.set_context('poster', font_scale=1.5, rc={'lines.linewidth': 1., 'lines.markersize': 6.})
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    sns.set_style('ticks')
    palette = sns.color_palette("colorblind", 11)
    m = mc.DalecModel(dC)
    mod_lst = m.mod_list(pvals)
    mod_lst_truth = m.mod_list(dC.x_truth)
    obs_lst = m.oblist(ob, mod_lst)
    y_obs = m.oblist(ob, mod_lst_truth)
    one_one = np.arange(int(min(min(y_obs), min(obs_lst))),
                        int(max(max(y_obs), max(obs_lst))))
    plt.plot(one_one, one_one, color=palette[0])
    if bfa == 'b' or bfa == 'a':
        ax.plot(y_obs[0:awindl], obs_lst[0:awindl], 'o', color=palette[1])
        error = np.sqrt(np.nansum((y_obs[0:awindl] - obs_lst[0:awindl])**2)/len(y_obs[0:awindl]))
        yhx = np.nanmean(y_obs[0:awindl] - obs_lst[0:awindl])
        mod_obs_bar = np.mean(obs_lst[0:awindl])
        std_mod_obs = np.std(obs_lst[0:awindl])
        obs_bar = np.mean(y_obs[0:awindl])
        std_obs = np.std(y_obs[0:awindl])
        rms = np.sqrt(np.sum([((obs_lst[x]-mod_obs_bar)-(y_obs[x]-obs_bar))**2 for x in xrange(awindl)])/len(awindl))
        corr_coef = (np.sum([((obs_lst[x]-mod_obs_bar)*(y_obs[x]-obs_bar)) for x in xrange(awindl)])/len(awindl))\
                /(std_mod_obs*std_obs)
    elif bfa == 'f':
        ax.plot(y_obs[awindl:], obs_lst[awindl:], 'o', color=palette[1])
        error = np.sqrt(np.nansum((y_obs[awindl:] - obs_lst[awindl:])**2)/len(y_obs[awindl:]))
        yhx = np.nanmean(y_obs[awindl:] - obs_lst[awindl:])
        mod_obs_bar = np.mean(obs_lst[awindl:])
        std_mod_obs = np.std(obs_lst[awindl:])
        obs_bar = np.mean(y_obs[awindl:])
        std_obs = np.std(y_obs[awindl:])
        rms = np.sqrt(np.sum([((obs_lst[x]-mod_obs_bar)-(y_obs[x]-obs_bar))**2
                              for x in xrange(awindl,len(y_obs))])/len(y_obs[awindl:]))
        corr_coef = (np.sum([((obs_lst[x]-mod_obs_bar)*(y_obs[x]-obs_bar))
                             for x in xrange(awindl,len(y_obs))])/len(y_obs[awindl:]))/(std_mod_obs*std_obs)
    else:
        raise Exception('Please check function input for bfa variable')
    plt.xlabel(ob.upper()+r' observations (g C m$^{-2}$ day$^{-1}$)')
    plt.ylabel(ob.upper()+' model (g C m$^{-2}$ day$^{-1}$)')
    plt.title('mean(y-hx)=%.2f, rms=%.2f, corr_coef=%.2f' %( yhx, rms, corr_coef))
    print bfa+'_error=%f, mean(y-hx)=%f, rms=%f, corr_coef=%f' %(error, yhx, rms, corr_coef)
    #plt.xlim((-20, 15))
    #plt.ylim((-20, 15))
    return ax, fig


def plot_4dvar_twin(ob, dC, xb=None, xa=None, erbars=1, awindl=None, obdict_a=None):
    """Plots a model predicted observation value for two initial states (xb,xa)
    and also the actual observations taken of the physical quantity. Takes a ob
    string, two initial states (xb,xa), a dataClass and a start and finish
    time step.
    """
    sns.set_context(rc={'lines.linewidth':.8, 'lines.markersize':6})
    fig, ax = plt.subplots(nrows=1, ncols=1)
    palette = sns.color_palette("colorblind", 11)
    m = mc.DalecModel(dC)
    mod_lst = m.mod_list(dC.x_truth)
    obs_lst = m.oblist(ob, mod_lst)
    ax.plot(dC.dates, obs_lst, color=palette[3])
    if xb != None:
        mod_lst = m.mod_list(xb)
        obs_lst = m.oblist(ob, mod_lst)
        ax.plot(dC.dates, obs_lst, ':', color=palette[0])
    if xa != None:
        mod_lst = m.mod_list(xa)
        obs_lst = m.oblist(ob, mod_lst)
        ax.plot(dC.dates, obs_lst, color=palette[1])

    mod_lst = m.mod_list(dC.x_truth)
    obs_lst = m.oblist(ob, mod_lst)
    ax.plot(dC.dates, obs_lst, '--', color=palette[3])

    ob_dict = obdict_a
    ob_err_dict = dC.ob_err_dict
    #if ob in ob_dict.keys():
    #    if erbars == True:
    ##        ax.errorbar(dC.dates, ob_dict[ob], yerr=ob_err_dict[ob],
    #                     fmt='o', label=ob+'_o', color=palette[2], alpha=0.7)
    #    else:
    #        ax.plot(dC.dates, ob_dict[ob], 'o', label=ob+'_o', color=palette[2])
    if obdict_a != None:
        ax.plot(dC.dates[0:awindl], obdict_a[ob], 'o', color=palette[2])

    if awindl != None:
        ax.axvline(x=dC.dates[awindl], color='k', ls='dashed')

    ax.set_xlabel('Year')
    ax.set_ylabel(ob)
    plt.gcf().autofmt_xdate()

    return ax, fig


def plottwinerr(truth, xb, xa, xb_lab='xb', xa_lab='xa'):
    """Plot error between truth and xa/xb shows as a bar chart.
    """
    sns.set_context('poster', font_scale=1.5, rc={'lines.linewidth': 1, 'lines.markersize': 10})
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,5))
    sns.set_style('ticks')
    n = 23
    width = 0.35
    ind = np.arange(n)
    rects1 = ax.bar(ind, 100*abs(truth-xb)/truth, width, color=sns.xkcd_rgb["faded green"], label=xb_lab+'_err')
    rects2 = ax.bar(ind+width, 100*abs(truth-xa)/truth, width, color=sns.xkcd_rgb["pale red"], label=xa_lab+'_err')
    ax.set_ylabel('% error')
    ax.set_title('% error in parameter values for '+xa_lab+' and '+xb_lab)
    ax.set_xticks(ind+width)
    keys = [r'$\theta_{min}$', r'$f_{auto}$', r'$f_{fol}$', r'$f_{roo}$', r'$c_{lspan}$', r'$\theta_{woo}$',
            r'$\theta_{roo}$', r'$\theta_{lit}$', r'$\theta_{som}$', r'$\Theta$', r'$c_{eff}$', r'$d_{onset}$',
            r'$f_{lab}$', r'$c_{ronset}$', r'$d_{fall}$', r'$c_{rfall}$', r'$c_{lma}$', r'$C_{lab}$', r'$C_{fol}$',
            r'$C_{roo}$', r'$C_{woo}$', r'$C_{lit}$', r'$C_{som}$']
    ax.set_xticklabels(keys, rotation=90)
    ax.legend()
    return ax, fig