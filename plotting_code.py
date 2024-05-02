import numpy as np
import matplotlib.pyplot as plt
from VoigtFit import load_dataset
from astropy.constants import c as speed_of_light
import astropy.units as u
from VoigtFit.io.output import rebin_spectrum, rebin_bool_array
from VoigtFit.container.lines import show_transitions
#from VoigtFit.funcs.voigt import Voigt, convolve
from VoigtFit.funcs.voigt import Voigt
from scipy.signal import fftconvolve, gaussian

plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']

def plot_region_fit(self, region_ind, sub_region_ind = None, vel_range = None,
                        ax = None, ax_resid = None, figsize = None, labelx = True, 
                        labely = True, comp_scale = None, fit_kwargs = {}, comp_kwargs = {},
                        plot_indiv_comps = False, use_flags = None, ylabel_as_ion = False, rebin_n = 1,
                        **kwargs):
        """
        method to plot a single region fit

        Parameters
        ----------
        region_ind: `int`
            index of region to plot
        sub_region_ind: `int`, optional, must be keyword
            if region has multiple transitions, specifies which index to plot
        vel_range: `list`, optional, must be keyword
            velocity range to plot, defaults ot +/- 500 km/s
        ax: `matplotlib.pyplot.axes`, optional, must be keyword
            axis to plot onto
        ax_resid: `matplotlib.pyplot.axes`, optional, must be keyword
            residual axis to plot onto, if provided
        figsize: `tuple`, optional, must be keyword
            sets figure size if ax not specified
        labelx: `bool`
            if True, labels x axis
        labely: `bool`
            if True, labels y axis
        comp_scale, 'number', optional, must be keyword
            size to make component marker relative to continuum error, default to 1
        fit_kwargs: `dict`
            keywords for fit profile plotting
        comp_kwargs: `dict`
            keywords for vlines marker component velocities

        """
        # if self.pre_rebin:
            # rebin_n = 1

        if ax is None:
            fig = plt.figure(figsize = figsize)
            gs = plt.GridSpec(8,8)

            ax = fig.add_subplot(gs[1:,:])
            ax_resid = fig.add_subplot(gs[0,:], sharex = ax)

        # Get ion info:
        lines = self.regions[region_ind].lines

        n_lines = len(lines)
        if n_lines > 1:
            if sub_region_ind == None:
                self._plot_region_fit_sub_ind = 1
                sub_region_ind = 0
            elif (sub_region_ind < (n_lines - 1)):
                self._plot_region_fit_sub_ind = sub_region_ind + 1
            else:
                self._plot_region_fit_sub_ind = None
        else:
            sub_region_ind = 0

        line = lines[sub_region_ind]

        ion = line.ion
        tag = line.tag

        # get params
        try:
            pars = self.best_fit
        except AttributeError:
            pars = self.pars

        # extract relevant params
        n_comp = len(self.components[ion])

        ion_pars = {"v":[], "b":[], "logN":[]}
        for ell in range(n_comp):
            ion_pars["v"].append((pars["z{}_{}".format(ell, ion)].value) * speed_of_light.to(u.km/u.s).value)
            ion_pars["b"].append(pars["b{}_{}".format(ell, ion)].value)
            ion_pars["logN"].append(pars["logN{}_{}".format(ell, ion)].value)


        # Get data
        wl, spec, err, spectra_mask_o = self.regions[region_ind].unpack()
        # if not self.pre_rebin:
        #     if self.dataset.regions[region_ind].specID.split("_")[-1] != str(self.filter_dict["G160M"]):
        #         rebin_n = 5
        #     else:
        #         rebin_n = 3
        # else:
        #     rebin_n = 1

        wl_r, spec_r, err_r = rebin_spectrum(wl, spec, err, rebin_n, method = 'mean')
        spectra_mask = rebin_bool_array(spectra_mask_o, rebin_n)

        mask_idx = np.where(spectra_mask == 0)[0]
        mask_idxp1 = mask_idx+1
        mask_idxn1 = mask_idx-1
        big_mask_idx = np.union1d(mask_idxp1[mask_idxp1<(len(spectra_mask)-1)], mask_idxn1[mask_idxn1>0])
        big_mask = np.ones_like(spectra_mask, dtype=bool)
        big_mask[big_mask_idx] = False

        l0_ref, f_ref, _ = line.get_properties()
        l_ref = l0_ref*(self.redshift+1)
        vel = (wl - l_ref) / l_ref * speed_of_light.to(u.km/u.s).value
        vel_r = (wl_r - l_ref) / l_ref * speed_of_light.to(u.km/u.s).value

        profile = get_profile(tag, ion_pars, vel_arr = vel)
        profile_r = get_profile(tag, ion_pars, vel_arr = vel_r)

        resid = profile_r["spec"] - spec_r
        cont_err = self.regions[region_ind].cont_err


        # plot spectra
        # Check kwargs
        if "lw" not in kwargs:
            kwargs["lw"] = 2
        if "alpha" not in kwargs:
            kwargs["alpha"] = 0.7
        if "drawstyle" not in kwargs:
            kwargs["drawstyle"] = "steps-mid"
        if "color" not in kwargs:
            kwargs["color"] = "k"

        if "lw" not in fit_kwargs:
            fit_kwargs["lw"] = kwargs["lw"]
        if "alpha" not in fit_kwargs:
            fit_kwargs["alpha"] = kwargs["alpha"]
        # if "drawstyle" not in fit_kwargs:
        #     fit_kwargs["drawstyle"] = kwargs["drawstyle"]
        if "color" not in fit_kwargs:
            fit_kwargs["color"] = "r"

        masked_kwargs = kwargs.copy()
        masked_kwargs["alpha"] *= 0.25



        # plot masked region
        _ = ax.plot(np.ma.masked_where(big_mask, vel_r), 
                    np.ma.masked_where(big_mask, spec_r), 
                    **masked_kwargs)

        if ax_resid != None:

            _ = ax_resid.plot(np.ma.masked_where(big_mask, vel_r), 
                              np.ma.masked_where(big_mask, resid), 
                              **masked_kwargs)

        # plot main region
        _ = ax.plot(np.ma.masked_where(~spectra_mask, vel_r), 
                    np.ma.masked_where(~spectra_mask, spec_r),
                    **kwargs)

        if ax_resid != None:
            _ = ax_resid.plot(np.ma.masked_where(~spectra_mask, vel_r), 
                          np.ma.masked_where(~spectra_mask, resid), 
                          **kwargs)


        if vel_range is None:
            vel_range = ax.set_xlim(-450, 450)
        else:
            vel_range = ax.set_xlim(vel_range)

        
        # plot continuum marker
        _ = ax.axhline(1., ls = "--", lw = 1, color = "k", zorder = -2, alpha = 0.7)
        if ax_resid != None:
            _ = ax_resid.axhline(0., ls = "--", lw = 1, color = "k", zorder = -2, alpha = 0.7)

        # plot continuum error range
        _ = ax.axhline(1+cont_err, ls=":", lw = 1, color = "k", alpha = 0.5, zorder = -2)
        _ = ax.axhline(1-cont_err, ls=":", lw = 1, color = "k", alpha = 0.5, zorder = -2)

        # plot resid error range
        if ax_resid != None:
            _ = ax_resid.fill_between(vel_r, 3*err_r, -3*err_r, color = fit_kwargs["color"], alpha = 0.1)

        # plot fit
        _ = ax.plot(profile["vel"], profile["spec"], **fit_kwargs)

        # lims
        xlim = ax.set_xlim(vel_range)
        ylim = ax.get_ylim()

        # add label
        if not ylabel_as_ion:
            _ = ax.text(xlim[0]+np.abs(np.diff(xlim)*.05), ylim[0]+np.abs(np.diff(ylim)*0.05), 
                        r"{} $\lambda${}".format(tag.split("_")[0], tag.split("_")[1]), 
                        fontsize = 12, 
                        ha = "left", 
                        va = "bottom")

        # axis labels
        if labelx:
            _ = ax.set_xlabel("LSR Velocity (km/s)", fontsize = 12)
        if labely:
            if not ylabel_as_ion:
                _ = ax.set_ylabel("Normalized Flux", fontsize = 12)
            else:
                _ = ax.set_ylabel(r"{} $\lambda${}".format(tag.split("_")[0], tag.split("_")[1]), 
                                  fontsize = 12)

        # set resid ylim
        if ax_resid != None:
            _ = ax_resid.set_ylim(np.nanmin(-7*err_r), np.nanmax(7*err_r))

        # mark components
        if comp_scale == None:
            comp_scale = .1
        for vel in ion_pars["v"]:
            vel_shifted = vel - self.redshift * speed_of_light.to(u.km/u.s).value
            if "alpha" not in comp_kwargs:
                comp_kwargs["alpha"] = kwargs["alpha"]
            if "color" not in comp_kwargs:
                comp_kwargs["color"] = "b"
            if "lw" not in comp_kwargs:
                comp_kwargs["lw"] = 2
            _ = ax.plot([vel_shifted, vel_shifted], [1 - comp_scale, 1 + comp_scale], 
                **comp_kwargs)

        if plot_indiv_comps:

            if use_flags != None:
                color_dict = {
                    "U":"orange",
                    "MC":"blue",
                    "MW":"magenta"
                }
                width_dict = {
                    "U":2,
                    "MC":3,
                }
                alpha_dict = {
                    "MC":1,
                    "U":0.8,
                }
                line_dict = {
                    "BB":"--"
                }
            else:
                color_dict = None
                line_dict = None
                width_dict = None


            for ell in range(n_comp):
                ion_pars = {"v":[], "b":[], "logN":[]}
                ion_pars["v"].append((pars["z{}_{}".format(ell, ion)].value) * speed_of_light.to(u.km/u.s).value)
                ion_pars["b"].append(pars["b{}_{}".format(ell, ion)].value)
                ion_pars["logN"].append(pars["logN{}_{}".format(ell, ion)].value)
                if use_flags != None:
                    if not ("B" in use_flags["{}_{}".format(ell, ion)]) | ("C" in use_flags["{}_{}".format(ell, ion)]):
                        if "MC" in use_flags["{}_{}".format(ell, ion)]:
                            alpha = alpha_dict["MC"]
                            color = color_dict["MC"]
                            width = width_dict["MC"]
                        elif "U" in use_flags["{}_{}".format(ell, ion)]:
                            alpha = alpha_dict["U"]
                            width = alpha_dict["U"]
                            color = color_dict["U"]
                        elif "MW" in use_flags["{}_{}".format(ell, ion)]:
                            color = color_dict["MW"]
                            alpha = 0.6
                            width = 2
                        else:
                            color = "green"
                            width = 2
                            alpha = 0.6


                        if "BB" in use_flags["{}_{}".format(ell, ion)]:
                            ls = line_dict["BB"]
                        else:
                            ls = "-"


                        single_profile = get_profile(tag, ion_pars, vel_arr = profile["vel"])
                        single_profile_masked = np.ma.masked_array(single_profile["spec"], 
                                                                    mask = single_profile["spec"] > .99)

                        _ = ax.plot(single_profile["vel"], single_profile_masked, 
                                    alpha = alpha, color = color, lw = width, ls = ls)

                else:
                    single_profile = get_profile(tag, ion_pars, vel_arr = profile["vel"])
                    single_profile_masked = np.ma.masked_array(single_profile["spec"], 
                                                                mask = single_profile["spec"] > .99)

                    _ = ax.plot(single_profile["vel"], single_profile_masked, 
                                alpha = 0.6, color = 'b', lw = 1, ls = "-")




        return plt.gcf()

def get_profile(ion_wav, pars, wl_arr = None, vel_arr = None, 
                    resolution = None, redshift = None, **kwargs):
        """
        Get absorption line profile for specified ion_wavelength

        Parameters
        ----------
        ion_wav: `str`
            ion_wavelength to get profile for
        pars: `dict`
            parameters for line profile
            must include "v", "b", and "logN"
        resolution: `number`, optional, must be keyword
            instrument resolution for convolution in km/s
            defualt to 20 km/s
        redshift: `number`, optional, must be keyword
            system redshift, default to 0
        wl_arr: `list-like`, optional, must be keyword
            array of wavelengths to compute spectrum to
        vel_arr: `list_like`, optional, must be keyword
            array of velocities to compute spectrum to
        """

        # find the right transition
        atomic_data = show_transitions(ion = ion_wav.split("_")[0])
        names = [at[0] for at in atomic_data]
        match = np.array(names) == ion_wav

        _, _, l0, f, gam, _ = np.array(atomic_data)[match][0]

        # check for resolution
        if resolution == None:
            resolution = 20. #km/s for COS

        # check redshift
        if redshift == None:
            redshift = 0.

        v = pars["v"]
        b = pars["b"]
        logN = pars["logN"]

        if ((type(v) == float) | (type(v) == np.float64)):
            v = list([v])
            b = list([b])
            logN = list([logN])

        l_center = l0 * (redshift + 1.)

        # find wl_line
        if vel_arr is None:
            if wl_arr is None:
                # use default wavelength range of +-500 km/s
                wl_arr = np.arange(l_center - .01*250, l_center +0.01*250, 0.01)

            vel_arr = (wl_arr - l_center)/l_center*(speed_of_light.to(u.km/u.s).value)

        elif wl_arr == None:
            wl_arr = (vel_arr / (speed_of_light.to(u.km/u.s).value) * l_center) + l_center

        tau = np.zeros_like(vel_arr)

        for (vv, bb, NN) in zip(v, b, logN):
            tau += Voigt(wl_arr, l0, f, 10**NN, 1.e5*bb, gam, z = vv/(speed_of_light.to(u.km/u.s).value))


        # Compute profile
        profile_int = np.exp(-tau)

        # convolve with instrument profile
        if isinstance(resolution, float):
            pxs = np.diff(wl_arr)[0] / wl_arr[0] * (speed_of_light.to(u.km/u.s).value)
            sigma_instrumental = resolution / 2.35482 / pxs
            LSF = gaussian(len(wl_arr) // 2, sigma_instrumental)
            LSF = LSF/LSF.sum()
            profile = fftconvolve(profile_int, LSF, 'same')
        else:
            profile = voigt.convolve(profile_int, resolution)



        out = {
            "wl":wl_arr,
            "vel":vel_arr,
            "spec":profile
        }

        return out
    
def plot_all_region_fits(self, fig = None, n_cols = None, 
                             figsize = None, vel_range = None,
                             ratio = None, ylim_lock = None, ylabel_as_ion = False,
                             fit_kwargs = {}, plot_indiv_comps = False, use_flags = None,
                             **kwargs):
        """
        method to plot all region fit

        Parameters
        ----------
        fig: `matplotlib.pyplot.figure`, optional, must be keyword
            Figure to use
        n_cols: `int`, optional, must be keyword 
            number of columns to create
        vel_range: `list`, optional, must be keyword
            velocity range to plot, defaults ot +/- 500 km/s
        figsize: `tuple`, optional, must be keyword
            sets figure size if ax not specified
        ratio: `int`, optional, must be keyword
            sets scaling of main plot to residiual plot, default to 5
        ylim_lock: `list`, optional, must be keyword
            if provided, sets all ylims to match provided values
        fit_kwargs: `dict`
            keywords for fit profile plotting

        """
        # check for figure
        if fig == None:
            fig = plt.figure(figsize = figsize)

        
        if ratio == None:
            ratio = 5
        gs_frame = ratio

        # determine number of plots needed to create
        all_lines = np.concatenate([region.lines for region in self.regions])
        n_lines = len(all_lines)

        if n_cols == None:
            n_cols = 1

        n_rows = np.ceil(n_lines/n_cols)

        gs_size = int(gs_frame * n_rows + n_rows - 1)
        gs = plt.GridSpec(gs_size,n_cols, hspace=0.)

        

        # make all axes:
        axs = []
        ax_resids = []
        row_counter = 0
        start_counter = 0
        col_counter = 0
        for ell in range(n_lines):
            if np.any(ylim_lock) != None:
                if ell == 0:
                    axs.append(fig.add_subplot(gs[start_counter+1:start_counter+gs_frame,col_counter]))
                    ylim = axs[0].set_ylim(ylim_lock)
                else:
                    axs.append(fig.add_subplot(gs[start_counter+1:start_counter+gs_frame,col_counter], 
                                               sharey = axs[0]))
            else:
                axs.append(fig.add_subplot(gs[start_counter+1:start_counter+gs_frame,col_counter]))
            ax_resids.append(fig.add_subplot(gs[start_counter,col_counter], sharex = axs[ell]))
            ax_resids[ell].axes.get_yaxis().set_visible(False)
            ax_resids[ell].axes.get_xaxis().set_visible(False)



            # add ylabel if necessary
            if not ylabel_as_ion:
                if col_counter == 0:
                    _ = axs[ell].set_ylabel("Normalized Flux")
            else:
                if col_counter == 1:
                    _ = axs[ell].yaxis.tick_right()
                    _ = axs[ell].yaxis.set_label_position("right")

            if row_counter == n_rows-1:
                _ = axs[ell].set_xlabel("LSR Velocity (km/s)")
            row_counter += 1
            if row_counter == n_rows:
                start_counter = 0
                col_counter += 1
            else:
                start_counter += gs_frame+1

        axs[-1].set_xlabel("LSR Velocity (km/s)")

        self._plot_region_fit_sub_ind = None
        plot_counter = 0
        for ell, region in enumerate(self.regions):
            # plot each spec

            _ = plot_region_fit(self, ell, 
                                     sub_region_ind = self._plot_region_fit_sub_ind, 
                                     ax = axs[plot_counter], 
                                     ax_resid = ax_resids[plot_counter], 
                                     labelx = False, 
                                     labely = ylabel_as_ion, 
                                     vel_range = vel_range, 
                                     plot_indiv_comps = plot_indiv_comps, 
                                     use_flags = use_flags,
                                     fit_kwargs = fit_kwargs,
                                     ylabel_as_ion = ylabel_as_ion,
                                     **kwargs)
            plot_counter += 1

            while self._plot_region_fit_sub_ind != None:
                _ = plot_region_fit(self, ell, 
                                         sub_region_ind = self._plot_region_fit_sub_ind, 
                                         ax = axs[plot_counter], 
                                         ax_resid = ax_resids[plot_counter], 
                                         labelx = False, 
                                         labely = ylabel_as_ion, 
                                         vel_range = vel_range,
                                         plot_indiv_comps = plot_indiv_comps, 
                                         use_flags = use_flags,
                                         fit_kwargs = fit_kwargs,
                                         ylabel_as_ion = ylabel_as_ion,
                                         **kwargs)
                plot_counter += 1



        return fig