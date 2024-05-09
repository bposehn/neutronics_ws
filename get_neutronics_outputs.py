import os
from typing import Callable

import matplotlib.axes
import matplotlib.contour
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy import integrate
from scipy import interpolate

from flagships.post_processing.ParseFlagshipsFile import BaseFlagshipsParser, FlagshipsParser2
from flagships.post_processing.EquilibriumPlotting import Plot2ndOrderContour
from flagships.Csharp.csharp_utils import asNetArray, asNumpyArray, set_debug_mode, append_gf_recon_dll_path

import clr
set_debug_mode(True)
append_gf_recon_dll_path()
clr.AddReference("GFRecon")
from Reconstruction import FSPostProcessor, MagFieldCalculator, PsiBarCalculator, FlagshipsMeshHelper
from Reconstruction import InterpArray, RZPoint, Contour, SurfaceAreaCalc, FunctionData, TriangleFuncSurface

UNEXTENDED_CHORD = {'r_collimator': 0.0075, 'dist_to_plasma': 3, 'chord_name': 'NES_test', 'r1': 0.08270806550754287, \
         'r2': 1.4548300691146028, 'x1': -0.03204, 'x2': 0.9022, 'y1': -0.07625, 'y2': 1.1413, 'z1': 0.0, 'z2': 2.0}

MIN_REACTION_VESSEL_Z = -0.5767828282828283
CHORD = FlagshipsParser2.extend_chord_to_z_value(UNEXTENDED_CHORD, MIN_REACTION_VESSEL_Z)

EQUIL_DIR = 'csim_027c_equil'
OUTPUT_DIR = 'neutron_calcs_out'

PSIBAR_PROFILE = np.linspace(0, 1, 101)
# BASE_T_e = 300*np.ones_like(PSIBAR_PROFILE) 
BASE_T_e = 50 + 250*(1-PSIBAR_PROFILE)
BASE_n_e = 4e19*(1 - (2/3)*PSIBAR_PROFILE - (1/3)*PSIBAR_PROFILE**4)

BASE_NES_DIST = CHORD['dist_to_plasma']

BASE_T_e_CALLABLE = interpolate.interp1d(PSIBAR_PROFILE, BASE_T_e, bounds_error=False, \
                                         fill_value=(BASE_T_e[0], np.nan))
BASE_n_e_CALLABLE = interpolate.interp1d(PSIBAR_PROFILE, BASE_n_e, bounds_error=False, \
                                         fill_value=(BASE_n_e[0], np.nan))

PLASMA_DATA_CSV = 'data/csim_027c/plasma_data.csv'
PLASMA_DATA_DF = pd.read_csv(PLASMA_DATA_CSV)
CV_OF_T_INTERPD = interpolate.interp1d(PLASMA_DATA_DF['t(s)'], PLASMA_DATA_DF['CV'], bounds_error=False, \
                                       fill_value=(PLASMA_DATA_DF['CV'].iloc[0], PLASMA_DATA_DF['CV'].iloc[-1]))
T_GAIN_OF_T_INTERPD = interpolate.interp1d(PLASMA_DATA_DF['t(s)'], PLASMA_DATA_DF['T_gain'], bounds_error=False, \
                                    fill_value=(PLASMA_DATA_DF['T_gain'].iloc[0], PLASMA_DATA_DF['T_gain'].iloc[-1]))

def get_T_e_callable_at_time(time_s: float):
    return lambda psibar : T_GAIN_OF_T_INTERPD(time_s) * BASE_T_e_CALLABLE(psibar)

def get_n_e_callable_at_time(time_s: float):
    return lambda psibar : CV_OF_T_INTERPD(time_s) * BASE_n_e_CALLABLE(psibar)

def get_neutron_yield_df():
    equil_filepaths = [equil_name for equil_name in os.listdir(EQUIL_DIR) if equil_name.endswith('.hdf5')]

    data = np.empty((len(equil_filepaths), 2))
    column_names = ['time(s)', 'neutron rate(s^-1)']

    for i_equil, equil_name in enumerate(equil_filepaths):
        parser = BaseFlagshipsParser.create(EQUIL_DIR, equil_name)
        equil_time = float(equil_name[-13:-5])
        n_e_profile = get_n_e_callable_at_time(equil_time)
        t_e_profile = get_T_e_callable_at_time(equil_time)
        total_neutron_yield = parser.calc_total_DD_neutron_yield(n_e_profile, t_e_profile)
        
        data[i_equil, 0] = equil_time
        data[i_equil, 1] = total_neutron_yield

    df = pd.DataFrame(data, columns=column_names)
    df = df.sort_values(by='time(s)')
    
    return df

def get_nes_plasma_dist_df(nes_plasma_dists: np.ndarray):
    equil_filepaths = [equil_name for equil_name in os.listdir(EQUIL_DIR) if equil_name.endswith('.hdf5')]

    data = np.empty((len(equil_filepaths), len(nes_plasma_dists)+1))
    columns_names = ['time(s)'] + [f'nes_plasma_dist: {nes_plasma_dist}m' for nes_plasma_dist in nes_plasma_dists]

    for i_equil, equil_name in enumerate(equil_filepaths):
        parser = BaseFlagshipsParser.create(EQUIL_DIR, equil_name)
        equil_time = float(equil_name[-13:-5])
        data[i_equil, 0] = equil_time

        n_e_profile = get_n_e_callable_at_time(equil_time)
        t_e_profile = get_T_e_callable_at_time(equil_time)
        base_distance_spectrometer_output = parser.calc_DD_neutron_spectrometer_output(CHORD, n_e_profile, t_e_profile)

        for i_nes_plasma_dist, nes_plasma_dist in enumerate(nes_plasma_dists):
            data[i_equil, 1+i_nes_plasma_dist] = base_distance_spectrometer_output * (BASE_NES_DIST**2 / nes_plasma_dist**2)

    df = pd.DataFrame(data, columns=columns_names)
    df = df.sort_values(by='time(s)')

    return df

def get_nes_temperature_hists(nes_plasma_dists: np.ndarray, min_timestep: float, ion_temp_bins: np.ndarray, plot_output_dir: str):
    equil_filepaths = [equil_name for equil_name in os.listdir(EQUIL_DIR) if equil_name.endswith('.hdf5')]

    equil_filepaths_past_min_timestep = [filepath for filepath in equil_filepaths if float(filepath[-13:-5]) > min_timestep]
    equil_timesteps = np.array([float(filepath[-13:-5]) for filepath in equil_filepaths_past_min_timestep])

    equil_timesteps_order = np.argsort(equil_timesteps)

    equil_filepaths_past_min_timestep = [equil_filepaths_past_min_timestep[i] for i in equil_timesteps_order]
    equil_timesteps = equil_timesteps[equil_timesteps_order]    

    num_chord_points = 1000
    num_extra_time_resolution_bins = 500 # To get finer temperature resolution for numerical integration using midpoint method
    
    yields_along_chord = np.empty(((len(equil_filepaths_past_min_timestep)-1)*num_extra_time_resolution_bins, \
                                    len(nes_plasma_dists), num_chord_points))
    Ts_along_chord = np.copy(yields_along_chord)

    for i_equil in range(len(equil_filepaths_past_min_timestep)-1):
        equil_name_0 = equil_filepaths_past_min_timestep[i_equil]
        equil_name_1 = equil_filepaths_past_min_timestep[i_equil + 1]

        parser_0 = BaseFlagshipsParser.create(EQUIL_DIR, equil_name_0)
        parser_1 = BaseFlagshipsParser.create(EQUIL_DIR, equil_name_1)

        timestep_0 = equil_timesteps[i_equil]
        timestep_1 = equil_timesteps[i_equil + 1]

        n_e_profile_0 = get_n_e_callable_at_time(timestep_0)
        T_e_profile_0 = get_T_e_callable_at_time(timestep_0)
        base_nes_neutron_flux_along_chord_0, T_along_chord_0 = \
            parser_0.calc_DD_neutron_spectrometer_yield_and_temp_along_chord(CHORD, n_e_profile_0, \
                                                                    T_e_profile_0, num_chord_points=num_chord_points)
        
        n_e_profile_1 = get_n_e_callable_at_time(timestep_1)
        T_e_profile_1 = get_T_e_callable_at_time(timestep_1)
        base_nes_neutron_flux_along_chord_1, T_along_chord_1 = \
            parser_1.calc_DD_neutron_spectrometer_yield_and_temp_along_chord(CHORD, n_e_profile_1, \
                                                                    T_e_profile_1, num_chord_points=num_chord_points)

        print(f'Max temperature seen by chord: {max(np.nanmax(T_along_chord_0), np.nanmax(T_along_chord_1))}')

        T_slopes = (T_along_chord_1 - T_along_chord_0) / num_extra_time_resolution_bins
        extra_time_resolution_dt = (timestep_1 - timestep_0) / num_extra_time_resolution_bins

        for i_nes_plasma_dist, nes_plasma_dist in enumerate(nes_plasma_dists):
            nes_neutron_flux_along_chord_0 = \
                base_nes_neutron_flux_along_chord_0 * (BASE_NES_DIST**2 / nes_plasma_dist**2)
            nes_neutron_flux_along_chord_1 = \
                base_nes_neutron_flux_along_chord_1 * (BASE_NES_DIST**2 / nes_plasma_dist**2)

            nes_neutron_flux_slopes = (nes_neutron_flux_along_chord_1 - nes_neutron_flux_along_chord_0) / \
                                                                                 num_extra_time_resolution_bins

            for i_time_resolution in range(num_extra_time_resolution_bins):
                interpolated_neutron_flux_along_chord_0 = \
                    nes_neutron_flux_along_chord_0 + (i_time_resolution * nes_neutron_flux_slopes)
                interpolated_neutron_flux_along_chord_1 = \
                    nes_neutron_flux_along_chord_0 + ((i_time_resolution + 1) * nes_neutron_flux_slopes)

                interpolated_T_along_chord_0 = T_along_chord_0 + (i_time_resolution * T_slopes)
                interpolated_T_along_chord_1 = T_along_chord_0 + ((i_time_resolution + 1) * T_slopes)

                # print(f'max Rel T change: {np.nanmax(abs(interpolated_T_along_chord_1 - interpolated_T_along_chord_0)/(0.5*(interpolated_T_along_chord_1 + interpolated_T_along_chord_0)))}')

                #Manually integrate neutron flux over time to get yield
                yields_along_chord[i_equil*num_extra_time_resolution_bins + i_time_resolution, i_nes_plasma_dist] = \
                    0.5 * extra_time_resolution_dt * \
                        (interpolated_neutron_flux_along_chord_0 + interpolated_neutron_flux_along_chord_1) 
                Ts_along_chord[i_equil*num_extra_time_resolution_bins + i_time_resolution, i_nes_plasma_dist] = \
                    0.5 * (interpolated_T_along_chord_0 + interpolated_T_along_chord_1)
            
    os.makedirs(plot_output_dir, exist_ok=True)

    for i_nes_plasma_dist, nes_plasma_dist in enumerate(nes_plasma_dists):
        plt.hist(Ts_along_chord[:, i_nes_plasma_dist, :].flatten(),  \
                 weights=yields_along_chord[:, i_nes_plasma_dist, :].flatten(), bins=ion_temp_bins)
        plt.xlabel('T_i (eV)')
        plt.ylabel('Neutron Yield')
        plt.yscale('log')
        plt.title(f'Neutron Temp in Last 10us of Shot\nNES Dist to Plasma: {nes_plasma_dist}m')
        plt.savefig(os.path.join(plot_output_dir, f'{nes_plasma_dist}m.png'))
        plt.close()

def plot_function_of_psibar(ax: matplotlib.axes.Axes, parser: BaseFlagshipsParser,
                             function_of_psibar: Callable[[float], float], colorbar_label='',
                             n_contours=10, contour_values=None):
    psibar_field = asNumpyArray(parser.GetPsiBarField().DOFValues)
    function_field = function_of_psibar(psibar_field)

    if contour_values == None:
        contour_values = np.linspace(np.min(function_field), np.max(function_field)*.99, n_contours+1)[1:]

    function_data = FunctionData(function_field, parser.cs_helper)

    cmap = plt.cm.get_cmap('rainbow')
    norm = matplotlib.colors.Normalize(vmin=min(contour_values), vmax=max(contour_values))
    
    for i_contour_value, contour_value in enumerate(contour_values):
        color=cmap(norm(contour_value))
        contours = parser.cs_helper.GetContours(contour_value, function_data)
        for contour in contours:
            if contour.IsClosed:
                contour.PrepareForPlotting(function_data)
                Plot2ndOrderContour(parser.cs_helper, contour, color=color, function=function_data, axes=ax)
    
    smap = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    plt.colorbar(mappable=smap, ax=ax, label=colorbar_label)

def make_n_and_T_profile_plots():
    equil_filepaths = [equil_name for equil_name in os.listdir(EQUIL_DIR) if equil_name.endswith('.hdf5')]
    equil_timesteps = np.array([float(filepath[-13:-5]) for filepath in equil_filepaths])

    equil_timesteps_order = np.argsort(equil_timesteps)

    equil_filepaths = [equil_filepaths[i] for i in equil_timesteps_order]
    equil_timesteps = equil_timesteps[equil_timesteps_order] 

    n_pts = 1000
    chord_xs = np.linspace(CHORD['x1'], CHORD['x2'], n_pts)
    chord_ys = np.linspace(CHORD['y1'], CHORD['y2'], n_pts)
    chord_zs = np.linspace(CHORD['z1'], CHORD['z2'], n_pts)
    chord_rs = np.sqrt(chord_xs**2 + chord_ys**2)

    plot_loc_dir = 'n_and_T_profiles'
    os.makedirs(os.path.join(OUTPUT_DIR, plot_loc_dir), exist_ok=True)

    for i_equil, equil_filepath in enumerate(equil_filepaths):
        t = equil_timesteps[i_equil]

        n_e_callable = get_n_e_callable_at_time(t)
        T_e_callable = get_T_e_callable_at_time(t)

        parser =  BaseFlagshipsParser.create(EQUIL_DIR, equil_filepath)

        edge_values = parser.m2.GetEdgePoints()

        fig, axs = plt.subplots(1, 2, figsize=(15, 6))

        neutron_lum_callable = lambda psibar: parser.calc_DD_neutron_luminosity(n_e_callable(psibar), T_e_callable(psibar))
                
        plot_function_of_psibar(axs[0], parser, neutron_lum_callable, colorbar_label='Neutron Luminosity (n s^-1 m^-3)')
        plot_function_of_psibar(axs[1], parser, T_e_callable, colorbar_label='$T_i$ (eV)')

        for i_axis in range(2):
            axs[i_axis].plot(chord_rs, chord_zs, c='xkcd:hot pink', label='NES Chord')
            axs[i_axis].plot(edge_values[:, 0], edge_values[:, 1], c='k')
            axs[i_axis].set_xlabel('r(m)')
            axs[i_axis].set_ylabel('z(m)')
            axs[i_axis].legend()
            axs[i_axis].set_xlim([np.min(edge_values[:, 0])-.1, np.max(edge_values[:, 0])+.1])
            axs[i_axis].set_ylim([np.min(edge_values[:, 1])-.1, np.max(edge_values[:, 1])+.1])
        
        plt.suptitle(f'Compression time: {t}s')
        plt.savefig(os.path.join(OUTPUT_DIR, plot_loc_dir, f'{t:.6f}.png'))
        plt.close(fig)

def generate_outputs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    make_n_and_T_profile_plots()

    neutron_yield_df = get_neutron_yield_df()

    #Plot of peak ion temperature vs compression time
    peak_ion_temps = np.array([get_T_e_callable_at_time(time)(1) for time in neutron_yield_df['time(s)']])
    plt.plot(neutron_yield_df['time(s)'], peak_ion_temps*1e-3)
    plt.xlabel('Compression Time (s)')
    plt.ylabel('Peak Ion Temperature (keV)')
    plt.yscale('log')
    plt.savefig(os.path.join(OUTPUT_DIR, 'peak_T_i_vs_t.png'))
    plt.close()
    
    # Plot of neutron production rate (neutrons/s) vs compression time
    plt.plot(neutron_yield_df['time(s)'], neutron_yield_df['neutron rate(s^-1)'])
    plt.xlabel('Compression Time (s)')
    plt.ylabel('Neutron Production Rate (1/s)')
    plt.yscale('log')
    plt.savefig(os.path.join(OUTPUT_DIR, 'n_yield_vs_t.png'))
    plt.close()

    # Total neutron production yield from a shot
    total_neutron_production = integrate.trapezoid(neutron_yield_df['neutron rate(s^-1)'], neutron_yield_df['time(s)'])
    with open(os.path.join(OUTPUT_DIR, 'total_neutron_production.txt'), 'w') as f:
        f.write(str(int(total_neutron_production)))

    # NES
    nes_plasma_dists = np.arange(3, 11)
    # nes_plasma_dists = np.arange(3, 5)
    nes_plasma_dist_df = get_nes_plasma_dist_df(nes_plasma_dists)
    
    # Peak neutron rate at spectrometer
    peak_nes_rates = nes_plasma_dist_df.loc[:, nes_plasma_dist_df.columns != 'time(s)'].max()
    peak_nes_rates_df = pd.DataFrame(np.hstack((nes_plasma_dists[np.newaxis].T, peak_nes_rates.values[np.newaxis].T)), 
                                     columns=['NES Plasma Distance (m)', 'Peak NES Rate (1/s)'])
    peak_nes_rates_df.to_csv(os.path.join(OUTPUT_DIR,'peak_nes_rates.csv'), index=False)

    # Plot of neutron rate at spectrometer vs compression time
    nes_rate_vs_time_dirname = os.path.join(OUTPUT_DIR, 'nes_rate_vs_time_plots')
    os.makedirs(nes_rate_vs_time_dirname, exist_ok=True)
    non_time_columns = [colname for colname in nes_plasma_dist_df.columns if 'time' not in colname]
    for colname in non_time_columns:
        nes_distance_str = colname[colname.find(':')+2:]
        plt.plot(nes_plasma_dist_df['time(s)'], nes_plasma_dist_df[colname])
        plt.xlabel('time(s)')
        plt.ylabel('Neutron Rate at NES (1/s)')
        plt.title(f'NES Dist to Plasma: {nes_distance_str}')
        plt.yscale('log')
        plt.savefig(os.path.join(nes_rate_vs_time_dirname, nes_distance_str))
        plt.close()

    # Histogram of number of neutrons produced vs temperature during the final 10 us of compression time, 1 keV binning; 
    # min_timestep = nes_plasma_dist_df['time(s)'].max() - 10e-6
    min_timestep = PLASMA_DATA_DF['t(s)'].max() - 10e-6
    nes_hists_dirname = os.path.join(OUTPUT_DIR, 'nes_final_10us_hists')
    ion_temp_bins = np.arange(1, 11)*1.0e3

    get_nes_temperature_hists(nes_plasma_dists, min_timestep, ion_temp_bins, nes_hists_dirname)

if __name__ == '__main__':
    generate_outputs()