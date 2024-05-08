import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy import interpolate

from flagships.post_processing.ParseFlagshipsFile import BaseFlagshipsParser, FlagshipsParser2

UNEXTENDED_CHORD = {'r_collimator': 0.0075, 'dist_to_plasma': 3, 'chord_name': 'NES_test', 'r1': 0.08270806550754287, \
         'r2': 1.4548300691146028, 'x1': -0.03204, 'x2': 0.9022, 'y1': -0.07625, 'y2': 1.1413, 'z1': 0.0, 'z2': 2.0}

MIN_REACTION_VESSEL_Z = -0.5767828282828283
CHORD = FlagshipsParser2.extend_chord_to_z_value(UNEXTENDED_CHORD, MIN_REACTION_VESSEL_Z)

EQUIL_DIR = 'csim_027c_equil'
OUTPUT_DIR = 'neutron_calcs_out'

PSIBAR_PROFILE = np.linspace(0, 1, 101)
# BASE_T_e = 300*np.ones_like(PSIBAR_PROFILE) 
BASE_T_e = 50 + 250*PSIBAR_PROFILE
BASE_n_e = 4e19*(1 - (2/3)*PSIBAR_PROFILE - (1/3)*PSIBAR_PROFILE**4)

BASE_NES_DIST = CHORD['dist_to_plasma']

BASE_T_e_CALLABLE = interpolate.interp1d(PSIBAR_PROFILE, BASE_T_e, fill_value='extrapolate')
BASE_n_e_CALLABLE = interpolate.interp1d(PSIBAR_PROFILE, BASE_n_e, fill_value='extrapolate')

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
    yields_along_chord = np.empty((len(equil_filepaths_past_min_timestep), len(nes_plasma_dists), num_chord_points))
    Ts_along_chord = np.copy(yields_along_chord)

    for i_equil, equil_name in enumerate(equil_filepaths_past_min_timestep):
        parser = BaseFlagshipsParser.create(EQUIL_DIR, equil_name)

        if i_equil == 0:
            timestep_0 = equil_timesteps[i_equil]
            timestep_1 = equil_timesteps[i_equil + 1]
        else:
            timestep_0 = equil_timesteps[i_equil - 1]
            timestep_1 = equil_timesteps[i_equil]

        n_e_profile_0 = get_n_e_callable_at_time(timestep_0)
        T_e_profile_0 = get_T_e_callable_at_time(timestep_0)
        base_nes_neutron_flux_along_chord_0, T_along_chord_0 = \
            parser.calc_DD_neutron_spectrometer_yield_and_temp_along_chord(CHORD, n_e_profile_0, T_e_profile_0)
        
        n_e_profile_1 = get_n_e_callable_at_time(timestep_1)
        T_e_profile_1 = get_T_e_callable_at_time(timestep_1)
        base_nes_neutron_flux_along_chord_1, T_along_chord_1 = \
            parser.calc_DD_neutron_spectrometer_yield_and_temp_along_chord(CHORD, n_e_profile_1, T_e_profile_1)

        print(f'Max temperature seen by chord: {max(np.nanmax(T_along_chord_0), np.nanmax(T_along_chord_1))}')

        for i_nes_plasma_dist, nes_plasma_dist in enumerate(nes_plasma_dists):
            nes_neutron_flux_along_chord_0 = base_nes_neutron_flux_along_chord_0 * (BASE_NES_DIST**2 / nes_plasma_dist**2)
            nes_neutron_flux_along_chord_1 = base_nes_neutron_flux_along_chord_1 * (BASE_NES_DIST**2 / nes_plasma_dist**2)

            #Integrate neutron flux over time to get yield
            yields_along_chord[i_equil, i_nes_plasma_dist] = 0.5 * (nes_neutron_flux_along_chord_1 + nes_neutron_flux_along_chord_0) * (timestep_1 - timestep_0)
            Ts_along_chord[i_equil, i_nes_plasma_dist] = 0.5 * (T_along_chord_0 + T_along_chord_1)

            # TODO determine if it is a reasonable assumption to say that all neutrons produced at a point in a given time interval can be the assumed as the average
            
            
            # Get that T changes mean ~20% as much as nflux changes, max 
            # print(f'mean Rel T change: {np.nanmean(abs(T_along_chord_0 - T_along_chord_1)/(0.5*(T_along_chord_0 + T_along_chord_1)))}')
            # print(f'mean Rel nflux change: {np.nanmean(abs(nes_neutron_flux_along_chord_0 - nes_neutron_flux_along_chord_1)/(0.5*(nes_neutron_flux_along_chord_0 + nes_neutron_flux_along_chord_1)))}')

            print(f'max Rel T change: {np.nanmax(abs(T_along_chord_0 - T_along_chord_1)/(0.5*(T_along_chord_0 + T_along_chord_1)))}')
            # print(f'max Rel nflux change: {np.nanmax(abs(nes_neutron_flux_along_chord_0 - nes_neutron_flux_along_chord_1)/(0.5*(nes_neutron_flux_along_chord_0 + nes_neutron_flux_along_chord_1)))}')

    os.makedirs(plot_output_dir, exist_ok=True)

    for i_nes_plasma_dist, nes_plasma_dist in enumerate(nes_plasma_dists):
        plt.hist(Ts_along_chord[:, i_nes_plasma_dist, :].flatten(),  \
                 weights=yields_along_chord[:, i_nes_plasma_dist, :].flatten(), bins=ion_temp_bins)
        plt.xlabel('T_i (eV)')
        plt.ylabel('Neutron Yield')
        plt.title(f'Neutron Temp in Last 10us of Shot\nNES Dist to Plasma: {nes_plasma_dist}m')
        plt.savefig(os.path.join(plot_output_dir, f'{nes_plasma_dist}m.png'))
        plt.clf()

def generate_outputs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    neutron_yield_df = get_neutron_yield_df()

    #Plot of peak ion temperature vs compression time
    peak_ion_temps = np.array([get_T_e_callable_at_time(time)(1) for time in neutron_yield_df['time(s)']])
    plt.plot(neutron_yield_df['time(s)'], peak_ion_temps*1e-3)
    plt.xlabel('Compression Time (s)')
    plt.ylabel('Peak Ion Temperature (keV)')
    plt.yscale('log')
    plt.savefig(os.path.join(OUTPUT_DIR, 'peak_T_i_vs_t.png'))
    plt.clf()
    
    # Plot of neutron production rate (neutrons/s) vs compression time
    plt.plot(neutron_yield_df['time(s)'], neutron_yield_df['neutron rate(s^-1)'])
    plt.xlabel('Compression Time (s)')
    plt.ylabel('Neutron Production Rate (1/s)')
    plt.yscale('log')
    plt.savefig(os.path.join(OUTPUT_DIR, 'n_yield_vs_t.png'))
    plt.clf()

    # Total neutron production yield from a shot
    total_neutron_production = integrate.trapezoid(neutron_yield_df['neutron rate(s^-1)'], neutron_yield_df['time(s)'])
    with open(os.path.join(OUTPUT_DIR, 'total_neutron_production.txt'), 'w') as f:
        f.write(str(int(total_neutron_production)))

    # NES
    nes_plasma_dists = np.arange(3, 11)
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
        plt.clf()

    # Histogram of number of neutrons produced vs temperature during the final 10 us of compression time, 1 keV binning; 
    # min_timestep = nes_plasma_dist_df['time(s)'].max() - 10e-6
    min_timestep = PLASMA_DATA_DF['t(s)'].max() - 10e-6
    nes_hists_dirname = os.path.join(OUTPUT_DIR, 'nes_final_10us_hists')
    ion_temp_bins = np.arange(1, 11)*1.0e3

    get_nes_temperature_hists(nes_plasma_dists, min_timestep, ion_temp_bins, nes_hists_dirname)

if __name__ == '__main__':
    generate_outputs()