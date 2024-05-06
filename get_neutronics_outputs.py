import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy import interpolate

from flagships.post_processing.ParseFlagshipsFile import BaseFlagshipsParser

CHORD = {'r_collimator': 0.0075, 'dist_to_plasma': 3, 'chord_name': 'NES_test', 'r1': 0.08270806550754287, \
         'r2': 1.4548300691146028, 'x1': -0.03204, 'x2': 0.9022, 'y1': -0.07625, 'y2': 1.1413, 'z1': 0.0, 'z2': 2.0}

psibar_profile = np.linspace(0, 1, 101)
T_e = 50 + 250*psibar_profile
n_e = 4e19*(1 - (2/3)*psibar_profile - (1/3)*psibar_profile**3)

T_e_callable = interpolate.interp1d(psibar_profile, T_e, fill_value='extrapolate')
n_e_callable = interpolate.interp1d(psibar_profile, n_e, fill_value='extrapolate')

EQUIL_DIR = 'equil'
OUTPUT_DIR = 'neutron_calcs_out'

def get_neutron_yield_df():
    equil_filepaths = [equil_name for equil_name in os.listdir(EQUIL_DIR) if equil_name.endswith('.hdf5')]

    data = np.empty((len(equil_filepaths), 2))
    column_names = ['time(s)', 'neutron rate(s^-1)']

    for i_equil, equil_name in enumerate(equil_filepaths):
        parser = BaseFlagshipsParser.create(EQUIL_DIR, equil_name)
        total_neutron_yield = parser.calc_total_DD_neutron_yield(n_e_callable, T_e_callable)
        equil_time = float(equil_name[-13:-5])
        
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
        for i_nes_plasma_dist, nes_plasma_dist in enumerate(nes_plasma_dists):
            data[i_equil, 1+i_nes_plasma_dist] = parser.calc_DD_neutron_spectrometer_output(CHORD, n_e_callable,\
                                                                                     T_e_callable, nes_plasma_dist)

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
        if i_equil == len(equil_timesteps) - 1:
            timestep = equil_timesteps[i_equil] - equil_timesteps[i_equil - 1]
        else:
            timestep = equil_timesteps[i_equil + 1] - equil_timesteps[i_equil]

        for i_nes_plasma_dist, nes_plasma_dist in enumerate(nes_plasma_dists):
            # some of these values are none....
            nes_neutron_flux_along_chord, T_along_chord = \
                parser.calc_DD_neutron_spectrometer_yield_and_temp_along_chord(CHORD, n_e_callable, T_e_callable, nes_plasma_dist)
            
            yields_along_chord[i_equil, i_nes_plasma_dist] = nes_neutron_flux_along_chord * timestep
            Ts_along_chord[i_equil, i_nes_plasma_dist] = T_along_chord

    os.makedirs(plot_output_dir, exist_ok=True)

    for i_nes_plasma_dist, nes_plasma_dist in enumerate(nes_plasma_dists):
        plt.hist(Ts_along_chord[:, i_nes_plasma_dist, :].flatten(),  \
                 weights=yields_along_chord[:, i_nes_plasma_dist, :].flatten(), bins=ion_temp_bins)
        plt.xlabel('T_i (eV)')
        plt.ylabel('Neutron Yield')
        plt.title(f'Neutron Temp in Last 10us of Shot\nNES Dist to Plasma: {nes_plasma_dist}m')
        plt.savefig(os.path.join(plot_output_dir, f'{nes_plasma_dist}m.png'))

def generate_outputs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    neutron_yield_df = get_neutron_yield_df()

    # Plot of neutron production rate (neutrons/s) vs compression time
    plt.scatter(neutron_yield_df['time(s)'], neutron_yield_df['neutron rate(s^-1)'])
    plt.xlabel('Compression Time (s)')
    plt.ylabel('Neutron Production Rate (1/s)')
    plt.savefig(os.path.join(OUTPUT_DIR, 'n_yield_vs_t.png'))

    # Total neutron production yield from a shot
    total_neutron_production = integrate.trapezoid(neutron_yield_df['neutron rate(s^-1)'], neutron_yield_df['time(s)'])
    with open(os.path.join(OUTPUT_DIR, 'total_neutron_production.txt'), 'w') as f:
        f.write(str(total_neutron_production))

    # NES
        
    nes_plasma_dists = np.arange(3, 11)

    nes_plasma_dist_df = get_nes_plasma_dist_df(nes_plasma_dists)
    
    # Peak neutron rate at spectrometer
    peak_nes_rates = nes_plasma_dist_df.loc[:, nes_plasma_dist_df.columns != 'time(s)'].max()
    peak_nes_rates.to_csv('peak_nes_rates.csv', index=False)

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
        plt.savefig(os.path.join(nes_rate_vs_time_dirname, nes_distance_str))

    # Histogram of number of neutrons produced vs temperature during the final 10 us of compression time, 1 keV binning; 
    min_timestep = nes_plasma_dist_df['time(s)'].max() - 10e-6
    nes_hists_dirname = os.path.join(OUTPUT_DIR, 'nes_final_10us_hists')
    ion_temp_bins = np.arange(1, 11)*1.0e3

    get_nes_temperature_hists(nes_plasma_dists, min_timestep, ion_temp_bins, nes_hists_dirname)

if __name__ == '__main__':
    generate_outputs()