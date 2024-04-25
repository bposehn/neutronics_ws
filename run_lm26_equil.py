import os

import numpy as np
import pandas as pd

from flagships.gs_solver.fs_flagships import FluxConstrainedLambdaAndBetaPol1Params, RunFlagships, LambdaAndBetaPol1Params
from flagships.gs_solver.fs_curves import PiecewiseCurve
from flagships.gs_solver.fs_profile import LambdaCurveProfile
from flagships.gs_solver.fs_gs_solver import FSFluxNormalizedSolver

EXT_PSI_DIR = 'data/csim-027c_ext_psi'
GEOM_DIR = 'data/csim-027c_geom'
SCALARS_TABLE_PATH = 'data/csim-027c_0.15Wb_n2e19_lampeak_eps0.7-t300.csv'

EXT_PSI_PREFEX = 'ext_psi_'
GEOM_PREFIX = 'geom_'

PSI_AXIS_SETPOINT = 150e-3 #Wb

'''
lambda = lambda0*(1-psibar^2), lambda0 = 150mWb
get Ishaft from scalars
Te = 50 + 250*psibar
ne = 4e19*( 1 - 0.67*psibar - 0.33*psibar​^4 )

'''

num_psibar = 1000
psibar = np.linspace(0, 1, num_psibar)

lambda_0_guess = 2
lambda_values = lambda_0_guess*(1 - psibar**2)
lambda_curve = PiecewiseCurve(psibar, lambda_values) 

ext_psi_scale_loc= [0.9999, 0.0] # Shouldn't matter as just setting scale to 1
out_folder = 'out'
soakscale = 0 # Can just use 0 and ext psi for file
psi_lim_in = 0

scalars_df = pd.read_csv(SCALARS_TABLE_PATH, delimiter='\t')

flux_normalized_solver = FSFluxNormalizedSolver(PSI_AXIS_SETPOINT)

all_run_params = []
for i_row, row in scalars_df.iterrows():
    # row = scalars_df.iloc[-1]
    timestep_str = f'{row["time(us)"]:.6f}'
    ext_psi_file = os.path.join(EXT_PSI_DIR, EXT_PSI_PREFEX + timestep_str + '.csv')
    geom_file = os.path.join(GEOM_DIR, GEOM_PREFIX + timestep_str + '.csv')

    # run_params = FluxConstrainedLambdaAndBetaPol1Params(out_folder=out_folder, out_file=timestep_str, geom_file=geom_file,
    #                                      ext_psi_scale_loc=ext_psi_scale_loc, gun_femm_file=ext_psi_file, flux_setpoint=PSI_AXIS_SETPOINT,
    #                                      soak_file=ext_psi_file, soakscale=soakscale, lambda_curve=lambda_curve,
    #                                      Ishaft_in=1e6*row['Ishaft(MA)'], Ipl_in=1e6*row['Ipl'], beta_pol1_in=row['betap1'],
    #                                      psi_lim_in=psi_lim_in, expected_opoint=[0.5, 0], verbosity=3)
    
    run_params = FluxConstrainedLambdaAndBetaPol1Params(out_folder=out_folder, out_file=timestep_str, geom_file=geom_file,
                                        ext_psi_scale_loc=ext_psi_scale_loc, gun_femm_file=ext_psi_file, flux_setpoint=PSI_AXIS_SETPOINT,
                                        soak_file=ext_psi_file, soakscale=soakscale, lambda_curve=lambda_curve,
                                        Ishaft_in=1e6*row['Ishaft(MA)'], Ipl_in=1e6*row['Ipl'], beta_pol1_in=0,
                                        psi_lim_in=psi_lim_in, expected_opoint=[0.5, 0], verbosity=3)
    
    # run_params = LambdaAndBetaPol1Params(out_folder=out_folder, out_file=timestep_str, geom_file=geom_file,
    #                                      ext_psi_scale_loc=ext_psi_scale_loc, gunscale=150e-3, gun_femm_file=ext_psi_file, 
    #                                      soak_file=ext_psi_file, soakscale=soakscale, lambda_curve=lambda_curve,
    #                                      Ishaft_in=1e6*row['Ishaft(MA)'], Ipl_in=1e6*row['Ipl'], beta_pol1_in=row['betap1'],
    #                                      psi_lim_in=psi_lim_in, verbosity=3, expected_opoint=[0.5, 0])

    # RunFlagships(run_params)
    # # all_run_params.append(run_params)
    # breakpoint()
    try:
        RunFlagships(run_params, write_history=True)
        breakpoint()
    except Exception as e:
        print(e)

# RunFlagships(run_params)
# RunFlagships(all_run_params)