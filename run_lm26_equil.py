import os
import sys
sys.path.append(os.environ['AURORA_REPOS'])

import numpy as np
import pandas as pd
import math
from flagships.gs_solver.fs_flagships import RunFlagships, LambdaAndBetaPol1Params, FluxConstrainedLambdaAndBetaPol1Params
from flagships.gs_solver.fs_curves import PiecewiseCurve
from flagships.gs_solver.fs_profile import LambdaCurveProfile
from flagships.gs_solver.fs_gs_solver import FSFluxNormalizedSolver
from flagships.gs_solver.fs_init_guess import PrevRunInitialGuess

EXT_PSI_DIR = 'data/csim_027c/csim-027c_ext_psi'
GEOM_DIR = 'data/csim_027c/csim-027c_geom'
SCALARS_TABLE_PATH = 'data/csim_027c/csim-027c_0.15Wb_n2e19_lampeak_eps0.7-t300.csv'

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

lambda_0_guess = 8
lambda_values = lambda_0_guess*(1 - psibar**2)
lambda_curve = PiecewiseCurve(psibar, lambda_values) 

ext_psi_scale_loc= [0.9999, 0.0] # Shouldn't matter as just setting scale to 1
out_folder = os.path.join('out')
soakscale = 0 
psi_lim_in = 0

scalars_df = pd.read_csv(SCALARS_TABLE_PATH, delimiter='\t')

prev_file = None
all_run_params = []
for i_row, row in scalars_df.iterrows():
    time = row["time(us)"]
    timestep_str = f'{time:.6f}'
    ext_psi_file = os.path.join(EXT_PSI_DIR, EXT_PSI_PREFEX + timestep_str + '.csv')
    geom_file = os.path.join(GEOM_DIR, GEOM_PREFIX + timestep_str + '.csv')
    out_file = f'lm26_test_{time:.6f}.hdf5'

    expected_opoint_r = 0.4 - time * 120
    res = int(30 * math.sqrt(0.35 / (expected_opoint_r - 0.05)))
    
    if prev_file is not None:
        run_params.init_guess = PrevRunInitialGuess(prev_file)

    run_params = FluxConstrainedLambdaAndBetaPol1Params(out_folder=out_folder, out_file=out_file, geom_file=geom_file,
                                        ext_psi_scale_loc=ext_psi_scale_loc, gun_femm_file=ext_psi_file, flux_setpoint=PSI_AXIS_SETPOINT,
                                        soak_file=ext_psi_file, soakscale=soakscale, lambda_curve=lambda_curve,
                                        Ishaft_in=1e6*row['Ishaft(MA)'], Ipl_in=1e6*row['Ipl'], beta_pol1_in=row['betap1'],
                                        psi_lim_in=psi_lim_in, expected_opoint=[expected_opoint_r, 0], use_csharp_solver=True,
                                        mesh_resolution=res)

    prev_file = os.path.join(out_folder, out_file)

    try:
        RunFlagships(run_params, write_history=True)
    except Exception as e:
        print(e)