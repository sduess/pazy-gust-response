from inspect import GEN_RUNNING
import numpy as np
import configobj

import os, sys


sys.path.append(os.path.join(os.path.dirname(__file__), "lib/pazy-model"))
sys.path.append(os.path.join(os.path.dirname(__file__), "lib/sharpy"))

import sharpy.sharpy_main
from pazy_wing_model import PazyWing

import sharpy.utils.algebra as algebra

def set_simulation_settings_dynamic(case_name, output_folder, case_route, gust_settings, u_inf = 1, rho = 1.225, surface_m = 4, gust_vanes = False, alpha = np.deg2rad(5.), symmetry_condition = False, delft_model = False):
     # simulation settings
    config = configobj.ConfigObj()
    config.filename = case_route + '/{}.sharpy'.format(case_name)
    settings = dict()
    gravity = True #bool(not gust_vanes)
    n_step = 5
    structural_relaxation_factor = 0.6
    relaxation_factor = 0.2
    tolerance = 1e-6 #
    fsi_tolerance = 1e-4 #8
    num_cores = 4
    variable_wake =  False #gust_vanes

    u_inf = 18.
    rho = 1.205
    # geometry parameters 
    chord = 0.1
    # if gust_vanes:
    #     chord = 0.3
    #dt = 0.1/surface_m #2.5e-4
    CFL = 1
    dt = CFL * chord / surface_m / u_inf
    # dtu_inf  = 2.5e-4
    n_tstep = 10000
    alpha = np.deg2rad(10.)
    # # Test case 1
    # alpha = np.deg2rad(5.)
    gust_frequency = 5.7 # Hz
    gust_T = 0.175439 #s 
    gust_amplitude = 0.805562
    # Test case 2
    # alpha = np.deg2rad(10.)
    # frequency_gust = 3.2 # Hz
    # gust_amplitude = 0.64458
    # gust_T = 0.312500000000000

    

    wake_length = 20
    settings['SHARPy'] = {
        'flow': ['BeamLoader',
                 'Modal',
                 'AerogridLoader',
                 'BeamPlot',
                 'AerogridPlot',
                #  'StaticUvlm',
                 'StaticCoupled',
                 'AeroForcesCalculator',
                 'LiftDistribution',
                 'Modal',
                #  'BeamPlot',
                #  'AerogridPlot',
                #  'DynamicCoupled',
                'SaveData'
                 ],
        'case': case_name, 'route': case_route,
        'write_screen': 'on', 'write_log': 'on',
        'log_folder': output_folder + '/' + case_name + '/',
        'log_file': case_name + '.log'}
    print("\n\nALPHA  = ", alpha)
    print("delft model = ", delft_model)
    if delft_model:
        orientation =  algebra.euler2quat(np.array([0,
                                                    0,
                                                    alpha]))
    else:
        orientation =  algebra.euler2quat(np.array([0.,
                                                    alpha,
                                                    0]))
    settings['BeamLoader'] = {'unsteady': 'on',
                              'orientation': orientation}

    settings['AerogridLoader'] = {
        'unsteady': 'on',
        'aligned_grid': 'on',
        'mstar': wake_length * surface_m,
        'wake_shape_generator': 'StraightWake',
        'wake_shape_generator_input': {'u_inf': u_inf,
                                       'dt': dt}}

    settings['NonLinearStatic'] = {'print_info': 'off',
                                   'max_iterations': 1000,
                                   'num_load_steps': 12,
                                   # 'num_steps': 10,
                                   # 'dt': 2.5e-4,
                                   'delta_curved': 1e-5,
                                   'min_delta': 1e-6,
                                   'gravity_on': gravity,
                                   'relaxation_factor': 0.1,
                                   'gravity': 9.81}
    settings['AeroForcesCalculator'] =  {'write_text_file': 'off',
                                         'screen_output': 'on'}
    print("symmetry_condition = ", symmetry_condition)
    settings['StaticUvlm'] = {
            'rho': rho, # Check why?? 1e-8,
            'print_info': True,
            'horseshoe': 'off',
            'num_cores': 4,
            'n_rollup': 0, #settings['AerogridLoader']['mstar'],
            'rollup_dt': dt,
            'rollup_aic_refresh': 1,
            'rollup_tolerance': 1e-4,
            'velocity_field_generator': 'SteadyVelocityField',
            'velocity_field_input': {
                'u_inf': u_inf,
                'u_inf_direction': [1.0, 0., 0.]},
            'vortex_radius': 1e-9,
            'symmetry_condition': symmetry_condition,
        }
    settings['StaticCoupled'] = {
        'print_info': 'on',
        'max_iter': 200,
        'n_load_steps': 8,
        'tolerance': 1e-5,
        'relaxation_factor': 0.2,
        'aero_solver': 'StaticUvlm',
        'aero_solver_settings':  {
            'rho': rho, # Check why?? 1e-8,
            'print_info': True,
            'horseshoe': 'off',
            'num_cores': 4,
            'n_rollup': 0, #settings['AerogridLoader']['mstar'],
            'rollup_dt': dt,
            'rollup_aic_refresh': 1,
            'rollup_tolerance': 1e-4,
            'velocity_field_generator': 'SteadyVelocityField',
            'velocity_field_input': {
                'u_inf': u_inf,
                'u_inf_direction': [1.0, 0., 0.]},
            'vortex_radius': 1e-9,
            'symmetry_condition': symmetry_condition,
        },
        'structural_solver': 'NonLinearStatic',
        'structural_solver_settings': settings['NonLinearStatic']}

    settings['BeamPlot'] = {}

    settings['AerogridPlot'] = {'include_rbm': 'off',
                                'include_applied_forces': 'on',
                                'minus_m_star': 0}
    settings['SaveData'] = {'save_aero': True,
                            'save_struct': True,
                            'save_linear': True,}


    settings['Modal'] = {'NumLambda': 40,
                        'rigid_body_modes': 'on',
                        'print_matrices': 'off',
                        'continuous_eigenvalues': 'off',
                        'write_modes_vtk': 'off',
                        'use_undamped_modes': 'on'}

    if gust_vanes: 
        cs_deflection_file = '/home/sduess/Documents/Aircraft Models/Pazy/pazy-gust-response/02_gust_vanes/cs_deflection_amplitude_{}_frequency_{}_mean_{}.csv'.format(gust_amplitude, gust_frequency, 0)
        time = np.linspace(0., n_tstep * dt, n_tstep)
        cs_deflection_prescribed = float(gust_amplitude) * np.sin(2 * np.pi * float(gust_frequency) * time)
        np.savetxt(cs_deflection_file, cs_deflection_prescribed)

        
    settings['StepUvlm'] = {'num_cores': num_cores,
                            'convection_scheme': 3,
                            'gamma_dot_filtering': 7,
                            'cfl1': variable_wake,
                            'velocity_field_generator': 'SteadyVelocityField',
                            'velocity_field_input': {'u_inf':u_inf,
                                                    'u_inf_direction': [1., 0, 0]},
                            # 'control_surface_deflection' : ['DynamicControlSurface'],
                            # 'control_surface_deflection_generator_settings': {'0': {'dt': dt,'deflection_file': cs_deflection_file},},
                            'velocity_field_generator': 'GustVelocityField',
                            'velocity_field_input':{'u_inf': u_inf,
                                                    'u_inf_direction': [1., 0, 0],
                                                    'relative_motion': True,
                                                    'offset': 10 *dt * u_inf, #gust_settings['offset'],
                            #                         # 'gust_shape': 'time varying',
                            #                         # 'gust_parameters': {'file': '../02_gust_input/turbulence_time_600s_uinf_45_altitude_800_moderate_noise_seeds_10434_10435_10436_10437.txt',
                            #                         #                     'yaw_angle': np.deg2rad(45),},
                            #                         'gust_shape': '1-cos',
                            #                         'gust_parameters': {'gust_length': gust_settings['length'],
                            #                                             'gust_intensity': gust_settings['intensity']*u_inf,
                            #                                             'yaw_angle': np.deg2rad(0.),
                            #                                         }     
                                                    'gust_shape': 'continuous_sin',
                                                    'gust_parameters': {'gust_length':  gust_T * u_inf, #gust_settings['length'],
                                                                        'gust_intensity': 2*gust_amplitude, # gust_settings['intensity']*u_inf,
                                                                    }                                                                
                                                },
                            'rho': rho,
                            'n_time_steps': n_tstep,
                            'dt': dt,
                            'symmetry_condition': symmetry_condition,
                            }

    settings['NonLinearDynamicPrescribedStep'] = {'print_info': 'on',
                'max_iterations': 950,
                'delta_curved': 1e-1,
                'min_delta': tolerance,
                'newmark_damp': 1e-4,
                'gravity_on': gravity,
                'gravity': 9.81,
                'num_steps': n_tstep,
                'dt': dt,
                # 'initial_velocity': u_inf,
                }
    
    settings['DynamicCoupled'] = {'structural_solver':'NonLinearDynamicPrescribedStep',
        'structural_solver_settings': settings['NonLinearDynamicPrescribedStep'],
        'aero_solver': 'StepUvlm',
        'aero_solver_settings': settings['StepUvlm'],
        'fsi_substeps': 200,
        'fsi_tolerance': fsi_tolerance,
        'relaxation_factor': relaxation_factor,
        'minimum_steps': 1,
        'relaxation_steps': 150,
        'final_relaxation_factor': 0.05,
        'n_time_steps': n_tstep,
        'print_info': True,
        'dt': dt,
        'include_unsteady_force_contribution': True, 
    #   'postprocessors': ['BeamLoads','SaveData'],
        'postprocessors': ['BeamLoads', 'BeamPlot', 'AerogridPlot', 'SaveData'],
        'postprocessors_settings': {
                                    'BeamLoads': {'csv_output': 'off'},
                                    'BeamPlot': {'include_rbm': 'on',
                                                'include_applied_forces': 'on'},
                                    'AerogridPlot': {
                                        'include_rbm': 'on',
                                        'include_applied_forces': 'on',
                                        # 'minus_m_star': 60,
                                        },
                                    'SaveData': settings['SaveData'],
                                    },
    }
    print('Gust vanes = ', gust_vanes)

    # if not gust_vanes:
        
        # settings['StepUvlm']['velocity_field_generator'] = 'SteadyVelocityField'
        # settings['StepUvlm']['velocity_field_input'] = {'u_inf':u_inf,
        #                                                 # 'relative_motion': True,
        #                                                 'u_inf_direction': [1., 0, 0]}
        # settings['StepUvlm']['convection_scheme'] = 2

        # settings['AerogridLoader']['gust_vanes'] = False

        # settings['DynamicCoupled']['controller_id'] = {'controller_tip': 'SurfaceOscillator'}
        # settings['DynamicCoupled']['controller_settings'] = {'controller_tip': {'frequency': gust_settings['frequency'],
        #                                                                         'amplitude': gust_settings['amplitude'],
        #                                                                         'mean': gust_settings['mean'],                                                                                ''
        #                                                                         'dt': dt,
        #                                                                         'time_history_input_file': 'psi.csv',
        #                                                                         'controller_log_route': './output/' + case_name + '/',
        #                                                                         'controlled_surfaces': [0]}}
    # else:
    if gust_vanes:
        gust_vane_parameters = {
            'M': 8,
            'N':20, #, 
            'M_star': 20, 
            'span': 5, #10., 
            'chord': 0.3, 
            'control_surface_deflection_generator_settings': {
                'dt': dt, 
                'deflection_file': '/home/sduess/Documents/Aircraft Models/Pazy/pazy-gust-response/02_gust_vanes/cs_deflection_amplitude_0.08726646259971647_frequency_5.7_mean_0.0.csv' 
            }
        }

        settings['AerogridLoader']['gust_vanes'] = True
        settings['AerogridLoader']['gust_vanes_generator_settings'] = {'n_vanes': 2,
                                                                       'streamwise_position': [-1.5, -1.5],
                                                                       'vertical_position': [-0.25, 0.25],
                                                                       'symmetry_condition': symmetry_condition,
                                                                       'vane_parameters': [gust_vane_parameters, gust_vane_parameters]
                                                                      }
        settings['StepUvlm']['convection_scheme'] = 3

        #  For convergence study steady velocity field
        settings['StepUvlm']['velocity_field_generator'] = 'SteadyVelocityField'
        settings['StepUvlm']['velocity_field_input'] = {'u_inf':u_inf,
                                                        'u_inf_direction': [1., 0, 0]}
                                                                       
    # settings['StepUvlm']['velocity_field_generator'] = 'GustVelocityField',
    # settings['StepUvlm']['velocity_field_input'] = {'u_inf': u_inf,
    #                                             'u_inf_direction': [1., 0, 0],
    #                                             'relative_motion': True,
    #                                             'offset': gust_settings['offset'],
    #                                             'gust_shape': 'continuous_sin',
    #                                             'gust_parameters': {'gust_length':  gust_T * u_inf, #gust_settings['length'],
    #                                                                 'gust_intensity': gust_amplitude, # gust_settings['intensity']*u_inf,
    #                                                             }                                                                
    #                                         },
    for k, v in settings.items():
        config[k] = v

    config.write()
    return settings

def setup_pazy_model(case_name, case_route, pazy_settings, gust_vanes = False, symmetry_condition = False):
    pazy = PazyWing(case_name, case_route, pazy_settings)
    pazy.generate_structure()
    if not symmetry_condition:
        print('mirror wing')
        pazy.structure.mirror_wing()
    pazy.generate_aero()

    pazy.save_files()
    return pazy

# def setup_gust_vanes(case_name, case_route = './cases/', output_route = './output/',
#                      surface_m = 8, n_elem_multiplier = 1):

#     # import gust_vane_model
#     gust_vane_model = gust_vane_model.GustVane(case_name, case_route, output_route)
#     gust_vane_model.clean()
#     gust_vane_model.init_structure(sigma=1, n_elem_multiplier=n_elem_multiplier,
#                                                chord = 0.3, span = 3.,
#                                                streamwise_position = 0.,
#                                                vertical_position = 0)

#     gust_vane_model.init_aero(m=surface_m) 
#     gust_vane_model.generate()   

def run_test_gust_vane(case_name, test_case = '1', case_root='./cases/', output_folder='./output/'):
    surface_m = 4
    n_elem_multiplier = 2
    # setup_gust_vanes(case_name, surface_m = surface_m, n_elem_multiplier=n_elem_multiplier)

    gust_settings = {'amplitude': np.deg2rad(5.),
                     'frequency': 5.7,
                     'mean': 0.}
    set_simulation_settings_dynamic(case_name, output_folder, case_root, gust_settings, surface_m, gust_vanes = True)

    data = sharpy.sharpy_main.main(['', case_root + case_name + '.sharpy'])
    return data
def run_dynamic_prescriped_simulation_with_gust_input(skin_on, case_root='./cases/', output_folder='./output/', gust_vanes = False, symmetry_condition = False):
    # pazy model settings
    # Norberto:  M = 16, N = 64
    pazy_model_settings = {'skin_on': skin_on,
                        'discretisation_method': 'michigan',
                        'num_elem': 2,
                        'surface_m': 8}
    case_name = 'pazy_nonsymmetry_unsteady_plus_gust_vanes_fine'
    case_route = case_root + '/' + case_name + '/'

    if not os.path.isdir(case_route):
        os.makedirs(case_route, exist_ok=True)

    setup_pazy_model(case_name, case_route, pazy_model_settings, symmetry_condition = symmetry_condition)

    gust_settings= {'length': 5,
                    'offset': 0,
                    'intensity': 0.2,}
    set_simulation_settings_dynamic(case_name, output_folder, case_route, gust_settings, pazy_model_settings['surface_m'], gust_vanes = gust_vanes, symmetry_condition = symmetry_condition)

    data = sharpy.sharpy_main.main(['', case_route + case_name + '.sharpy'])
    return data

if __name__ == '__main__':

    dict_test_cases = {'1': {'alpha': np.deg2rad(5.),
                             'gust_T': 0.175439,
                             'frequency_gust_vane': 5.7, #Hz
                             'gust_amplitude':0.805562,
                             },
                        '2': {'alpha': np.deg2rad(10.),
                             'gust_T': 0.64458,
                             'frequency_gust_vane': 3.2, #Hz
                             'gust_amplitude':0.3125}}

    symmetry_condition = False #True
    data = run_dynamic_prescriped_simulation_with_gust_input(skin_on='on',
                                                             case_root='./cases/',
                                                             output_folder='./output/',
                                                             gust_vanes = True,
                                                             symmetry_condition = symmetry_condition)
    # run_test_gust_vane('gust_vane_test')
    # setup_gust_vanes('gust_vane_test', 
    #                  surface_m = 8,
    #                  n_elem_multiplier=1)

                                                              

