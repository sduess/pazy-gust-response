from inspect import GEN_RUNNING
import numpy as np
import configobj

import os, sys


sys.path.append(os.path.join(os.path.dirname(__file__), "lib/pazy-model"))
sys.path.append(os.path.join(os.path.dirname(__file__), "lib/sharpy"))

import sharpy.sharpy_main
from pazy_wing_model import PazyWing

import sharpy.utils.algebra as algebra


route_test_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

def set_simulation_settings_dynamic(case_name, output_folder, case_route, gust_settings, u_inf = 1, rho = 1.225, surface_m = 4, gust_vanes = False, alpha = np.deg2rad(5.), symmetry_condition = False, delft_model = False, test_case_settings = None, use_polars=False):
     # simulation settings
    config = configobj.ConfigObj()
    config.filename = case_route + '/{}.sharpy'.format(case_name)
    settings = dict()
    gravity = True
    n_step = 5
    structural_relaxation_factor = 0.6
    relaxation_factor = 0.2
    tolerance = 1e-6 #
    fsi_tolerance = 1e-4 #
    num_cores = 4
    variable_wake =  False 

    u_inf = 18.3
    rho = 1.205

    chord = 0.1
    CFL = 1
    dt = CFL * chord / surface_m / u_inf
    
    n_tstep = 10000
    alpha = np.deg2rad(5.)
    gust_frequency = 5.7 # Hz
    gust_T = 0.175439 #s 
    gust_amplitude = 0.805562

    if test_case_settings is not None:
        print("TEST CASE:", test_case_settings)
        alpha = test_case_settings['alpha'] 
        gust_T = test_case_settings['gust_T']
        gust_frequency = test_case_settings['frequency_gust_vane']
        gust_amplitude = test_case_settings['gust_amplitude']

    

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
                 'DynamicCoupled',
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
            'num_cores': num_cores,
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
            'symmetry_plane': 2
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
            'horseshoe': False,
            'num_cores': num_cores,
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
            'symmetry_plane': 2
        },
        'structural_solver': 'NonLinearStatic',
        'structural_solver_settings': settings['NonLinearStatic']}

    settings['LiftDistribution'] = {'coefficients': True,
                                    'rho': 1.225}
    settings['BeamPlot'] = {}

    settings['AerogridPlot'] = {'include_rbm': 'off',
                                'include_applied_forces': 'on',
                                'minus_m_star': 0}
    settings['SaveData'] = {'save_aero': True,
                            'save_struct': True,
                            'save_linear': True,}


    settings['Modal'] = {'NumLambda': 40,
                        'rigid_body_modes': 'off',
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
                            'convection_scheme': 2,
                            'gamma_dot_filtering': 7,
                            'cfl1': bool(not variable_wake),
                            # 'velocity_field_generator': 'SteadyVelocityField',
                            # 'velocity_field_input': {'u_inf':u_inf,
                            #                         'u_inf_direction': [1., 0, 0]},
                            'velocity_field_generator': 'GustVelocityField',
                            'velocity_field_input':{'u_inf': u_inf,
                                                    'u_inf_direction': [1., 0, 0],
                                                    'relative_motion': True,
                                                    'offset': 10 *dt * u_inf,    
                                                    'gust_shape': 'continuous_sin',
                                                    'gust_parameters': {'gust_length':  gust_T * u_inf, #gust_settings['length'],
                                                                        'gust_intensity': 2*gust_amplitude, # gust_settings['intensity']*u_inf,
                                                                        'gust_component': 1
                                                                    }                                                                
                                                },
                            'rho': rho,
                            'n_time_steps': n_tstep,
                            'dt': dt,
                            'symmetry_condition': symmetry_condition,
                            'symmetry_plane': 2
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
        'relaxation_factor': 0., #relaxation_factor,
        'minimum_steps': 1,
        'relaxation_steps': 150,
        'final_relaxation_factor': 0.0,
        'n_time_steps': n_tstep,
        'print_info': True,
        'dt': dt,
        'include_unsteady_force_contribution': True, 
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

    if gust_vanes:
        gust_vane_parameters = {
            'M': 8,
            'N':20, #, 
            'M_star': 100, 
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
                                                                       'vane_parameters': [gust_vane_parameters, gust_vane_parameters],
                                                                       'vertical': delft_model
                                                                      }
    
        settings['StepUvlm']['convection_scheme'] = 3
        settings['StepUvlm']['velocity_field_generator'] = 'SteadyVelocityField'
        settings['StepUvlm']['velocity_field_input'] = {'u_inf':u_inf,
                                                        'u_inf_direction': [1., 0, 0]}
                                                                    

    # settings['StaticCoupled']['correct_forces_method'] = 'EfficiencyCorrection'
    # settings['DynamicCoupled']['correct_forces_method'] = 'EfficiencyCorrection'
    if use_polars:
        print("Settings for PolarCorrection!")
        settings['StaticCoupled']['correct_forces_method'] = 'PolarCorrection'
        settings['StaticCoupled']['correct_forces_settings'] = {'cd_from_cl': 'off',
                                                                'correct_lift': 'on',
                                                                'moment_from_polar': 'on',
                                                                # 'skip_surfaces':[2,3],
                                                                'aoa_cl0': [0., 0.],
                                                                # 'write_induced_aoa': True,
                                                                }
                                                                

        settings['DynamicCoupled']['correct_forces_method'] = 'PolarCorrection'
        settings['DynamicCoupled']['correct_forces_settings'] = {'cd_from_cl': 'off',
                                                                'correct_lift': 'on',
                                                                'moment_from_polar': 'on',
                                                                # 'skip_surfaces':[2,3],
                                                                'aoa_cl0': [0., 0.],
                                                                # 'write_induced_aoa': True,
                                                                }
    for k, v in settings.items():
        config[k] = v

    config.write()
    return settings

def setup_pazy_model(case_name, case_route, pazy_settings, gust_vanes = False, symmetry_condition = False, polars=None):
    pazy = PazyWing(case_name, case_route, pazy_settings)
    pazy.generate_structure()
    if not symmetry_condition:
        pazy.structure.mirror_wing()
    if pazy_settings['model_id'] == 'delft':
        pazy.structure.rotate_wing()
    tip_load = 0# 0.285
    if tip_load > 0.:
        mid_chord_b = (pazy.get_ea_reference_line() - 0.5) * 0.1
        pazy.structure.add_lumped_mass((tip_load, pazy.structure.n_node//2, np.zeros((3, 3)),
                                np.array([0, mid_chord_b, 0])))
        if not symmetry_condition:
            pazy.structure.add_lumped_mass((tip_load, pazy.structure.n_node//2 + 1, np.zeros((3, 3)),
                                            np.array([0, mid_chord_b, 0])))


    pazy.generate_aero(polars=polars) 
    pazy.save_files()
    return pazy


def generate_polar_arrays(airfoils):
    # airfoils = {name: filename}
    # pazy model settings
    # Norberto:  M = 16, N = 64
    pazy_model_settings = {'skin_on': skin_on,
                        'discretisation_method': 'michigan',
                        'model_id': 'delft',
                        'num_elem': 2,
                        'surface_m': 16,
                        'symmetry_condition': symmetry_condition,
                        }
    case_name = 'pazy_vertical_case_{}_polars{:g}_dynamic_coarse_gust_vanes'.format(case, int(use_polars)) #_alpha_{:04g}'.format(100*test_case_settings['alpha']) #pazy_dynamic_alpha_587_gust_vane_test' #'pazy_modal_delft_aplha_12_symmetry_steady'
    case_route = case_root + '/' + case_name + '/'

    if not os.path.isdir(case_route):
        os.makedirs(case_route, exist_ok=True)

    if airfoil_polar is not None:

    else:
        polar_arrays = None
    setup_pazy_model(case_name, 
                     case_route, 
                     pazy_model_settings, 
                     symmetry_condition=symmetry_condition,
                     polars=polar_arrays)
    if airfoil_polar is not None:
        use_polars = True
    else:
    gust_settings= {'length': 5,
                    'offset': 0,
                    'intensity': 0.2,
                    }
    set_simulation_settings_dynamic(case_name, output_folder, case_route, gust_settings, pazy_model_settings['surface_m'], 
                                    gust_vanes = gust_vanes, 
                                    symmetry_condition = symmetry_condition, 
                                    delft_model = True, 
                                    test_case_settings = test_case_settings,
                                    use_polars=use_polars,)

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

    use_polars = False #True
    case = 1
    data = run_dynamic_prescriped_simulation_with_gust_input(skin_on='on',
                                                             case_root='./cases/',
                                                             output_folder='./output/',
                                                             gust_vanes = False, #True,
                                                             symmetry_condition = symmetry_condition,
                                                             test_case_settings = dict_test_cases[str(case)]
                                                             )
                                             

