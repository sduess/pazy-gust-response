import numpy as np
import sharpy.utils.algebra as algebra
import configobj

class SettingsGenerator():
    def __init__(self):
        self.settings = dict() 

    def write_settings(self, case_name, case_route, flow):
        
        config = configobj.ConfigObj()
        config.filename = case_route + '/%s.sharpy'%(case_name)
        print("file name .sharpy file = ", config.filename)
        for k, v in self.settings.items():
            print(k)
            config[k] = v
        config.write()

    def generate_settings(self, case_name, case_route, output_folder, flow, settings_input, gust_vanes = False, write_variables_time = False):
        self.set_defaul_settings(case_name, case_route, output_folder,**settings_input)
        # gust_vanes = kwargs.get('gust_vanes', False)
        self.settings['SHARPy']['flow'] = flow
        if gust_vanes:
            self.add_gust_vane_settings(**settings_input)
        if write_variables_time:
            self.add_write_variables_time_settings(**settings_input)
        self.write_settings(case_name, case_route, flow)

    def set_simulation_settings(self, **kwargs):
        self.num_cores = kwargs.get('num_cores', 2)
        self.static_trim = kwargs.get('static_trim', False)
        self.dynamic = kwargs.get('dynamic', False)
        self.gust_vanes = kwargs.get('gust_vanes', False)
        self.symmetry_condition = kwargs.get('symmetry_condition', False)
        self.n_tstep= kwargs.get('n_tstep', 1)    

    def set_flight_conditions(self, **kwargs):
        self.u_inf = kwargs.get('u_inf', 1.)
        self.rho = kwargs.get('rho', 1.225)
        self.gravity = kwargs.get('gravity', True)
        self.alpha =  kwargs.get('alpha', 0.)

    def set_grid_discretisation(self, **kwargs):
        self.wake_length =kwargs.get('wake_length',5)
        self.surface_m= kwargs.get('surface_m', 4)
        self.dt = kwargs.get('dt', 1)
        self.horseshoe = kwargs.get('horseshoe', False)
        
    def load_user_specified_settings(self,  **kwargs):
        self.set_simulation_settings(**kwargs)
        self.set_flight_conditions(**kwargs)
        self.set_grid_discretisation(**kwargs)

    def set_defaul_settings(self,case_name, case_route, output_folder, **kwargs):
        self.load_user_specified_settings(**kwargs)
        self.settings = dict()
        #TODO: solution for flow!!
        self.settings['SHARPy'] = {
            'flow': ['BeamLoader',
                    ],
            'case': case_name, 'route': case_route,
            'write_screen': 'on', 'write_log': 'on',
            'log_folder': output_folder + '/' + case_name + '/',
            'log_file': case_name + 'log'}


        self.settings['BeamLoader'] = {'unsteady': 'on',
                                'orientation': algebra.euler2quat(np.array([0.,
                                                                            self.alpha,
                                                                            0]))}

        self.settings['AerogridLoader'] = {
            'unsteady': 'on',
            'aligned_grid': 'on',
            'mstar': self.wake_length * self.surface_m,
            'wake_shape_generator': 'StraightWake',
            'wake_shape_generator_input': {'u_inf': self.u_inf,
                                        'dt': self.dt}}

        self.settings['NonLinearStatic'] = {'print_info': 'off',
                                    'max_iterations': 1000,
                                    'num_load_steps': 12,
                                    'delta_curved': 1e-5,
                                    'min_delta': 1e-6,
                                    'gravity_on': self.gravity,
                                    'relaxation_factor': 0.1,
                                    'gravity': 9.81}

        self.settings['StaticUvlm'] = {
                'rho': self.rho, 
                'print_info': True,
                'horseshoe': self.horseshoe,
                'num_cores': self.num_cores,
                'n_rollup':  self.settings['AerogridLoader']['mstar'], 
                'rollup_dt': self.dt,
                'rollup_aic_refresh': 1,
                'rollup_tolerance': 1e-4,
                'velocity_field_generator': 'SteadyVelocityField',
                'velocity_field_input': {
                    'u_inf': self.u_inf,
                    'u_inf_direction': [1.0, 0., 0.]},
                'vortex_radius': 1e-9,
                'symmetry_condition': self.symmetry_condition,
            }
        self.settings['StaticCoupled'] = {
            'print_info': 'on',
            'max_iter': 200,
            'n_load_steps': 8,
            'tolerance': 1e-5,
            'relaxation_factor': 0.2,
            'aero_solver': 'StaticUvlm',
            'aero_solver_settings':  {
                'rho': self.rho, 
                'print_info': True,
                'horseshoe': 'off',
                'num_cores': self.num_cores,
                'n_rollup': 0, 
                'rollup_dt': self.dt,
                'rollup_aic_refresh': 1,
                'rollup_tolerance': 1e-4,
                'velocity_field_generator': 'SteadyVelocityField',
                'velocity_field_input': {
                    'u_inf': self.u_inf,
                    'u_inf_direction': [1.0, 0., 0.]},
                'vortex_radius': 1e-9,
                'symmetry_condition': self.symmetry_condition,
            },
            'structural_solver': 'NonLinearStatic',
            'structural_solver_settings': self.settings['NonLinearStatic']}

        self.settings['BeamPlot'] = {}

        self.settings['AerogridPlot'] = {'include_rbm': 'off',
                                    'include_applied_forces': 'on',
                                    'minus_m_star': 0}
        self.settings['SaveData'] = {'save_aero': True,
                                'save_struct': True,}


        self.settings['Modal'] = {'NumLambda': 40,
                            'rigid_body_modes': 'on',
                            'print_matrices': 'off',
                            'continuous_eigenvalues': 'off',
                            'write_modes_vtk': 'off',
                            'use_undamped_modes': 'on'}
                 
        self.settings['StepUvlm'] = {'num_cores': self.num_cores,
                                'convection_scheme': 3,
                                'gamma_dot_filtering': 7,
                                'cfl1': True,
                                'velocity_field_generator': 'SteadyVelocityField',
                                'velocity_field_input': {'u_inf':self.u_inf,
                                                        'u_inf_direction': [1., 0, 0]},
                                'rho': self.rho,
                                'n_time_steps': self.n_tstep,
                                'dt': self.dt,
                                'symmetry_condition': self.symmetry_condition,
                                }

        self.settings['NonLinearDynamicPrescribedStep'] = {'print_info': 'on',
                    'max_iterations': 950,
                    'delta_curved': 1e-1,
                    'min_delta':1e-6,
                    'newmark_damp': 1e-4,
                    'gravity_on': self.gravity,
                    'gravity': 9.81,
                    'num_steps': self.n_tstep,
                    'dt': self.dt,
                    }
        
        self.settings['DynamicCoupled'] = {'structural_solver':'NonLinearDynamicPrescribedStep',
            'structural_solver_settings': self.settings['NonLinearDynamicPrescribedStep'],
            'aero_solver': 'StepUvlm',
            'aero_solver_settings': self.settings['StepUvlm'],
            'fsi_substeps': 200,
            'fsi_tolerance': 1e-4,
            'relaxation_factor': 0.2,
            'minimum_steps': 1,
            'relaxation_steps': 150,
            'final_relaxation_factor': 0.05,
            'n_time_steps': self.n_tstep,
            'print_info': True,
            'dt': self.dt,
            'include_unsteady_force_contribution': True, 
            'postprocessors': ['BeamLoads', 'BeamPlot', 'AerogridPlot', 'SaveData'],
            'postprocessors_settings': {
                                        'BeamLoads': {'csv_output': 'off'},
                                        'BeamPlot': {'include_rbm': 'on',
                                                    'include_applied_forces': 'on'},
                                        'AerogridPlot': {
                                            'include_rbm': 'on',
                                            'include_applied_forces': 'on',
                                            },
                                        'SaveData': self.settings['SaveData'],
                                        },
            }
    
    def add_gust_vane_settings(self,**kwargs):
        n_vanes =  kwargs.get('n_vanes', 1)
        streamwise_position = kwargs.get('streamwise_position', ([-1.5]))
        vertical_position = kwargs.get('vertical_position', ([-0.25]))

        gust_vane_parameters = kwargs.get('gust_vane_parameters',
                                          [{'M': 8,
                                           'N':20, 
                                           'M_star': 20, 
                                           'span': 5, #10., 
                                           'chord': 0.3, 
                                           'control_surface_deflection_generator_settings': {'dt': self.dt, 
                                                                                             'deflection_file': '/home/sduess/Documents/Aircraft Models/Pazy/pazy-gust-response/02_gust_vanes/cs_deflection_amplitude_0.08726646259971647_frequency_5.7_mean_0.0.csv' 
                                                }
                                            }]
                                         )
        
        # print("\n\n\n\ngust vane mstar = ", gust_vane_parameters['M_star'])
        self.settings['AerogridLoader']['gust_vanes'] = True
        self.settings['AerogridLoader']['gust_vanes_generator_settings'] = {'n_vanes': n_vanes,
                                                                       'streamwise_position': streamwise_position,
                                                                       'vertical_position': vertical_position,
                                                                       'symmetry_condition': self.symmetry_condition,
                                                                       'vane_parameters': gust_vane_parameters
                                                                      }
        self.settings['StepUvlm']['convection_scheme'] = 3
        self.settings['StepUvlm']['velocity_field_generator'] = 'SteadyVelocityField'
        self.settings['StepUvlm']['velocity_field_input'] = {'u_inf':self.u_inf,
                                                             'u_inf_direction': [1., 0, 0]}


        # if kwargs.get('export_induced_velocities', True):
        #     self.settings['DynamicCoupled']['postprocessors_settings']['AerogridPlot']['include_velocities'] = True
        #     self.settings['DynamicCoupled']['postprocessors_settings']['AerogridPlot']['save_induced_velocities'] = True

        self.settings['StepUvlm']['cfl1'] = True
        self.settings['StaticUvlm']['cfl1'] = True
        # self.settings['AerogridLoader']['wake_shape_generator_input']['ndx1'] = 80 #[80, 80, 16, 16]
        # self.settings['AerogridLoader']['wake_shape_generator_input']['r'] =  1. #[1., 1., 1.5, 1.5]
        # self.settings['AerogridLoader']['wake_shape_generator_input']['dxmax'] = 0.1/self.surface_m #[0.1/self.surface_m, 0.1/self.surface_m, 0.1/self.surface_m*2, 0.1/self.surface_m*2]

    def add_write_variables_time_settings(self, **kwargs):
        # if not ('WriteVariablesTime' in self.settings['SHARPy']['flow']):
        #     self.settings['SHARPy']['flow'].insert(-1, 'WriteVariablesTime')
        self.settings['DynamicCoupled']['postprocessors_settings']['WriteVariablesTime'] = kwargs.get('write_variables_time_settings', dict())
        self.settings['DynamicCoupled']['postprocessors'].append('WriteVariablesTime')
