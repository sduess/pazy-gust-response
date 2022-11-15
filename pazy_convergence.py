import numpy as np
import os, sys
from helper_functions.settings_generator import SettingsGenerator
from helper_functions.generate_pazy_model import setup_pazy_model
sys.path.append(os.path.join(os.path.dirname(__file__), "lib/sharpy"))

import sharpy.sharpy_main
import sharpy.utils.algebra as algebra

'''
    Script to automatical run a convergence study for the Pazy wing (TODO adjust later to apply it on any aircraft model)
    and also for the gust vanes. The process chain looks as follows: 
    A) Pazy Wing
        1. Check structural convergence by varying spanwise structural nodes (n_elem_mulitplier) and check convergence of first modal frequencies.
        2. Check chordwise discretisation by increasing M panels with horseshoe wake.
        3. Check wake stremawise discretisation by increasing wake length.
    B) Gust Vanes
        1. Start with reasonable spanwise discretisation,
'''


def structural_convergence(case_name, case_route, output_folder, simulation_settings):
    # A1) Structural convergence
     #*** Here, trust Norberto's Michigan discretisation for now. Implement for FLEXOP e.g.
    flow = ['BeamLoader', 'Modal']
    settings = SettingsGenerator()
    #update simulation settings
    settings.generate_settings(case_name, case_route, output_folder, flow, **simulation_settings)
    # for loop over N nodes/ elem multiplier

def chordwise_aero_convergence(case_name, case_route, output_folder, simulation_settings):
    # TODO replace SaveData by writeVariablesTime
    flow = ['BeamLoader', 'AerogridLoader', 'StaticCoupled', 'SaveData']
    settings = SettingsGenerator()
    #update simulation settings
    settings.generate_settings(case_name, case_route, output_folder, flow, **simulation_settings)
    # update simulation settings (e.g. horseshoe on!!)
    list_surface_m = [2, 4, 8, 16, 32, 64]

def wake_length_convergence(case_name, case_route, output_folder, simulation_settings):
    list_wake_length = [5, 10, 15, 20]
    # first run with horseshoe on
    #surface m from before!
    flow = ['BeamLoader', 'AerogridLoader', 'StaticCoupled', 'SaveData']
    settings = SettingsGenerator()
    #update simulation settings
    settings.generate_settings(case_name, case_route, output_folder, flow, **simulation_settings)

def gust_vane_convergence(case_name, case_route, output_folder, simulation_settings):
    # Pazy parameters (TODO: change to global variables or input json file)
    pazy_wing_span = 0.549843728
    pazy_chord = 0.1
    ea_main = 0.4410

    #settings
    settings = SettingsGenerator()
    surface_m = 4 # coarse for now
    dt = pazy_chord / surface_m / simulation_settings['u_inf'] # Check influence!
    pazy_model_settings = {'skin_on': True,
                    'discretisation_method': 'michigan',
                    'symmetry_condition': False,
                    'model_id': 'pazy',
                    'num_elem': 2,
                    'surface_m': surface_m}

    simulation_settings['symmetry_condition'] = pazy_model_settings['symmetry_condition']
    # simulation_settings['dynamic'] = True
    simulation_settings['n_tstep'] = 2000
    simulation_settings['wake_length'] = 5
    simulation_settings['surface_m'] = surface_m
    simulation_settings['dt'] = dt

    flow = ['BeamLoader', 'AerogridLoader', 'StaticCoupled', 'DynamicCoupled', 'SaveData']
    # flow = ['BeamLoader', 'AerogridLoader', 'DynamicCoupled']
    # flow = ['BeamLoader', 'AerogridLoader', 'AerogridPlot'] #, 'DynamicCoupled']

    # Setup write variables time
    velocity_field_points = [ pazy_chord * (1 - ea_main) / 2., 0., 0., #first six distributed on the wing
                            0., 1*pazy_wing_span/5., 0.,
                            pazy_chord * (1 - ea_main) / 2., 2*pazy_wing_span/5., 0.,
                            pazy_chord * (1 - ea_main) * 2 / 3., 3*pazy_wing_span/5., 0.,
                            pazy_chord * (1 - ea_main) / 2., 4*pazy_wing_span/5., 0.,
                            0., 5*pazy_wing_span/5., 0.,
                            - pazy_chord *ea_main, 0., 0., # y-axis LE, and subsequent points in front of LE
                            - pazy_chord *(ea_main + 1./5.), 0., 0.,
                            - pazy_chord *(ea_main + 2./5.), 0., 0.,
                            - pazy_chord *(ea_main + 3./5.), 0., 0.,
                            - pazy_chord *(ea_main + 4./5.), 0., 0.,
                            - pazy_chord *(ea_main + 5./5.), 0., 0.,
                            - pazy_chord *(ea_main + 6./5.), 0., 0.,
                            - pazy_chord *(ea_main + 7./5.), 0., 0.,
                            - pazy_chord *(ea_main + 8./5.), 0., 0.,
                            - pazy_chord *(ea_main + 9./5.), 0., 0.,
                            - pazy_chord *(ea_main + 10./5.), 0., 0.,
                            - pazy_chord *(ea_main + 15./5.), 0., 0.,
                            - pazy_chord *(ea_main + 20./5.), 0., 0.,
                            - pazy_chord *(ea_main + 25./5.), 0., 0.,
                            - pazy_chord *(ea_main + 30./5.), 0., 0.,
                            - pazy_chord *(ea_main + 35./5.), 0., 0.,
                            - pazy_chord *(ea_main + 40./5.), 0., 0.,
                            - pazy_chord *(ea_main + 1./5.), 1* pazy_wing_span/5., 0., # spanwise shortly in front
                            - pazy_chord *(ea_main + 1./5.), 2* pazy_wing_span/5., 0., 
                            - pazy_chord *(ea_main + 1./5.), 3* pazy_wing_span/5., 0., 
                            - pazy_chord *(ea_main + 1./5.), 4* pazy_wing_span/5., 0., 
                            - pazy_chord *(ea_main + 1./5.), 5* pazy_wing_span/5., 0., 
                            - pazy_chord *(ea_main + 1./5.), 6* pazy_wing_span/5., 0., 
                            pazy_chord *(1-ea_main), 0., 0., # y-axis LE, and subsequent points in front of TE
                            pazy_chord *(1-ea_main + 1./5.), 0., 0.,
                            pazy_chord *(1-ea_main + 2./5.), 0., 0.,
                            pazy_chord *(1-ea_main + 3./5.), 0., 0.,
                            pazy_chord *(1-ea_main + 4./5.), 0., 0.,
                            pazy_chord *(1-ea_main + 5./5.), 0., 0.,
                            pazy_chord *(1-ea_main + 6./5.), 0., 0.,
                            pazy_chord *(1-ea_main + 7./5.), 0., 0.,
                            pazy_chord *(1-ea_main + 8./5.), 0., 0.,
                            pazy_chord *(1-ea_main + 9./5.), 0., 0.,
                            pazy_chord *(1-ea_main + 10./5.), 0., 0.,
                            pazy_chord *(1-ea_main + 15./5.), 0., 0.,
                            pazy_chord *(1-ea_main + 20./5.), 0., 0.,
                            pazy_chord *(1-ea_main + 25./5.), 0., 0.,
                            pazy_chord *(1-ea_main + 30./5.), 0., 0.,
                            pazy_chord *(1-ea_main + 35./5.), 0., 0.,
                            pazy_chord *(1-ea_main + 40./5.), 0., 0.,
                            pazy_chord *(1-ea_main + 1./5.), 1* pazy_wing_span/5., 0., # spanwise shortlyaft Te
                            pazy_chord *(1-ea_main + 1./5.), 2* pazy_wing_span/5., 0., 
                            pazy_chord *(1-ea_main + 1./5.), 3* pazy_wing_span/5., 0., 
                            pazy_chord *(1-ea_main + 1./5.), 4* pazy_wing_span/5., 0., 
                            pazy_chord *(1-ea_main + 1./5.), 5* pazy_wing_span/5., 0., 
                            pazy_chord *(1-ea_main + 1./5.), 6* pazy_wing_span/5., 0., 
                            ]


    simulation_settings['write_variables_time_settings'] = {'cleanup_old_solution': True,
                                                            'vel_field_variables': ['uind'],
                                                            'vel_field_points': velocity_field_points
                                                            }
    # Setup gust vanes
    simulation_settings['gust_vanes'] = True
    simulation_settings['n_vanes']= 2
    simulation_settings['streamwise_position'] = [-1.50, -1.50]
    simulation_settings['vertical_position']= [-0.25, 0.25]
                                    
    


    # define gust vane deflection
    gust_settings = {'amplitude': np.deg2rad(5.),
                     'frequency': 5.7,
                     'mean': 0.}

    cs_deflection_file = '/home/sduess/Documents/Aircraft Models/Pazy/pazy-gust-response/02_gust_vanes/cs_deflection_amplitude_{}_frequency_{}_mean_{}.csv'.format(gust_settings['amplitude'],  gust_settings['frequency'],  gust_settings['mean'])
    write_deflection_file(simulation_settings['n_tstep'], dt, gust_settings['amplitude'],  gust_settings['frequency'],  gust_settings['mean'])
    gust_vane_parameters = {'M': 8,
                            'N':20, 
                            'M_star': 20, 
                            'span': 5,  
                            'chord': 0.3, 
                            'control_surface_deflection_generator_settings': {'dt': dt, 
                                 'deflection_file': cs_deflection_file, 
                                }
                            }
    list_surface_m_vanes = [12] #, 16,32]
    list_wake_length_vanes = [3] #2,3,4] 
    list_spanwise_nodes_vanes = [10] 
    for n in list_spanwise_nodes_vanes:
        gust_vane_parameters['N'] = n
        for m in list_surface_m_vanes:
            gust_vane_parameters['M'] = m
            for wake_length in list_wake_length_vanes: # in m
                gust_vane_parameters['M_star'] = int(wake_length/(dt * simulation_settings['u_inf'])) # m * wake_length
                simulation_settings['gust_vane_parameters'] = [gust_vane_parameters, gust_vane_parameters]
                case_name_parametric = case_name + '_vanes_N%i_M%i_Mstar%i_'%(n, m, wake_length)

                settings.generate_settings(case_name_parametric, case_route, output_folder, flow, simulation_settings, gust_vanes = True, write_variables_time = True)
                setup_pazy_model(case_name_parametric, case_route, pazy_model_settings, 
                                symmetry_condition = simulation_settings['symmetry_condition']) # TODO: rename files plus parameters??
                sharpy.sharpy_main.main(['', case_route + case_name_parametric + '.sharpy'])


def write_deflection_file(n_tstep, dt, amplitude, frequency, mean):
    cs_deflection_file = '/home/sduess/Documents/Aircraft Models/Pazy/pazy-gust-response/02_gust_vanes/cs_deflection_amplitude_{}_frequency_{}_mean_{}.csv'.format(amplitude, frequency, mean)
    time = np.linspace(0., n_tstep * dt, n_tstep)
    cs_deflection_prescribed = float(amplitude) * np.sin(2 * np.pi * float(frequency) * time)
    np.savetxt(cs_deflection_file, cs_deflection_prescribed)

def main():

    case_name = 'pazy_convergence_wake_test_coarse_'
    case_route = './cases/'
    output_folder = './output/'
    
    general_settings_parameter = {
        'num_cores': 2,
        'dynamic': True,
        'static_trim': False,
        'symmetry_condition': False,
        'u_inf': 18.3,
        'rho': 1.225, # TODO: Check christoph value
        'alpha': np.deg2rad(5.),
        }        
    gust_vane_convergence(case_name, case_route, output_folder, general_settings_parameter)
    # structural_convergence(case_name, case_route, output_folder, general_settings_parameter)


if __name__ == '__main__':
    main()
       

