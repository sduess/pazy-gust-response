import numpy as np
import os, sys
import h5py
import matplotlib.pyplot as plt

DIRECTORY = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + '/')

def postprocess_convergence_study(case_name, output_folder, dict_simulation_parameters):
    list_induced_velocities = []
    list_coordinates_surfaces = []
    max_ts = 300
    num_evaluation_points = 51
    num_cases = len(dict_simulation_parameters['list_spanwise_nodes_vanes']) \
        * len(dict_simulation_parameters['list_surface_m_vanes']) \
        * len(dict_simulation_parameters['list_wake_length_vanes'])
    induced_vel_point = np.zeros((num_evaluation_points, num_cases, 3, max_ts # possible cases dim, ts, 
        ))
    list_labels = []
    counter = 0

    idx_reference_case =None
    for n in dict_simulation_parameters['list_spanwise_nodes_vanes']:
        for m in dict_simulation_parameters['list_surface_m_vanes']:
            for wake_length in dict_simulation_parameters['list_wake_length_vanes']:
                if idx_reference_case is None:
                    if (m == dict_simulation_parameters['reference_solution_m_mstar_n'][0]) and\
                    (wake_length ==dict_simulation_parameters['reference_solution_m_mstar_n'][1]) and \
                    (n ==dict_simulation_parameters['reference_solution_m_mstar_n'][2]):
                        idx_reference_case = counter
                list_labels.append('N{}, M{}, M*{}'.format(n, m, wake_length))
                case_name_parametric = case_name + '_vanes_N%i_M%i_Mstar%i_'%(n, m, wake_length)
                output_folder_parametric = os.path.join(output_folder, case_name_parametric, case_name_parametric, 'WriteVariablesTime')
                for ipoint, file in enumerate(os.listdir(output_folder_parametric)):
                    if ipoint < num_evaluation_points:
                        data = np.loadtxt(os.path.join(output_folder_parametric,file), skiprows=1)
                        # print("ipoint = ", ipoint)
                        # print(data[1:10, 1:])
                        induced_vel_point[ipoint, counter, :,:] = np.transpose(data[:max_ts,1:])

                counter += 1

    # Plot results
    plot_results(num_evaluation_points, num_cases, induced_vel_point, list_labels)
    # Plot errors
    error_induced_vel_point = induced_vel_point.copy()
    for icase in range(num_cases):
        error_induced_vel_point[:,icase,:,:] = induced_vel_point[:,icase,:,:] - induced_vel_point[:,idx_reference_case,:,:]
    plot_results(num_evaluation_points, num_cases,error_induced_vel_point, list_labels, error = True)

def plot_results(num_evaluation_points, num_cases, induced_vel_point, list_labels, error = False):
    list_dim_str = ['x', 'y', 'z']
    if error:
        title_start = 'error u_ind '
    else:
        title_start = 'u_ind '
    for idim in range(3):
        for ipoint in range(num_evaluation_points):
            title = title_start + list_dim_str[idim] + ", point " + str(ipoint)
            plt.figure()
            plt.title(title)
            for icase in range(num_cases):
                plt.plot(induced_vel_point[ipoint, icase, idim, :], label = list_labels[icase])
            plt.grid()
            plt.legend()
            plt.savefig(title)
            plt.close()
            # plt.show()


def main():
    case_name = 'pazy_convergence_wake_test_' #'pazy_convergence_wake_test_'
    output_folder = './output/'
    dict_simulation_parameters = { 
        'list_surface_m_vanes': [8], #, 8, 16, 32, 64],
        'list_wake_length_vanes': [2,3,4], #5, 10, 15, 20],
        'list_spanwise_nodes_vanes': [10], #, 20]} #, 40, 80]}
        'reference_solution_m_mstar_n': [8, 4, 10]
    }
    postprocess_convergence_study(case_name, output_folder, dict_simulation_parameters)


if __name__ == main():
    main()