import numpy as np
import os, sys
import h5py as h5
import matplotlib.pyplot as plt
import scipy.signal as scipy_signal


output_folder = '../lib/sharpy/output/'
pazy_half_span = 0.55 #5.49843728e-01 # TODO get value
list_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
list_linestyles = ['solid', 'dashed', 'dotted']

def get_mean(zeta):
    n_nodes =  np.shape(zeta)[-1]
    wing_deformation_mean = np.zeros((n_nodes,3))
    for inode in range(n_nodes):
        for idim in range(3):
            wing_deformation_mean[inode, idim] = np.mean(zeta[idim, :, inode])
    np.savetxt("./wing_deformation_mean.csv", wing_deformation_mean)
    return wing_deformation_mean

def steady_results():
    list_alpha = [5, 10] #,10]
    # list_alpha = [6,11]
    
    n_nodes = 60
    list_wing_deformation = []
    list_tip_load = ['0 kg', '0.285 kg', '0.5 kg']
    list_rho = ['1.225 kg/m3', '1.0 kg/m3']
    counter_tip_load = 0
    legend_tip_load = ['']
    # for tip_load in ['0']:
    for rho in ['0']:
        for counter, alpha in enumerate(list_alpha):
            # route_case = 'pazy_modal_delft_aplha_{}_symmetry_steady'.format(alpha)
            # route_case = 'pazy_modal_delft_aplha_{}_rotate_horseshoe_symmetry'.format(alpha)
            route_case = 'pazy_vertical_case_{}_no_free_wake'.format(counter + 1)#pazy_steady_alpha_{}_uinf_18'.format(alpha)
            if rho != '0':
                route_case += '_rho_' + rho
            else:            
                ref_span, ref_deformation = get_reference_deformation(counter + 1)
                print("reference tip deformation = ", ref_deformation[-1])
                plt.scatter(ref_span, ref_deformation, color = list_colors[counter], marker='d' ,label = r'Experiments - $\alpha$ = {} deg'.format(alpha))
            
            file = os.path.join(output_folder, route_case, route_case, 'savedata', route_case + '.data.h5')
            print(file)
            wing_deformation, _ = read_structural_deformation(0, file, 0)
            print(np.shape(wing_deformation))
            # wing_deformation /= pazy_half_span 
            wing_deformation = wing_deformation[:n_nodes, :] / pazy_half_span
            
            label =  r'SHARPy - $\alpha$ = {} deg'.format(alpha)
            plt.plot(np.abs(wing_deformation[:,2]), np.abs(wing_deformation[:,1]), linestyle=list_linestyles[counter_tip_load], color = list_colors[counter],label =label)
            get_error(ref_span, ref_deformation, wing_deformation)
            # wing_deformation = read_structural_deformation(0, file, 1)/pazy_half_span
            # wing_deformation /= pazy_half_span #np.max(wing_deformation[:,1])
            
            # label =  'alpha = {} deg, isurf 1'.format(alpha, list_rho[counter_tip_load])

            # plt.plot(np.abs(wing_deformation[:,2]), np.abs(wing_deformation[:,1]), linestyle=list_linestyles[counter_tip_load+1], color = list_colors[counter],label =label)
            # np.savetxt('./wing_deformation_{}.csv'.format(alpha), wing_deformation)
            # counter += 1
        counter_tip_load += 1
    # read_reference_deformation(plt)
    plt.grid()
    plt.ylabel('y/s')
    plt.xlabel('z/s')
    plt.legend()
    plt.savefig('wing_deformation_vertical_uinf_18.png')
    plt.show()

def dynamic_results():
    list_alpha = [5, 10]  
    for counter, alpha in enumerate(list_alpha):          
        route_case = 'pazy_vertical_case_1_gust_comp_1' # 'pazy_modal_delft_aplha_{}_symmetry_dynamic_gust'.format(alpha)

        file = os.path.join(output_folder, route_case, route_case, 'savedata', route_case + '.data.h5')
        wing_deformation, spanwise_node = read_structural_deformation(0, file, 0, steady=False)#/pazy_half_span
        # wing_deformation /= pazy_half_span #np.max(wing_deformation[:,1])
        
        label =  r'$\alpha$ = {} deg'.format(alpha)
        plt.plot(list(range(len(wing_deformation[:,spanwise_node,1]))), wing_deformation[:,spanwise_node,1]-wing_deformation[0,spanwise_node,1], linestyle=list_linestyles[0], color = list_colors[counter],label =label + ' - z/s = 0.9')
        extract_period_from_unsteady_data(-wing_deformation[:,spanwise_node,1], counter+1)
        spanwise_node = 30

        plt.plot(list(range(len(wing_deformation[:,spanwise_node,1]))), wing_deformation[:,spanwise_node,1]-wing_deformation[0,spanwise_node,1], linestyle=list_linestyles[1], color = list_colors[counter],label =label+ ' - z/s = 0.5')
        print("alpha = ", alpha, ", max tip deflection = ", np.max(wing_deformation[:,-1,1]), ", min tip deflection = ", np.min(wing_deformation[:,-1,1]))

    plt.grid()
    plt.ylabel(r'$y/s')
    plt.xlabel('ts')
    plt.legend()
    plt.savefig('wing_deformation_dynamic_alpha_10.png')
    plt.show()

def extract_period_from_unsteady_data(array, test_case):
    # find local extrema
    local_maxima = scipy_signal.argrelextrema(array, np.greater)[0]
    local_minima = scipy_signal.argrelextrema(array, np.less)[0] # not neded to identify a period

    # Identify one period of oscillation
    # option 1a: use last oscillation (for now it should be enough)
    # option 1b: use one of the oscillations with the most frequent occuring local maxima
    period_of_oscillation = array[local_maxima[-3]:local_maxima[-2]]
    # Check if peaks are the same?


    # Get mean of oscillation (Check if this is the start of Christoph's period) (Yes he does)
    period_of_oscillation -= np.mean(period_of_oscillation)
    period_of_oscillation *= 1000 # convert rom m to mm
    idx_zero = find_index_of_closest_entry(period_of_oscillation, 0.)
    period_of_oscillation = np.roll(period_of_oscillation, idx_zero)  # roll to start with zero
    plt.figure()
    plt.title('Test case #{}'.format(test_case))
    plt.text(0.6, max(period_of_oscillation)*2/3, 'max y = {:.2f} mm\n min y = {:.2f} mm'.format(array[local_maxima[-2]], 100*array[local_minima[-3]]))
    plt.plot(np.arange(0,1,1./len(period_of_oscillation)),period_of_oscillation)
    plt.xlabel('t/T')
    plt.ylabel(r'$\Delta y_{tip}$, mm')
    plt.xlim([0,1])
    plt.show()
   


def get_error(ref_span, ref_data, numerical_data):
    print(len(ref_span))
    error_array = np.zeros((len(ref_span),1))
    for ipos in range(len(ref_span)):
        deformation_interp = np.interp(ref_span[ipos], np.abs(numerical_data[:,2]), np.abs(numerical_data[:,1]))
        error_array[ipos,0] = (deformation_interp - ref_data[ipos])/ref_data[ipos]*100
    print("error = ", error_array)
    return error_array
   
def get_reference_deformation(icase):
    file = '../CM_static_dataset/Static_Dataset/Deflect_Case{}_mm.txt'.format(icase)
    data_deformation = np.loadtxt(file)/pazy_half_span/1000
    file = '../CM_static_dataset/Static_Dataset/Deflect_span_Case{}_mm.txt'.format(icase)
    data_span = np.loadtxt(file)#/pazy_half_span/1000
    return data_span, data_deformation


def find_index_of_closest_entry(array_values, target_value):
    return np.argmin(np.abs(array_values - target_value))


def read_structural_deformation(timestep, file, i_surf,steady=True, spanwise_node_position_z_over_s = 1.):
    ts_str = '00000'
    print(file)
    with h5.File(file, "r") as f:     
        if steady:   
            wing_deformation = np.array(f['data']['structure']['timestep_info'][ts_str]['pos'])

            # wing_deformation = get_mean(np.array(f['data']['aero']['timestep_info'][ts_str]['zeta']['_as_array'][i_surf, :,:,:]))
            spanwise_node = None
        else:
            ts_max = len(f['data']['structure']['timestep_info'].keys())-2
            n_nodes = len( np.array(f['data']['structure']['timestep_info'][ts_str]['pos'])[:,0]) 
            print("n nodes = ", n_nodes)
            spanwise_positions = np.array(f['data']['structure']['timestep_info'][ts_str]['pos'])[2]
            spanwise_node = 53 #find_index_of_closest_entry(spanwise_positions/pazy_half_span, spanwise_node_position_z_over_s)
            fill_str = '0000'
            list_time_steps_str = [fill_str[len(str(ts))-1:] + str(ts)  for ts in range(1,ts_max + 1)]
            wing_deformation = np.zeros((ts_max,n_nodes,3))
            for its, ts_str in enumerate(list_time_steps_str):
                wing_deformation[its, :, :] =np.array(f['data']['structure']['timestep_info'][ts_str]['pos'])

    return wing_deformation, spanwise_node

if __name__ == '__main__':
    steady_results()
    dynamic_results()