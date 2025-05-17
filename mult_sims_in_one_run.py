import numpy as np
from main_AKF_w_sd_R import main_func_demo2
import matplotlib.pyplot as plt

# <editor-fold desc="txt name, monte carlo no, automatic or manual cov">
txt_file_name = "last JSONs - VWP"
monte_carlo_sim_amount = 1

# automatic_or_manual_cov_define = 'automatic'
automatic_or_manual_cov_define = 'manual'
# </editor-fold>


# <editor-fold desc="Mn manually defined value">
if automatic_or_manual_cov_define == 'manual':
    # Mns = np.array([
    #                     [[[0.01, 0.0005],[0.0005, 0.01]],500,1],  # [Mn, N_train, no_real]
    #                     [[[0.1, 0.005], [0.005, 0.1]], 500, 1],
    #                     [[[0.2, 0.005], [0.005, 0.2]], 500, 1],
    #
    #                     [[[0.01, 0.0005], [0.0005, 0.01]], 5000, 1],
    #                     [[[0.1, 0.005], [0.005, 0.1]], 5000, 1],
    #                     [[[0.2, 0.005], [0.005, 0.2]], 5000, 1],
    #
    #                     [[[0.01, 0.0005], [0.0005, 0.01]], 500, 10],
    #                     [[[0.1, 0.005], [0.005, 0.1]], 500, 10],
    #                     [[[0.2, 0.005],[0.005, 0.2]], 500, 10],
    #
    #                ])

    # Mns = np.array([
    #                     [[0.01, 0.0005], [0.0005, 0.01]],
    #                     [[0.1, 0.005], [0.005, 0.1]],
    #                     [[0.2, 0.05], [0.05, 0.2]],
    #                     [[0.5, 0.05], [0.05, 0.5]],
    #                     [[1, 0.1], [0.1, 1]],
    #                     [[2, 0.5], [0.5, 2]]
    #                 ])
    #
    Mns = np.array([
                        # [[0.01, 0], [0, 0.01]],
                        [[0.1, 0], [0, 0.1]],
                        [[0.2, 0], [0, 0.2]],
                        # [[0.5, 0], [0, 0.5]],
                        [[1, 0], [0, 1]],
                        [[2, 0], [0, 2]]
                    ])

# </editor-fold>

# <editor-fold desc="automatic cov values">
# Mn_diag = [0.001, 0.01, 0.1, 0.2, 0.5]
# Mn_off_diag = [0.0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4]

Mn_diag = [0.01, 0.1, 0.2, 0.5]
Mn_off_diag = [0.005, 0.01, 0.05, 0.1]
# </editor-fold>

# <editor-fold desc="Automatically define cov matrices">
if automatic_or_manual_cov_define == 'automatic':
    print('No of diags:', len(Mn_diag), ',   No of off diags:', len(Mn_off_diag))
    no_Mn_to_define = 0
    how_many_times_off_diag_greater_than_diag = 0
    for i in Mn_diag:
        for j in Mn_off_diag:
            if j<i:
                how_many_times_off_diag_greater_than_diag = 0
                no_Mn_to_define += 1
            elif j > i and how_many_times_off_diag_greater_than_diag == 0:
                no_Mn_to_define += 1
                how_many_times_off_diag_greater_than_diag += 1

    no_Mn_to_define += 1 #for the last one's (a-a/10)
    print('No of cov matrices: ', no_Mn_to_define, '\n')

    Mns = np.zeros((no_Mn_to_define,2,2))
    Mns_counter = 0
    how_many_times_off_diag_greater_than_diag = 0
    for i2 in Mn_diag:
        for j2 in Mn_off_diag:
            if j2 < i2:
                how_many_times_off_diag_greater_than_diag = 0
                Mns[Mns_counter] = np.array([[i2,j2],[j2,i2]])
                Mns_counter += 1
            elif j2 > i2 and how_many_times_off_diag_greater_than_diag == 0:
                Mns[Mns_counter] = np.array([[i2, i2-i2/10], [i2-i2/10, i2]])
                Mns_counter += 1
                how_many_times_off_diag_greater_than_diag += 1

    Mns[Mns_counter] = np.array([[i2, i2-i2/10], [i2-i2/10, i2]]) #for the last one's a-a/10
# </editor-fold>
a = len(Mns[0])
if len(Mns[0])==2:
    Mns = np.round(Mns, decimals=4)
np.set_printoptions(suppress=True)
print('Input cov matrices to be used:\n', Mns)
print('\nNo of cov matrices: ', len(Mns),
      '\nNo of monte-carlo for each cov:',monte_carlo_sim_amount,
      '\nTotal simulations:',( len(Mns) * monte_carlo_sim_amount))


# <editor-fold desc="Running the code over all Mns and printing Json files">
json_file_names = ''
no_of_json_files = 0
for i in range(len(Mns)): #run over all cov matrices

    for monte_carlo_sim_no in range(monte_carlo_sim_amount): #run over all monte carlo samples
        print('\n\n\n************************************'
              '\nNow started working with the Mn no:', str(i+1), '\nmonte-carlo sim no:', str(monte_carlo_sim_no+1), '\n', Mns[i],
              '\n************************************\n\n\n')
        new_json_name = main_func_demo2(Mns[i],  monte_carlo_sim_no+1, None, None, None)
        json_file_names = json_file_names + '\n' + new_json_name
        no_of_json_files += 1
        if (i < len(Mns)-1) or (monte_carlo_sim_no < monte_carlo_sim_amount):
            with open(txt_file_name, "w") as text_file:
                text_file.write(json_file_names)
            print('\n\n\n----------------------------------------------------------------------------------------------------'
                  '\n( for, Mn no:',str(i+1),', monte-carlo no:', str(monte_carlo_sim_no+1), ')',
                  '\nSaved', no_of_json_files, 'Json files until now:', json_file_names,
                  '\n----------------------------------------------------------------------------------------------------')

with open("saved json file names.txt", "w") as text_file:
    text_file.write(json_file_names)
print('\n\n\nComplete !!! --->Saved', no_of_json_files, 'Json files:', json_file_names)
print('for', len(Mns), 'Mns and', monte_carlo_sim_amount, 'monte-carlos')
print('\n\n\nComplete !!!')
# </editor-fold>


plt.plot(np.linspace(1, 10), np.linspace(1, 10))
plt.show()