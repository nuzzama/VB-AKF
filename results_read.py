import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
np.random.seed(1)

# <editor-fold desc="options">
plot_states = False
plot_states_error = False
plot_train_and_test_location_only = False
plot_cov = True
plot_cov_error = False
plot_state_estimation_RMSE_separately = False
plot_state_estimation_RMSE_subplot = True
plot_states_subplots = True


enforece_ylim_for_velo_states = False
y_lim_value_for_velo_states = 3


ylim_enforce_for_cov_sub_plots = False
ylim_low_cov_plots = -0.6
ylim_high_cov_plots = 1.2
ylim_for_cov_error_plots = 0.6



samples_of_meas_for_cov_calc = 100
# no_of_MC_samples_for_state_RMSE = 20
ylim_for_RMSE = None


do_KF_TMC_monte_carlo = True
do_KF_LMC_monte_carlo = True

do_KF_TMC_RMSE_calc_here = False

do_monte_carlo_here = False
# </editor-fold>





# <editor-fold desc="data files all">

file_name = 'savedir/r2-p5,TruCov=sample,newLocPrior V-0,Init V-2,mc-None,k-200,lr-0.01,i-20000.npz'

# file_name = 'savedir/r2-p007,TruCov=sample,newLocPrior V-0,Init V-2,mc-None,k-200,lr-0.01,i-20000.npz'

# </editor-fold>






# <editor-fold desc="data extractions">
preds = np.load(file_name)

x_states = preds['x_states']
x_true = preds['x_true']
G_mean, G_cov = preds['G_mean'], preds['G_cov']  # (N_test, D, nu)
N_test, D, nu = G_mean.shape
Ar_scale_diag = preds['Ar_scale_diag']
x0_mean = preds['x0_mean']
x0_cov = preds['x0_cov']
qV_mu = preds['qV_mu']
qV_sqrt = preds['qV_sqrt']
logP = preds['logP']
target_motion_type = preds['target_motion_type']
y_measured = preds['y_measured']

if 'no_of_cov_test_re_run' in preds:
    no_of_cov_test_re_run = preds['no_of_cov_test_re_run']
    print('no_of_cov_test_re_run:', no_of_cov_test_re_run)

if 'cov_test_re_run' in preds:
    cov_test_re_run = preds['cov_test_re_run']

if 'y1y2_stored' in preds:
    y1y2_stored = preds['y1y2_stored']

if 'add_Q_status' in preds:
    # add_Q_status = preds['add_Q_status']
    add_Q_status = True #for some reason, this comes as a array instead of a boolean varaible. which cannot be accesed for if condition. for this reason, this is a temporary fix, as I know thsi is True for all results.
else:
    add_Q_status = False

if 'Q_value' in preds:
    Q_value = preds['Q_value']
# else:
#     Q_multiplier = 0.01
#     # Q_value = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,0],[0,0,0,0]]) * Q_multiplier
#     # Q_value = np.eye(4)*Q_multiplier
#     Q_value = np.array([[Q_multiplier ** 2, 0, 0, 0], [0, Q_multiplier ** 2, 0, 0], [0, 0, Q_multiplier, 0], [0, 0, 0, Q_multiplier]])

if 'x_test' in preds:
    x_test = preds['x_test']
else:
    x_test = x_true[1:,:].copy()


if 'cov_test_location_type' in preds:
    cov_test_location_type = preds['cov_test_location_type']
else:
    cov_test_location_type = 'on x_true'



if 'cov_true' in preds:
    cov_true = preds['cov_true']

if 'cov_predicted' in preds:
    cov_predicted = preds['cov_predicted']
else:
    n_samples = 1000
    G_samps = np.random.randn(n_samples, N_test, D, nu) * (G_cov ** 0.5) + G_mean  # (n_samples, N_test, D, nu)
    ArG = Ar_scale_diag * G_samps  # (n_samples, N_test, D, nu)
    cov_predicted_n_samps = np.matmul(ArG, np.transpose(ArG, [0, 1, 3, 2]))  # (n_samples, N_test, D, D)
    cov_predicted = np.average(cov_predicted_n_samps, axis=0)


if 'x_state_KF_LMC' in preds:
    x_state_KF_LMC = preds['x_state_KF_LMC']

if 'P_KFLMC' in preds:
    P_KFLMC = preds['P_KFLMC']


if 'no_of_MC_samples_for_state_RMSE' in preds:
    no_of_MC_samples_for_state_RMSE = preds['no_of_MC_samples_for_state_RMSE']
else:
    no_of_MC_samples_for_state_RMSE = 20

if 'x_state_KF_LMC_Monte_Carlo' in preds:
    x_state_KF_LMC_Monte_Carlo = preds['x_state_KF_LMC_Monte_Carlo']

if 'P_KFLMC_Monte_Carlo' in preds:
    P_KFLMC_Monte_Carlo = preds['P_KFLMC_Monte_Carlo']

if 'y_measured_Monte_Carlo' in preds:
    y_measured_Monte_Carlo = preds['y_measured_Monte_Carlo']
else:
    y_measured_Monte_Carlo = preds['y_measured']


if 'x_state_KF_TMC_monte_carlo' in preds:
    x_state_KF_TMC_monte_carlo = preds['x_state_KF_TMC_monte_carlo']
if 'P_KF_TMC_monte_carlo' in preds:
    P_KF_TMC_monte_carlo = preds['P_KF_TMC_monte_carlo']


if 'cov_true_on_train_data' in preds:
    cov_true_on_train_data = preds['cov_true_on_train_data']
else:
    cov_true_on_train_data = preds['cov_true']


if 'cov_predicted_on_train_data' in preds:
    cov_predicted_on_train_data = preds['cov_predicted_on_train_data']
else:
    cov_predicted_on_train_data = preds['cov_true']

if 'y1y2_stored_with_noise' in preds:
    y1y2_stored_with_noise = preds['y1y2_stored_with_noise']
else:
    y1y2_stored_with_noise = preds['y1y2_stored']


if 'x_state_NOMINAL_monte_carlo' in preds:
    x_state_NOMINAL_monte_carlo=preds['x_state_NOMINAL_monte_carlo']

# </editor-fold>



# <editor-fold desc="print - lopg, file name, Q value">
print('\nFile used: ', file_name)
print('\nLog P opitmized: ', logP)
# print('x_0 mean: ', x0_mean)
# print('x_0 variance: ', x0_cov)
if 'Q_value' in preds:
    print('Q value:\n',Q_value)
# if 'no_of_cov_test_re_run' in preds:
#     print('no_of_cov_test_re_run:', no_of_cov_test_re_run)
# </editor-fold>





#only calculates cov_true if preds doesn't have 'cov_true' or 'y1y2_stored'
# <editor-fold desc="Cov_true and y1y2_stored calculation (not running, since they are in preds)">

dt = 0.1
V = preds['V']  # linear veloctity of the target
r = preds['r']  # radius of the target's circling
d_theta = (V / r) * dt
k = len(x_true)-1  # x is k+1 amount, since it goes from 0 to k.

def sigma_R(x):
    K = 0.0001
    y_1 = np.sqrt(x[0] ** 2 + x[1] ** 2)
    y_2 = np.arctan(x[1] / x[0])
    a0 = 1
    a1 = 1
    a2 = 1
    g_y1 = a0 + a1 * (a2 - y_1) ** 2
    sigma = K * g_y1 / (np.cos(y_2)) ** 2
    return sigma

def sigma_R_for_KF_TMC(y):
    K = 0.0001
    y_1 = y[0]
    y_2 = y[1]
    a0 = 1
    a1 = 1
    a2 = 1
    g_y1 = a0 + a1 * (a2 - y_1) ** 2
    sigma = K * g_y1 / (np.cos(y_2)) ** 2
    return sigma

if 'y1y2_stored' not in preds:
    y1y2_stored = np.zeros((k + 1, 2))
    y_for_regular_KF = np.zeros((k+1, 2))  # y is k amount, since it goes from 1 to k. y_0 is just dummy. keeping it for indexing ease. And 2 measurements: range and bearing, or, x and y cartesian positions
    for l in range(1, k + 1):  # loops from 1 to k
        var_of_epsilon_1 = 0.01
        epsilon1_l = np.random.normal(0, np.sqrt(var_of_epsilon_1))
        r_l = sigma_R(x_true[l, :])
        epsilon2_l = np.random.normal(0, np.sqrt(r_l))
        y1_l = np.sqrt(x_true[l, 0] ** 2 + x_true[l, 1] ** 2) + epsilon1_l
        y2_l = np.arctan(x_true[l, 1] / x_true[l, 0]) + epsilon2_l
        y1y2_stored[l, :] = [y1_l,y2_l]
        y_for_regular_KF[l, :] = [y1_l * np.cos(y2_l), y1_l * np.sin(y2_l)]




# <editor-fold desc="Old cov true calc (commented out)">
#
# if 'cov_true' in preds:
#     cov_true = cov_true
# else: #will not run
#
#     y_measured_for_cov_calc = np.zeros((samples_of_meas_for_cov_calc, k+1, 2))  # y is k amount, since it goes from 1 to k. y_0 is just dummy. keeping it for indexing ease. And 2 measurements: range and bearing, or, x and y cartesian positions
#
#     for n in range(samples_of_meas_for_cov_calc):
#         for l in range(1, k + 1):  # loops from 1 to k
#             var_of_epsilon_1 = 0.01
#             epsilon1_l = np.random.normal(0, np.sqrt(var_of_epsilon_1))
#             r_l = sigma_R(x_true[l, :])
#             epsilon2_l = np.random.normal(0, np.sqrt(r_l))
#             y1_l = np.sqrt(x_true[l, 0] ** 2 + x_true[l, 1] ** 2) + epsilon1_l
#             y2_l = np.arctan(x_true[l, 1] / x_true[l, 0]) + epsilon2_l
#             y_measured_for_cov_calc[n, l, :] = [y1_l * np.cos(y2_l), y1_l * np.sin(y2_l)]
#
#
#     err_for_cov_calc = y_measured_for_cov_calc-x_true[:,0:2]
#     cov_true_calc = np.zeros((k+1,2,2)) # cov_true_0 is dummy
#     for i in range(1,k+1): # y is k amount, since it goes from 1 to k. y_0 is just dummy. since we calculating cov of y, so we should go from 1 to k. In for loop (1,k+1) means we will go until k.
#         cov_true_calc[i,:,:] = np.cov(err_for_cov_calc[:,i,:], rowvar=False)
#
#     cov_true = cov_true_calc[1:,:,:]  # cov_true_0 is dummy
# </editor-fold>



# </editor-fold>





# <editor-fold desc="KF - nominal">


if target_motion_type == 'consnant linear':
    A_dynamics = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])  # A for constant linear motion
if target_motion_type == 'consnant circular':
    A_dynamics = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, np.cos(d_theta), -np.sin(d_theta)],[0, 0, np.sin(d_theta), np.cos(d_theta)]])  # A for anti-clockwise circular motion
    # A_dynamics = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, np.cos(d_theta), np.sin(d_theta)],[0, 0, -np.sin(d_theta), np.cos(d_theta)]])  # A for clockwise circular motion

C_meas_eq = np.array([[1,0,0,0],[0,1,0,0]])

x0_sampled = np.random.normal(x0_mean, x0_cov)
# x0_sampled = np.random.normal(x0_mean, np.sqrt(x0_cov))
x_state_KF_nominal = np.zeros((len(x_states), 4))
x_state_KF_nominal[0, :] = x0_sampled
P_KF_nominal = np.zeros((len(x_states), 4, 4))

if 'Q_value' in preds:
    Q_value_diag = np.diag(Q_value)



for l in range(1, len(x_states)): #KF update loop
    if add_Q_status is True:
        w = np.random.normal(np.array([0, 0, 0, 0]), np.sqrt(Q_value_diag)).reshape((1, 4))
    else:
        # w = np.array([0, 0, 0, 0]).reshape((1, 4))
        w = np.random.normal(np.array([0, 0, 0, 0]), np.sqrt(Q_value_diag)).reshape((1, 4))

    x_hat_predicted_state = np.matmul(A_dynamics, x_state_KF_nominal[l - 1, :]) + w
    x_hat_predicted_state = x_hat_predicted_state.transpose()
    P_hat_predicted_P = np.matmul(np.matmul(A_dynamics, P_KF_nominal[l - 1, :, :]), np.transpose(A_dynamics)) + Q_value



    y1 = y1y2_stored_with_noise[l, 0]
    y2 = y1y2_stored_with_noise[l, 1]
    var_of_epsilon_1 = 0.01
    r1 = var_of_epsilon_1
    # r2 = sigma_R(x_true[l, :])
    r2 = 0.005
    interim_mat1 = np.array([[np.cos(y2), -y1 * np.sin(y2)], [np.sin(y2), y1 * np.cos(y2)]])
    interim_mat2 = np.array([[r1, 0], [0, r2]])
    R_l = np.matmul(np.matmul(interim_mat1, interim_mat2), np.transpose(interim_mat1))

    y_tilde_innovation_residual = y_measured[l, :].reshape(2,1) - np.matmul(C_meas_eq, x_hat_predicted_state)
    S_innovation_cov = np.matmul(np.matmul(C_meas_eq, P_hat_predicted_P), np.transpose(C_meas_eq)) + R_l

    K_kalman_gain = np.matmul(np.matmul(P_hat_predicted_P, np.transpose(C_meas_eq)), np.linalg.inv(S_innovation_cov))

    x_updated_states = x_hat_predicted_state + np.matmul(K_kalman_gain,y_tilde_innovation_residual)
    P_updated_cov = np.matmul(  np.eye(4)-np.matmul(K_kalman_gain,C_meas_eq)   ,   P_hat_predicted_P)


    x_state_KF_nominal[l, :] = x_updated_states.transpose()
    P_KF_nominal[l, :] = P_updated_cov
# </editor-fold>



# <editor-fold desc="(KF-TMC) regular Kalman filter with true R">

if target_motion_type == 'consnant linear':
    A_dynamics = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])  # A for constant linear motion
if target_motion_type == 'consnant circular':
    A_dynamics = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, np.cos(d_theta), -np.sin(d_theta)],[0, 0, np.sin(d_theta), np.cos(d_theta)]])  # A for anti-clockwise circular motion
    # A_dynamics = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, np.cos(d_theta), np.sin(d_theta)],[0, 0, -np.sin(d_theta), np.cos(d_theta)]])  # A for clockwise circular motion

C_meas_eq = np.array([[1,0,0,0],[0,1,0,0]])

x0_sampled = np.random.normal(x0_mean, x0_cov)
# x0_sampled = np.random.normal(x0_mean, np.sqrt(x0_cov))
x_state_KFTMC = np.zeros((len(x_states), 4))
x_state_KFTMC[0, :] = x0_sampled
P_KFTMC = np.zeros((len(x_states), 4, 4))

if 'Q_value' in preds:
    Q_value_diag = np.diag(Q_value)

for l in range(1, len(x_states)): #KF update loop
    if add_Q_status is True:
        w = np.random.normal(np.array([0, 0, 0, 0]), np.sqrt(Q_value_diag)).reshape((1, 4))
    else:
        w = np.array([0, 0, 0, 0]).reshape((1, 4))

    x_hat_predicted_state = np.matmul(A_dynamics, x_state_KFTMC[l - 1, :]) + w
    x_hat_predicted_state = x_hat_predicted_state.transpose()
    P_hat_predicted_P = np.matmul(np.matmul(A_dynamics, P_KFTMC[l - 1, :, :]), np.transpose(A_dynamics)) + Q_value

    R_l = cov_true[l,:,:]

    y_tilde_innovation_residual = y_measured[l, :].reshape(2,1) - np.matmul(C_meas_eq, x_hat_predicted_state)
    S_innovation_cov = np.matmul(np.matmul(C_meas_eq, P_hat_predicted_P), np.transpose(C_meas_eq)) + R_l

    K_kalman_gain = np.matmul(np.matmul(P_hat_predicted_P, np.transpose(C_meas_eq)), np.linalg.inv(S_innovation_cov))

    x_updated_states = x_hat_predicted_state + np.matmul(K_kalman_gain,y_tilde_innovation_residual)
    P_updated_cov = np.matmul(  np.eye(4)-np.matmul(K_kalman_gain,C_meas_eq)   ,   P_hat_predicted_P)


    x_state_KFTMC[l, :] = x_updated_states.transpose()
    P_KFTMC[l, :] = P_updated_cov



# </editor-fold>




# <editor-fold desc="(KFLMC) VB-AKF">

if 'x_state_KF_LMC' not in preds:

    x_state_KF_LMC = np.zeros((len(x_states), 4))
    x_state_KF_LMC[0, :] = x0_sampled
    P_KFLMC = np.zeros((len(x_states), 4,4))


    for l in range(1, len(x_states)): #KF update loop
        if add_Q_status is True:
            w = np.random.normal(np.array([0, 0, 0, 0]), np.sqrt(Q_value_diag)).reshape((1, 4))
        else:
            w = np.array([0, 0, 0, 0]).reshape((1, 4))

        x_hat_predicted_state = np.matmul(A_dynamics, x_state_KF_LMC[l - 1, :]) + w
        x_hat_predicted_state = x_hat_predicted_state.transpose()
        P_hat_predicted_P = np.matmul(np.matmul(A_dynamics, P_KFLMC[l - 1, :, :]), np.transpose(A_dynamics)) + Q_value

        try:
            R_l = cov_predicted[l-1,:,:] #although indexing stars from (l-1) it actually is k-th R, because R_l stars from k=1 to k. But since python index must start from 0 so I need to use l-1.
        except:
            R_l = np.zeros((2,2))
        y_tilde_innovation_residual = y_measured[l, :].reshape(2,1) - np.matmul(C_meas_eq, x_hat_predicted_state)
        S_innovation_cov = np.matmul(np.matmul(C_meas_eq, P_hat_predicted_P), np.transpose(C_meas_eq)) + R_l

        K_kalman_gain = np.matmul(np.matmul(P_hat_predicted_P, np.transpose(C_meas_eq)), np.linalg.inv(S_innovation_cov))

        x_updated_states = x_hat_predicted_state + np.matmul(K_kalman_gain,y_tilde_innovation_residual)
        P_updated_cov = np.matmul(  np.eye(4)-np.matmul(K_kalman_gain,C_meas_eq)   ,   P_hat_predicted_P)


        x_state_KF_LMC[l, :] = x_updated_states.transpose()
        P_KFLMC[l,:] = P_updated_cov

# </editor-fold>





# <editor-fold desc="state plots">
if plot_states is True:
    plt.figure(1)
    plt.plot(np.linspace(1,len(x_states),len(x_states)), x_true[:,0], '--k', label='True State')
    plt.plot(np.linspace(1,len(x_states),len(x_states)), x_states[:,0], '-.r', label='Estimated State')
    plt.plot(np.linspace(1,len(x_states),len(x_states)), y_measured[:, 0], '.b', label='Measurement of State', markersize=1)
    plt.title('State 1 (position x)')
    plt.legend(loc='best')
    plt.ylabel('State value')
    plt.xlabel('Time steps')

    plt.figure(2)
    plt.plot(np.linspace(1,len(x_states),len(x_states)), x_true[:,1], '--k', label='True State')
    plt.plot(np.linspace(1,len(x_states),len(x_states)), x_states[:,1], '-.r', label='Estimated State')
    plt.plot(np.linspace(1,len(x_states),len(x_states)), y_measured[:, 1], '.b', label='Measurement of State', markersize=1)
    plt.title('State 2 (position y)')
    plt.legend(loc='best')
    plt.ylabel('State value')
    plt.xlabel('Time steps')

    plt.figure(3)
    plt.plot(np.linspace(1,len(x_states),len(x_states)), x_true[:,2], '--k', label='True State')
    plt.plot(np.linspace(1,len(x_states),len(x_states)), x_states[:,2], '-.r', label='Estimated State')
    plt.title('State 3 (velocity x)')
    plt.legend(loc='best')
    plt.ylabel('State value')
    plt.xlabel('Time steps')
    if enforece_ylim_for_velo_states == True:
        plt.ylim([0, y_lim_value_for_velo_states])


    plt.figure(4)
    plt.plot(np.linspace(1,len(x_states),len(x_states)), x_true[:,3], '--k', label='True State')
    plt.plot(np.linspace(1,len(x_states),len(x_states)), x_states[:,3], '-.r', label='Estimated State')
    plt.title('State 4 (velocity y)')
    plt.legend(loc='best')
    plt.ylabel('State value')
    plt.xlabel('Time steps')
    if enforece_ylim_for_velo_states == True:
        plt.ylim([0, y_lim_value_for_velo_states])


# <editor-fold desc="(comment it out) Target trajectory for few circles - for ACC poster">
if False:
    how_many_time_steps_in_trajectory_plot = 380
    # how_many_time_steps_in_trajectory_plot = len(x_true)
    plt.figure(5)
    plt.plot(x_true[:how_many_time_steps_in_trajectory_plot,0],x_true[:how_many_time_steps_in_trajectory_plot,1],'-k', label='True',markersize=2, linewidth=2)
    # plt.plot(x_state_regular_KF[:,0],x_state_regular_KF[:,1],'--b', label='KF-TMC',markersize=2, linewidth=1)
    plt.plot(x_state_KF_LMC[:how_many_time_steps_in_trajectory_plot,0], x_state_KF_LMC[:how_many_time_steps_in_trajectory_plot, 1], '--^r', label='AKF', markersize=2, linewidth=1)
    plt.plot(x_states[:how_many_time_steps_in_trajectory_plot,0],x_states[:how_many_time_steps_in_trajectory_plot,1],'--.c', label='Nominal KF',markersize=1, linewidth=1)
    plt.plot(y_measured[1:how_many_time_steps_in_trajectory_plot, 0], y_measured[1:how_many_time_steps_in_trajectory_plot, 1], '.b', label='Measurement', markersize=3)
    if cov_test_location_type == 'on outside uniform sq':
        plt.plot(x_test[:,0],x_test[:,1],'+g', label='Test locations', markersize=6)
    plt.title('Target Trajectory')
    plt.legend(loc='best')
    plt.xlabel('x')
    plt.ylabel('y')
# </editor-fold>


if True:
    plt.figure(5)
    plt.plot(x_true[:,0],x_true[:,1],'-k', label='True states',markersize=2, linewidth=2)
    plt.plot(x_state_KFTMC[:, 0], x_state_KFTMC[:, 1], '--b', label='KF-TMC', markersize=2, linewidth=1)
    plt.plot(x_state_KF_LMC[:,0], x_state_KF_LMC[:, 1], '--^g', label='VB-AKF', markersize=2, linewidth=1)
    plt.plot(x_state_KF_nominal[:,0],x_state_KF_nominal[:,1],'--.r', label='Nominal KF',markersize=1, linewidth=1)
    plt.plot(y_measured[:, 0], y_measured[:, 1], '.y', label='Measurement', markersize=3)
    if cov_test_location_type == 'on outside uniform sq':
        plt.plot(x_test[:,0],x_test[:,1],'+g', label='Test locations', markersize=6)
    plt.title('Estimated states with true and measured states')
    plt.legend(loc='best')
# </editor-fold>


# <editor-fold desc="error in state plots">
if plot_states_error is True:
    plt.figure(6)
    plt.plot(np.linspace(1, len(x_states), len(x_states)), np.abs(x_true[:, 0]-x_states[:, 0]), '--r', label='AKF error')
    plt.plot(np.linspace(1, len(x_states), len(x_states)), np.abs(x_true[:, 0] - x_state_KFTMC[:, 0]), '-k', label='KFTMC error')
    plt.title('State 1 error (position x)')
    plt.legend(loc='best')
    plt.ylabel('State error value')
    plt.xlabel('Time steps')

    plt.figure(7)
    plt.plot(np.linspace(1, len(x_states), len(x_states)), np.abs(x_true[:, 1]-x_states[:, 1]), '--r', label='AKF error')
    plt.plot(np.linspace(1, len(x_states), len(x_states)), np.abs(x_true[:, 1] - x_state_KFTMC[:, 1]), '-k', label='KFTMC error')
    plt.title('State 2 error (position y)')
    plt.legend(loc='best')
    plt.ylabel('State error value')
    plt.xlabel('Time steps')

    plt.figure(8)
    plt.plot(np.linspace(1, len(x_states), len(x_states)), np.abs(x_true[:, 2]-x_states[:, 2]), '--r', label='AKF error')
    plt.plot(np.linspace(1, len(x_states), len(x_states)), np.abs(x_true[:, 2] - x_state_KFTMC[:, 2]), '-k', label='KFTMC error')
    plt.title('State 3 error (velocity x)')
    plt.legend(loc='best')
    plt.ylabel('State error value')
    plt.xlabel('Time steps')
    if enforece_ylim_for_velo_states == True:
        plt.ylim([0, y_lim_value_for_velo_states])

    plt.figure(9)
    plt.plot(np.linspace(1, len(x_states), len(x_states)), np.abs(x_true[:, 3]-x_states[:, 3]), '--r', label='AKF error')
    plt.plot(np.linspace(1, len(x_states), len(x_states)), np.abs(x_true[:, 3] - x_state_KFTMC[:, 3]), '-k', label='KFTMC error')
    plt.title('State 4 error (velocity y)')
    plt.legend(loc='best')
    plt.ylabel('State error value')
    plt.xlabel('Time steps')
    if enforece_ylim_for_velo_states == True:
        plt.ylim([0, y_lim_value_for_velo_states])
# </editor-fold>






# <editor-fold desc="plot Train and test locations only">
if plot_train_and_test_location_only is True:
    plt.figure(10)
    plt.plot(x_true[:,0],x_true[:,1],'--k', label='True locations')
    plt.plot(x_test[:,0],x_test[:,1],'+g', label='Test locations', markersize=3)
    plt.title('Train and test locations')
    plt.legend(loc='best')
# </editor-fold>




# cov_predicted = cov_predicted[1:,:,:]


marker_size_for_sub_plots = 4
# <editor-fold desc="cov sub plots - 3*1">
if cov_test_location_type == 'on x_true':
    cov_true = cov_true[1:,:,:]

if plot_cov is True:
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6,12))


    ax1.plot(np.linspace(1,len(cov_true),len(cov_true)), cov_true[0:,0,0], 'o', mfc='none',label='True',markersize=marker_size_for_sub_plots)
    ax1.plot(np.linspace(1,len(cov_predicted),len(cov_predicted)), cov_predicted[:,0,0], '+',label='Estimated',markersize=marker_size_for_sub_plots)
    ax1.set_title('Covariance estimation on test data (Element 1)')
    ax1.set_ylabel('Covariance element value')
    if ylim_enforce_for_cov_sub_plots is True:
        ax1.set_ylim([ylim_low_cov_plots, ylim_high_cov_plots])
    ax1.legend()



    ax2.plot(np.linspace(1,len(cov_true),len(cov_true)), cov_true[0:,0,1], 'o', mfc='none',label='True',markersize=marker_size_for_sub_plots)
    ax2.plot(np.linspace(1,len(cov_predicted),len(cov_predicted)), cov_predicted[:,0,1], '+',label='Estimated',markersize=marker_size_for_sub_plots)
    ax2.set_title('Covariance estimation on test data (Element 2)')
    ax2.set_ylabel('Covariance element value')
    if ylim_enforce_for_cov_sub_plots is True:
        ax2.set_ylim([ylim_low_cov_plots, ylim_high_cov_plots])
    ax2.legend()



    ax3.plot(np.linspace(1,len(cov_true),len(cov_true)), cov_true[0:,1,1], 'o', mfc='none',label='True',markersize=marker_size_for_sub_plots)
    ax3.plot(np.linspace(1,len(cov_predicted),len(cov_predicted)), cov_predicted[:,1,1], '+',label='Estimated',markersize=marker_size_for_sub_plots)
    ax3.set_title('Covariance estimation on test data (Element 4)')
    ax3.set_ylabel('Covariance element value')
    ax3.set_xlabel('test point no')
    if ylim_enforce_for_cov_sub_plots is True:
        ax3.set_ylim([ylim_low_cov_plots, ylim_high_cov_plots])
    ax3.legend()
# </editor-fold>

# <editor-fold desc="cov error sub plots - 3*1">
if plot_cov_error is True:
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6,12))


    ax1.plot(np.linspace(1,len(cov_predicted),len(cov_predicted)), np.abs(cov_predicted[:,0,0]-cov_true[0:,0,0]), '.r',label='Covariance error',markersize=marker_size_for_sub_plots)
    ax1.set_title('Absolute covariance error on test data (Element 1)')
    ax1.set_ylabel('Covariance error value')
    if ylim_enforce_for_cov_sub_plots is True:
        ax1.set_ylim([0, ylim_for_cov_error_plots])
    ax1.legend()


    ax2.plot(np.linspace(1,len(cov_predicted),len(cov_predicted)), np.abs(cov_predicted[:,0,1]-cov_true[0:,0,1]), '.r',label='Covariance error',markersize=marker_size_for_sub_plots)
    ax2.set_title('Absolute covariance error on test data (Element 2)')
    ax2.set_ylabel('Covariance error value')
    if ylim_enforce_for_cov_sub_plots is True:
        ax2.set_ylim([0, ylim_for_cov_error_plots])
    ax2.legend()


    ax3.plot(np.linspace(1,len(cov_predicted),len(cov_predicted)), np.abs(cov_predicted[:,1,1]-cov_true[0:,1,1]), '.r', label='Covariance error',markersize=marker_size_for_sub_plots)
    ax3.set_title('Absolute covariance error on test data (Element 4)')
    ax3.set_ylabel('Covariance error value')
    ax3.set_xlabel('test point no')
    if ylim_enforce_for_cov_sub_plots is True:
        ax3.set_ylim([0, ylim_for_cov_error_plots])
    ax3.legend()
# </editor-fold>




# <editor-fold desc="cov (on train data) sub plots - 3*1">

if plot_cov is True:
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6,12))


    ax1.plot(np.linspace(1,len(cov_true_on_train_data),len(cov_true_on_train_data)), cov_true_on_train_data[0:,0,0], 'o', mfc='none',label='True',markersize=marker_size_for_sub_plots)
    ax1.plot(np.linspace(1,len(cov_predicted_on_train_data),len(cov_predicted_on_train_data)), cov_predicted_on_train_data[:,0,0], '+',label='Estimated',markersize=marker_size_for_sub_plots)
    ax1.set_title('Covariance estimation on train data (Element 1)')
    ax1.set_ylabel('Covariance element value')
    if ylim_enforce_for_cov_sub_plots is True:
        ax1.set_ylim([ylim_low_cov_plots, ylim_high_cov_plots])
    ax1.legend()



    ax2.plot(np.linspace(1,len(cov_true_on_train_data),len(cov_true_on_train_data)), cov_true_on_train_data[0:,0,1], 'o', mfc='none',label='True',markersize=marker_size_for_sub_plots)
    ax2.plot(np.linspace(1,len(cov_predicted_on_train_data),len(cov_predicted_on_train_data)), cov_predicted_on_train_data[:,0,1], '+',label='Estimated',markersize=marker_size_for_sub_plots)
    ax2.set_title('Covariance estimation on train data (Element 2)')
    ax2.set_ylabel('Covariance element value')
    if ylim_enforce_for_cov_sub_plots is True:
        ax2.set_ylim([ylim_low_cov_plots, ylim_high_cov_plots])
    ax2.legend()



    ax3.plot(np.linspace(1,len(cov_true_on_train_data),len(cov_true_on_train_data)), cov_true_on_train_data[0:,1,1], 'o', mfc='none',label='True',markersize=marker_size_for_sub_plots)
    ax3.plot(np.linspace(1,len(cov_predicted_on_train_data),len(cov_predicted_on_train_data)), cov_predicted_on_train_data[:,1,1], '+',label='Estimated',markersize=marker_size_for_sub_plots)
    ax3.set_title('Covariance estimation on train data (Element 4)')
    ax3.set_ylabel('Covariance element value')
    ax3.set_xlabel('test point no')
    if ylim_enforce_for_cov_sub_plots is True:
        ax3.set_ylim([ylim_low_cov_plots, ylim_high_cov_plots])
    ax3.legend()
# </editor-fold>

# <editor-fold desc="cov error (on train data) sub plots - 3*1">
if plot_cov_error is True:
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6,12))


    ax1.plot(np.linspace(1,len(cov_predicted_on_train_data),len(cov_predicted_on_train_data)), np.abs(cov_predicted_on_train_data[:,0,0]-cov_true_on_train_data[0:,0,0]), '.r',label='Covariance error',markersize=marker_size_for_sub_plots)
    ax1.set_title('Absolute covariance error on train data (Element 1)')
    ax1.set_ylabel('Covariance error value')
    if ylim_enforce_for_cov_sub_plots is True:
        ax1.set_ylim([0, ylim_for_cov_error_plots])
    ax1.legend()


    ax2.plot(np.linspace(1,len(cov_predicted_on_train_data),len(cov_predicted_on_train_data)), np.abs(cov_predicted_on_train_data[:,0,1]-cov_true_on_train_data[0:,0,1]), '.r',label='Covariance error',markersize=marker_size_for_sub_plots)
    ax2.set_title('Absolute covariance error on train data (Element 2)')
    ax2.set_ylabel('Covariance error value')
    if ylim_enforce_for_cov_sub_plots is True:
        ax2.set_ylim([0, ylim_for_cov_error_plots])
    ax2.legend()


    ax3.plot(np.linspace(1,len(cov_predicted_on_train_data),len(cov_predicted_on_train_data)), np.abs(cov_predicted_on_train_data[:,1,1]-cov_true_on_train_data[0:,1,1]), '.r', label='Covariance error',markersize=marker_size_for_sub_plots)
    ax3.set_title('Absolute covariance error on train data (Element 4)')
    ax3.set_ylabel('Covariance error value')
    ax3.set_xlabel('test point no')
    if ylim_enforce_for_cov_sub_plots is True:
        ax3.set_ylim([0, ylim_for_cov_error_plots])
    ax3.legend()
# </editor-fold>








# <editor-fold desc="COV - Abs error & RMSE clc and print">
absolute_error_at_each_point = np.abs(cov_true-cov_predicted)
total_absolute_error = np.sum(absolute_error_at_each_point, axis=0)
mean_absolute_error_at_each_point = np.mean(absolute_error_at_each_point, axis=0)
var_absolute_error_at_each_point = np.var(absolute_error_at_each_point, axis=0)
# print('\nTotal abs error, abs error mean, abs error variance: (respectively)')
print('\nElement 1: ', round(total_absolute_error[0,0], 2), ', ', round(mean_absolute_error_at_each_point[0,0],2), ', ', round(var_absolute_error_at_each_point[0,0],2))
print('Element 2: ', round(total_absolute_error[0,1],2), ', ', round(mean_absolute_error_at_each_point[0,1],2), ', ', round(var_absolute_error_at_each_point[0,1],2))
print('Element 4: ', round(total_absolute_error[1,1],2), ', ', round(mean_absolute_error_at_each_point[1,1],2), ', ', round(var_absolute_error_at_each_point[1,1],2))
print('\nTAE: ', round(total_absolute_error[0,0] + total_absolute_error[0,1] + total_absolute_error[1,0] + total_absolute_error[1,1] , 2))
sq_absolute_error_at_each_point = np.square(cov_true-cov_predicted)
sq_total_absolute_error = np.sum(sq_absolute_error_at_each_point)
rmse = np.sqrt(np.mean((cov_true-cov_predicted)**2))
# print('Err=Bias^2 + var: ', sq_total_absolute_error + np.sum(var_absolute_error_at_each_point), '   rmse:', rmse)
print('rmse:', rmse)

# </editor-fold>





# <editor-fold desc="state RMSE calc from monte-carlo states">
# if plot_state_estimation_RMSE_separately is True:

if 'dt' in preds:
    dt = preds['dt']
else:
    dt = 0.1

V = preds['V']  # linear veloctity of the target
r = preds['r']  # radius of the target's circling
d_theta = (V / r) * dt

if target_motion_type == 'consnant linear':
    A_dynamics = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])  # A for constant linear motion
if target_motion_type == 'consnant circular':
    A_dynamics = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, np.cos(d_theta), -np.sin(d_theta)],[0, 0, np.sin(d_theta), np.cos(d_theta)]])  # A for anti-clockwise circular motion
    # A_dynamics = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, np.cos(d_theta), np.sin(d_theta)],[0, 0, -np.sin(d_theta), np.cos(d_theta)]])  # A for clockwise circular motion

if 'x_state_NOMINAL_monte_carlo' not in preds:
    x_state_NOMINAL_monte_carlo = np.zeros((no_of_MC_samples_for_state_RMSE, len(x_states), 4))
if 'x_state_KF_TMC_monte_carlo' not in preds or do_KF_TMC_RMSE_calc_here is True:
    x_state_KF_TMC_monte_carlo = np.zeros((no_of_MC_samples_for_state_RMSE, len(x_states), 4))
    P_KF_TMC_monte_carlo = np.zeros((no_of_MC_samples_for_state_RMSE, len(x_states), 4, 4))
if 'x_state_KF_LMC_Monte_Carlo' not in preds:
    x_state_KF_LMC_Monte_Carlo = np.zeros((no_of_MC_samples_for_state_RMSE, len(x_states), 4))
    P_KF_LMC_monte_carlo = np.zeros((no_of_MC_samples_for_state_RMSE, len(x_states), 4, 4))
C_meas_eq = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
cov_true = preds['cov_true']

if do_monte_carlo_here is True or 'x_state_NOMINAL_monte_carlo' not in preds or 'x_state_KF_LMC_Monte_Carlo' not in preds or 'x_state_KF_TMC_monte_carlo' not in preds:
    for n_monte in range(no_of_MC_samples_for_state_RMSE):
        # if 'x_state_KF_TMC_monte_carlo' not in preds or 'x_state_KF_LMC_Monte_Carlo' not in preds:
        print('MC now:',n_monte)
        x0_sampled = np.random.normal(x0_mean, x0_cov)
        x0_sampled[0,0:2] = y_measured_Monte_Carlo[n_monte, 1, :]
        if 'x_state_KF_TMC_monte_carlo' not in preds  or do_KF_TMC_RMSE_calc_here is True:
            x_state_KF_TMC_monte_carlo[n_monte, 0, :] = x0_sampled
        if 'x_state_KF_LMC_Monte_Carlo' not in preds:
            x_state_KF_LMC_Monte_Carlo[n_monte, 0, :] = x0_sampled

        if 'Q_value' in preds:
            Q_value_diag = np.diag(Q_value)
        for l in range(1, len(x_states)):
            if add_Q_status is True:
                w = np.random.normal(np.array([0, 0, 0, 0]), np.sqrt(Q_value_diag)).reshape((1, 4))
            else:
                w = np.array([0, 0, 0, 0]).reshape((1, 4))

            """Nominal KF monte carlo here"""
            x_hat_predicted_state = np.matmul(A_dynamics, x_state_KF_nominal[l - 1, :]) + w
            x_hat_predicted_state = x_hat_predicted_state.transpose()
            P_hat_predicted_P = np.matmul(np.matmul(A_dynamics, P_KF_nominal[l - 1, :, :]),np.transpose(A_dynamics)) + Q_value

            y1 = y1y2_stored_with_noise[l, 0]
            y2 = y1y2_stored_with_noise[l, 1]
            var_of_epsilon_1 = 0.01
            r1 = var_of_epsilon_1
            # r2 = sigma_R(x_true[l, :])
            r2 = 0.005
            interim_mat1 = np.array([[np.cos(y2), -y1 * np.sin(y2)], [np.sin(y2), y1 * np.cos(y2)]])
            interim_mat2 = np.array([[r1, 0], [0, r2]])
            R_l = np.matmul(np.matmul(interim_mat1, interim_mat2), np.transpose(interim_mat1))

            y_tilde_innovation_residual = y_measured_Monte_Carlo[n_monte, l, :].reshape(2, 1) - np.matmul(C_meas_eq, x_hat_predicted_state)
            S_innovation_cov = np.matmul(np.matmul(C_meas_eq, P_hat_predicted_P), np.transpose(C_meas_eq)) + R_l

            K_kalman_gain = np.matmul(np.matmul(P_hat_predicted_P, np.transpose(C_meas_eq)),np.linalg.inv(S_innovation_cov))

            x_updated_states = x_hat_predicted_state + np.matmul(K_kalman_gain, y_tilde_innovation_residual)
            P_updated_cov = np.matmul(np.eye(4) - np.matmul(K_kalman_gain, C_meas_eq), P_hat_predicted_P)

            x_state_NOMINAL_monte_carlo[n_monte, l, :] = x_updated_states.transpose()


            """KF-TMC monte carlo starts here"""
            if 'x_state_KF_TMC_monte_carlo' not in preds or do_KF_TMC_RMSE_calc_here is True:
                x_hat_predicted_state = np.matmul(A_dynamics, x_state_KF_TMC_monte_carlo[n_monte, l - 1, :]) + w
                x_hat_predicted_state = x_hat_predicted_state.transpose()
                P_hat_predicted_P = np.matmul(np.matmul(A_dynamics, P_KF_TMC_monte_carlo[n_monte, l - 1, :, :]), np.transpose(A_dynamics)) + Q_value

                R_l = cov_true[l, :, :]

                if 'y_measured_Monte_Carlo' in preds:
                    y_tilde_innovation_residual = y_measured_Monte_Carlo[n_monte, l, :].reshape(2, 1) - np.matmul(C_meas_eq, x_hat_predicted_state)
                else:
                    y_tilde_innovation_residual = y_measured[l, :].reshape(2, 1) - np.matmul(C_meas_eq, x_hat_predicted_state)
                S_innovation_cov = np.matmul(np.matmul(C_meas_eq, P_hat_predicted_P), np.transpose(C_meas_eq)) + R_l

                K_kalman_gain = np.matmul(np.matmul(P_hat_predicted_P, np.transpose(C_meas_eq)),np.linalg.inv(S_innovation_cov))

                x_updated_states = x_hat_predicted_state + np.matmul(K_kalman_gain, y_tilde_innovation_residual)
                P_updated_cov = np.matmul(np.eye(4) - np.matmul(K_kalman_gain, C_meas_eq), P_hat_predicted_P)

                x_state_KF_TMC_monte_carlo[n_monte, l, :] = x_updated_states.transpose()
                P_KF_TMC_monte_carlo[n_monte, l, :] = P_updated_cov



            """KF-LMC monte carlo starts here"""
            if 'x_state_KF_LMC_Monte_Carlo' not in preds:
                x_hat_predicted_state = np.matmul(A_dynamics, x_state_KF_LMC_Monte_Carlo[n_monte, l - 1, :]) + w
                x_hat_predicted_state = x_hat_predicted_state.transpose()
                P_hat_predicted_P = np.matmul(np.matmul(A_dynamics, P_KF_LMC_monte_carlo[n_monte, l - 1, :, :]), np.transpose(A_dynamics)) + Q_value

                try:
                    R_l = cov_predicted[l - 1, :,:]  # although indexing stars from (l-1) it actually is k-th R, because R_l stars from k=1 to k. But since python index must start from 0 so I need to use l-1.
                except:
                    R_l = np.zeros((2, 2))

                y_tilde_innovation_residual = y_measured[l, :].reshape(2, 1) - np.matmul(C_meas_eq,x_hat_predicted_state)
                S_innovation_cov = np.matmul(np.matmul(C_meas_eq, P_hat_predicted_P), np.transpose(C_meas_eq)) + R_l

                K_kalman_gain = np.matmul(np.matmul(P_hat_predicted_P, np.transpose(C_meas_eq)),np.linalg.inv(S_innovation_cov))

                x_updated_states = x_hat_predicted_state + np.matmul(K_kalman_gain, y_tilde_innovation_residual)
                P_updated_cov = np.matmul(np.eye(4) - np.matmul(K_kalman_gain, C_meas_eq), P_hat_predicted_P)

                x_state_KF_LMC_Monte_Carlo[n_monte, l, :] = x_updated_states.transpose()
                P_KF_LMC_monte_carlo[n_monte, l, :] = P_updated_cov


state_RMSE = np.zeros((len(x_states),4))
KF_TMC_RMSE = np.zeros((len(x_states), 4))
KF_LMC_RMSE = np.zeros((len(x_states), 4))
for k in range(len(x_states)):
    state_RMSE[k,:] = np.sqrt(np.mean((x_state_NOMINAL_monte_carlo[:, k, :] - x_true[k, :]) ** 2, axis=0))
    KF_TMC_RMSE[k, :] = np.sqrt(np.mean((x_state_KF_TMC_monte_carlo[:, k, :] - x_true[k, :]) ** 2, axis=0))
    KF_LMC_RMSE[k,:] = np.sqrt(np.mean((x_state_KF_LMC_Monte_Carlo[:, k, :] - x_true[k, :]) ** 2, axis=0))
# </editor-fold>




x_state_KF_LMC_Monte_Carlo[0,:,:] = x_state_KF_LMC_Monte_Carlo[1,:,:]

# <editor-fold desc="state RMSE plots">
if plot_state_estimation_RMSE_separately==True: #without monte carlo nothing to plot for RMSE
    plt.figure(13)
    plt.plot(np.linspace(1, len(x_states), len(x_states)), state_RMSE[:,0], '-.r', label='RMSE')
    plt.title('State 1 estimation RMSE (position x)')
    plt.legend(loc='best')
    plt.ylabel('RMSE value')
    plt.xlabel('Time steps')
    plt.ylim([0, ylim_for_RMSE])


    plt.figure(14)
    plt.plot(np.linspace(1, len(x_states), len(x_states)), state_RMSE[:, 1], '-.r', label='RMSE')
    plt.title('State 2 estimation RMSE (position y)')
    plt.legend(loc='best')
    plt.ylabel('RMSE value')
    plt.xlabel('Time steps')
    plt.ylim([0, ylim_for_RMSE])

    plt.figure(15)
    plt.plot(np.linspace(1, len(x_states), len(x_states)), state_RMSE[:, 2], '-.r', label='RMSE')
    plt.title('State 3 estimation RMSE (velocity x)')
    plt.legend(loc='best')
    plt.ylabel('RMSE value')
    plt.xlabel('Time steps')
    # if enforece_ylim_for_velo_states == True:
    plt.ylim([0, ylim_for_RMSE])

    plt.figure(16)
    plt.plot(np.linspace(1, len(x_states), len(x_states)), state_RMSE[:, 3], '-.r', label='RMSE')
    plt.title('State 4 estimation RMSE (velocity y)')
    plt.legend(loc='best')
    plt.ylabel('RMSE value')
    plt.xlabel('Time steps')
    # if enforece_ylim_for_velo_states == True:
    plt.ylim([0, ylim_for_RMSE])
# </editor-fold>



# <editor-fold desc="state RMSE sub plots">
marker_size_for_RMSE_sub_plots = 2
plot_KFTMC_single_error = False

if plot_state_estimation_RMSE_subplot is True:
    fig, ((ax11, ax12), (ax21, ax22)) = plt.subplots(2, 2, figsize=(9,9))


    ax11.plot(np.linspace(1,len(x_states),len(x_states)), state_RMSE[:,0], '--r',label='Nominal KF',markersize=marker_size_for_RMSE_sub_plots)
    if plot_KFTMC_single_error is True:
        ax11.plot(np.linspace(1, len(x_states), len(x_states)), x_true[:, 0] - x_state_KFTMC[:, 0], '.b', label='KFTMC error', markersize=marker_size_for_RMSE_sub_plots)
    if do_KF_TMC_monte_carlo is True:
        ax11.plot(np.linspace(1, len(x_states), len(x_states)), KF_TMC_RMSE[:, 0], '--k', label='KF-TMC RMSE', markersize=marker_size_for_RMSE_sub_plots)
    if do_KF_LMC_monte_carlo is True:
        ax11.plot(np.linspace(1, len(x_states), len(x_states)), KF_LMC_RMSE[:, 0], '--g', label='VB-AKF RMSE', markersize=marker_size_for_RMSE_sub_plots)
    ax11.set_title('State 1 RMSE')
    ax11.set_ylabel('RMSE value')
    ax11.set_ylim([0, ylim_for_RMSE])
    ax11.legend()



    ax12.plot(np.linspace(1,len(x_states),len(x_states)), state_RMSE[:,1], '--r',label='Nominal KF',markersize=marker_size_for_RMSE_sub_plots)
    if plot_KFTMC_single_error is True:
        ax12.plot(np.linspace(1, len(x_states), len(x_states)), x_true[:, 1] - x_state_KFTMC[:, 1], '.k', label='KFTMC error', markersize=marker_size_for_RMSE_sub_plots)
    if do_KF_TMC_monte_carlo is True:
        ax12.plot(np.linspace(1, len(x_states), len(x_states)), KF_TMC_RMSE[:, 1], '--k', label='KF-TMC RMSE', markersize=marker_size_for_RMSE_sub_plots)
    if do_KF_LMC_monte_carlo is True:
        ax12.plot(np.linspace(1, len(x_states), len(x_states)), KF_LMC_RMSE[:, 1], '--g', label='VB-AKF RMSE', markersize=marker_size_for_RMSE_sub_plots)
    ax12.set_title('State 2 RMSE')
    # ax12.set_ylabel('RMSE value')
    ax12.set_ylim([0, ylim_for_RMSE])
    ax12.legend()



    ax21.plot(np.linspace(1, len(x_states), len(x_states)), state_RMSE[:,2], '--.r',label='Nominal KF', markersize=marker_size_for_RMSE_sub_plots)
    if plot_KFTMC_single_error is True:
        ax21.plot(np.linspace(1, len(x_states), len(x_states)), x_true[:, 2] - x_state_KFTMC[:, 2], '--k', label='KFTMC error', markersize=marker_size_for_RMSE_sub_plots)
    if do_KF_TMC_monte_carlo is True:
        ax21.plot(np.linspace(1, len(x_states), len(x_states)), KF_TMC_RMSE[:, 2], '--k', label='KF-TMC RMSE', markersize=marker_size_for_RMSE_sub_plots)
    if do_KF_LMC_monte_carlo is True:
        ax21.plot(np.linspace(1, len(x_states), len(x_states)), KF_LMC_RMSE[:, 2], '--g', label='VB-AKF RMSE', markersize=marker_size_for_RMSE_sub_plots)
    ax21.set_title('State 3 RMSE')
    ax21.set_ylabel('RMSE value')
    ax21.set_xlabel('Time steps')
    ax21.set_ylim([0, ylim_for_RMSE])
    ax21.legend()




    ax22.plot(np.linspace(1, len(x_states), len(x_states)), state_RMSE[:,3], '--.r',label='Nominal KF', markersize=marker_size_for_RMSE_sub_plots)
    if plot_KFTMC_single_error is True:
        ax22.plot(np.linspace(1, len(x_states), len(x_states)), x_true[:, 3] - x_state_KFTMC[:, 3], '--k', label='KFTMC error', markersize=marker_size_for_RMSE_sub_plots)
    if do_KF_TMC_monte_carlo is True:
        ax22.plot(np.linspace(1, len(x_states), len(x_states)), KF_TMC_RMSE[:, 3], '--k', label='KF-TMC RMSE', markersize=marker_size_for_RMSE_sub_plots)
    if do_KF_LMC_monte_carlo is True:
        ax22.plot(np.linspace(1, len(x_states), len(x_states)), KF_LMC_RMSE[:, 3], '--g', label='VB-AKF RMSE', markersize=marker_size_for_RMSE_sub_plots)
    ax22.set_title('State 4 RMSE')
    # ax22.set_ylabel('RMSE value')
    ax22.set_xlabel('Time steps')
    ax22.set_ylim([0, ylim_for_RMSE])
    ax22.legend()
# </editor-fold>





# <editor-fold desc="state sub plots">
marker_size_for_state_sub_plots = 1

if plot_states_subplots is True:
    fig, ((ax11, ax12), (ax21, ax22)) = plt.subplots(2, 2, figsize=(9,9))


    ax11.plot(np.linspace(1,len(x_states),len(x_states)), x_state_KF_nominal[:,0], '--r',label='Nominal KF',markersize=marker_size_for_state_sub_plots, linewidth=1)
    ax11.plot(np.linspace(1,len(x_states),len(x_states)), y_measured[:,0], '.y',label='Measured state',markersize=marker_size_for_state_sub_plots+2, linewidth=1)
    ax11.plot(np.linspace(1, len(x_states), len(x_states)), x_true[:, 0], '-k', label='True state',markersize=marker_size_for_state_sub_plots, linewidth=1)
    ax11.plot(np.linspace(1, len(x_states), len(x_states)), x_state_KFTMC[:, 0], '--b', label='KF-TMC', markersize=marker_size_for_state_sub_plots + 2, linewidth=1)
    ax11.plot(np.linspace(1, len(x_states), len(x_states)), x_state_KF_LMC[:, 0], '--g', label='VB-AKF', markersize=marker_size_for_state_sub_plots + 2, linewidth=1)
    ax11.set_title('State 1')
    ax11.set_ylabel('State value')
    # ax11.set_ylim([0, ylim_for_RMSE])
    ax11.legend()

    ax12.plot(np.linspace(1, len(x_states), len(x_states)), x_state_KF_nominal[:, 1], '--r', label='Nominal KF',markersize=marker_size_for_state_sub_plots, linewidth=1)
    ax12.plot(np.linspace(1, len(x_states), len(x_states)), y_measured[:, 1], '.y', label='Measured state',markersize=marker_size_for_state_sub_plots + 2, linewidth=1)
    ax12.plot(np.linspace(1, len(x_states), len(x_states)), x_true[:, 1], '-k', label='True state',markersize=marker_size_for_state_sub_plots, linewidth=1)
    ax12.plot(np.linspace(1, len(x_states), len(x_states)), x_state_KFTMC[:, 1], '--b', label='KF-TMC', markersize=marker_size_for_state_sub_plots + 2, linewidth=1)
    ax12.plot(np.linspace(1, len(x_states), len(x_states)), x_state_KF_LMC[:, 1], '--g', label='VB-AKF',markersize=marker_size_for_state_sub_plots + 2, linewidth=1)
    ax12.set_title('State 2')
    ax12.set_ylabel('State value')
    # ax12.set_ylim([0, ylim_for_RMSE])
    ax12.legend()

    ax21.plot(np.linspace(1, len(x_states), len(x_states)), x_state_KF_nominal[:, 2], '--r', label='Nominal KF',markersize=marker_size_for_state_sub_plots, linewidth=1)
    ax21.plot(np.linspace(1, len(x_states), len(x_states)), x_true[:, 2], '-k', label='True state',markersize=marker_size_for_state_sub_plots, linewidth=1)
    ax21.plot(np.linspace(1, len(x_states), len(x_states)), x_state_KFTMC[:, 2], '--b', label='KF-TMC', markersize=marker_size_for_state_sub_plots + 2, linewidth=1)
    ax21.plot(np.linspace(1, len(x_states), len(x_states)), x_state_KF_LMC[:, 2], '--g', label='VB-AKF',markersize=marker_size_for_state_sub_plots + 2, linewidth=1)
    ax21.set_title('State 3')
    ax21.set_ylabel('State value')
    # ax21.set_ylim([0, ylim_for_RMSE])
    ax21.legend()

    ax22.plot(np.linspace(1, len(x_states), len(x_states)), x_state_KF_nominal[:, 3], '--r', label='Nominal KF',markersize=marker_size_for_state_sub_plots, linewidth=1)
    ax22.plot(np.linspace(1, len(x_states), len(x_states)), x_true[:, 3], '-k', label='True state',markersize=marker_size_for_state_sub_plots, linewidth=1)
    ax22.plot(np.linspace(1, len(x_states), len(x_states)), x_state_KFTMC[:, 3], '--b', label='KF-TMC', markersize=marker_size_for_state_sub_plots + 2, linewidth=1)
    ax22.plot(np.linspace(1, len(x_states), len(x_states)), x_state_KF_LMC[:, 3], '--g', label='VB-AKF',markersize=marker_size_for_state_sub_plots + 2, linewidth=1)
    ax22.set_title('State 4')
    ax22.set_ylabel('State value')
    # ax22.set_ylim([0, ylim_for_RMSE])
    ax22.legend()
# </editor-fold>






plt.show()
