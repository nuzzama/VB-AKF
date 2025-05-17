import os
import json
import tensorflow as tf
import numpy as np
import gpflow
import gpflow.training.monitor as gpmon
# from gpflow.utilities import print_summary
from models import FullCovarianceRegression, LoglikelTensorBoardTask
from likelihoods import FullCovLikelihood
import matplotlib.pyplot as plt
def main_func_demo2(monte_carlo_sim_no, learning_rate, n_iterations, n_inducing):
    # <editor-fold desc="save dir, seed, print monte-carlo, seed">
    root_savedir = './savedir'
    root_logdir = os.path.join(root_savedir, 'tf_logs')
    if not os.path.exists(root_savedir):
        os.makedirs(root_savedir)
    if not os.path.exists(root_logdir):
        os.makedirs(root_logdir)

    gpflow.reset_default_graph_and_session()
    # tf.random.set_random_seed(1)
    print_monte_carlo_option = False
    # np.random.seed(1)
    # </editor-fold>

    if n_inducing is None:
        n_inducing = 110  # number of inducing points

    if n_iterations is None:
        n_iterations = 10000

    if learning_rate is None:
        learning_rate = 0.01

    k = 200
    # k=100

    no_of_MC_samples_for_state_RMSE = 100


    cov_test_location_type = 'on x_true'
    # cov_test_location_type = 'on outside uniform sq'

    new_measurement_Y_resampled = False
    new_measurement_in_diff_locations = True
    if new_measurement_Y_resampled is True and new_measurement_in_diff_locations is True:
        raise Exception("Only 1 of 'new_measurement_Y_resampled' and 'new_measurement_in_diff_locations' can be True")


    # target_motion_type = 'consnant linear'
    target_motion_type = 'consnant circular'

    true_cov_from = 'from cos/sin matrix'
    # true_cov_from = 'from y noise samples'


    specific_diff_name = 'try,r2-p007,TruCov=sample,newLoc' + ',Prior V-0,Init V-2'


    r2_for_nominal = 0.007


    cov_test_re_run = False
    no_of_cov_test_re_run = 20
    add_Q_status = True


    Q_multiplier = 0.0001
    # Q_value = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,0],[0,0,0,0]]) * Q_multiplier
    # Q_value = np.eye(4)*Q_multiplier
    Q_value = np.array([[Q_multiplier**2, 0, 0, 0], [0, Q_multiplier**2, 0, 0], [0, 0, Q_multiplier, 0], [0, 0, 0, Q_multiplier]])

    origin_at = np.array([4, 4])
    origin_at_for_new_test_loc = np.array([5, 5])
    N_test = 50
    samples_of_meas_for_true_cov_calc = 1000





    minibatch_size = None  # minibatch size for training

    if target_motion_type is not 'consnant circular' and cov_test_location_type is 'on outside uniform sq':
        raise Exception("if target_motion_type is not 'consnant circular' then cov_test_location_type must be 'on x_true'. because I set up the outside version of x_test around the circle only")

    if cov_test_location_type == 'on outside uniform sq' and cov_test_re_run is True:
        raise Exception("cov_test_re_run is only for cov_test_location_type = 'on x_true'. Right now cov_test_location_type = 'on outside uniform sq' and cov_test_re_run = True. \n"
                        "if we need cov_test_location_type = 'on outside uniform sq', then set cov_test_re_run=False")


    qx0_mean_trainable_status = True
    qx0_cov_trainable_status = True
    q_mu_of_V_trainable_status = True
    q_var_of_V_trainable_status = True

    n_samples = 1000  # number of Monte Carlo samples

    # kern = gpflow.kernels.Matern32(1)
    kern = gpflow.kernels.SquaredExponential(2)
    # kern = gpflow.kernels.Cosine(2)
    # kern = sq_exp_modified_kern(2)
    # kern = gpflow.kernels.Constant(10) + gpflow.kernels.SquaredExponential(1)



    # <editor-fold desc="AKF Train data">
    dt = 0.1
    V = 2  # linear veloctity of the target
    r = 2  # radius of the target's circling
    d_theta = (V / r) * dt


    origin_x_value = origin_at[0]
    origin_y_value = origin_at[1]
    init_x_state = origin_x_value + r
    init_y_state = origin_y_value

    origin_x_value = origin_at_for_new_test_loc[0]
    origin_y_value = origin_at_for_new_test_loc[1]
    init_x_state_for_new_test_location = origin_x_value + r
    init_y_state_for_new_test_location = origin_y_value



    def sigma_R(x):
        # alpha = 2*3.14*dt/c
        # K = 24/(alpha**2)
        K = 0.0001
        y_1 = np.sqrt(x[0] ** 2 + x[1] ** 2)
        y_2 = np.arctan(x[1] / x[0])
        a0 = 1
        a1 = 1
        a2 = 1
        g_y1 = a0 + a1 * (a2 - y_1) ** 2
        sigma = K * g_y1 / (np.cos(y_2)) ** 2
        return sigma

    x_true = np.zeros((k + 1, 4))  # x is k+1 amount, since it goes from 0 to k. And 4 states: x,y,x_dot,y_dot
    y_measured = np.zeros((k + 1,2))  # y is k amount, since it goes from 1 to k. y_0 is just dummy. keeping it for indexing ease. And 2 measurements: range and bearing, or, x and y cartesian positions
    y1y2_stored = np.zeros((k + 1, 2))
    if target_motion_type == 'consnant linear':
        x_true[0, :] = [0, 0, V, V]
        A_dynamics = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])  # A for constant linear motion
    if target_motion_type == 'consnant circular':
        # x_true[0, :] = [-r, 0, 0, V] #for clockwise rotation, circle center in origin
        # x_true[0, :] = [r, 0, 0, V]  # for anti-clockwise rotation, circle center in origin
        x_true[0, :] = [init_x_state, init_y_state, 0, V]  # circle center NOT in origin
        A_dynamics = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, np.cos(d_theta), -np.sin(d_theta)],[0, 0, np.sin(d_theta), np.cos(d_theta)]])  # A for anti-clockwise circular motion
        # A_dynamics = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, np.cos(d_theta), np.sin(d_theta)],[0, 0, -np.sin(d_theta), np.cos(d_theta)]])  # A for clockwise circular motion

    Q_value_diag = np.diag(Q_value)
    for l in range(1, k + 1):  # loops from 1 to k
        if add_Q_status is True:
            w = np.random.normal(np.array([0, 0, 0, 0]), np.sqrt(Q_value_diag)).reshape((1, 4))
        else:
            w = np.array([0, 0, 0, 0]).reshape((1, 4))

        x_true[l, :] = np.matmul(A_dynamics, x_true[l - 1, :]) + w
        var_of_epsilon_1 = 0.01
        epsilon1_l = np.random.normal(0, np.sqrt(var_of_epsilon_1))
        r2_l = sigma_R(x_true[l, :])
        epsilon2_l = np.random.normal(0, np.sqrt(r2_l))
        y1_l = np.sqrt(x_true[l, 0] ** 2 + x_true[l, 1] ** 2) + epsilon1_l
        y2_l = np.arctan(x_true[l, 1] / x_true[l, 0]) + epsilon2_l
        y1_l_noiseless = np.sqrt(x_true[l, 0] ** 2 + x_true[l, 1] ** 2)
        y2_l_noiseless = np.arctan(x_true[l, 1] / x_true[l, 0])
        y1y2_stored[l, :] = [y1_l_noiseless, y2_l_noiseless]
        y_measured[l, :] = [y1_l * np.cos(y2_l), y1_l * np.sin(y2_l)]
    # </editor-fold>

    Y_train = y_measured.copy()
    Y_valid = None




    # <editor-fold desc="Test data">
    test_sq_x_values = np.random.uniform(origin_x_value - r, origin_x_value + r, N_test)
    test_sq_y_values = np.random.uniform(origin_y_value - r, origin_y_value + r, N_test)

    x_test_outside_sq = np.transpose(np.array([test_sq_x_values, test_sq_y_values]))

    dummy_velocity_1 = np.reshape(np.repeat(0,N_test),(N_test,1)) # just dummy to meet dimension purpose, not used
    dummy_velocity_2 = np.reshape(np.repeat(0,N_test),(N_test,1)) # just dummy to meet dimension purpose, not used

    x_test_outside_sq = np.hstack((x_test_outside_sq,dummy_velocity_1,dummy_velocity_2))

    if cov_test_location_type == 'on outside uniform sq':
        x_test = x_test_outside_sq.copy()
    if cov_test_location_type == 'on x_true':
        x_test = x_true.copy() #these will not be used, in model.predict they will be replaced by x_test=x_estimated_states when cov_test_location_type='on x_true'
    # </editor-fold>




    # <editor-fold desc="Just to see velocities">
    velocity_point = 1
    init_velo_from_meas_1 = (y_measured[velocity_point + 1, 0] - y_measured[velocity_point, 0]) / dt
    init_velo_from_meas_2 = (y_measured[velocity_point + 1, 1] - y_measured[velocity_point, 1]) / dt
    velocity_true_1 = x_true[velocity_point, 2]
    velocity_true_2 = x_true[velocity_point, 3]
    # </editor-fold>



    # <editor-fold desc="cov_true from (cos, sin)(r1 r2)(cos, sin) matrix">
    r2_stored = np.zeros((1,len(x_true)))
    if true_cov_from == 'from cos/sin matrix':
        k_for_true_cov_calc = len(x_test) - 1  # x_test is k+1 amount, since it goes from 0 to k. That's why k is len(x_test)-1.
        cov_true_on_train_data = np.zeros((k_for_true_cov_calc + 1, 2, 2))  # cov_true_0 is dummy
        for l in range(1, k_for_true_cov_calc + 1):  # KF update loop
            # <editor-fold desc="R_l calc">
            y1 = y1y2_stored[l, 0]
            y2 = y1y2_stored[l, 1]
            var_of_epsilon_1 = 0.01
            r1 = var_of_epsilon_1
            # r2 = sigma_R(x_true[l, :])
            r2 = sigma_R(np.array([y1, y2]))
            r2_stored[0,l] = r2
            interim_mat1 = np.array([[np.cos(y2), -y1 * np.sin(y2)], [np.sin(y2), y1 * np.cos(y2)]])
            interim_mat2 = np.array([[r1, 0], [0, r2]])
            R_l = np.matmul(np.matmul(interim_mat1, interim_mat2), np.transpose(interim_mat1))
            # </editor-fold>
            cov_true_on_train_data[l,:,:] = R_l
    # </editor-fold>

    # plt.plot(np.linspace(1,len(x_true),len(x_true)),r2_stored[0,:])
    # plt.show()


    # <editor-fold desc="Cov_true from y noise samples">

    if true_cov_from == 'from y noise samples':
        k_for_true_cov_calc = len(x_test) - 1  # x_test is k+1 amount, since it goes from 0 to k. That's why k is len(x_test)-1.
        y_measured_for_cov_calc = np.zeros((samples_of_meas_for_true_cov_calc,k_for_true_cov_calc+1,2))  # y is k amount, since it goes from 1 to k. y_0 is just dummy. keeping it for indexing ease. And 2 measurements: range and bearing, or, x and y cartesian positions

        for n in range(samples_of_meas_for_true_cov_calc):
            for l in range(1, k_for_true_cov_calc + 1):  # loops from 1 to k
                var_of_epsilon_1 = 0.01
                epsilon1_l = np.random.normal(0, np.sqrt(var_of_epsilon_1))
                r_l = sigma_R(x_test[l, :])
                epsilon2_l = np.random.normal(0, np.sqrt(r_l))
                y1_l = np.sqrt(x_test[l, 0] ** 2 + x_test[l, 1] ** 2) + epsilon1_l
                y2_l = np.arctan(x_test[l, 1] / x_test[l, 0]) + epsilon2_l
                y_measured_for_cov_calc[n, l, :] = [y1_l * np.cos(y2_l), y1_l * np.sin(y2_l)]

        y_measured_for_cov_calc = y_measured_for_cov_calc - x_test[:, 0:2]
        cov_true_on_train_data = np.zeros((k_for_true_cov_calc + 1, 2, 2))  # cov_true_0 is dummy
        for i in range(1,k_for_true_cov_calc + 1):  # y is k amount, since it goes from 1 to k. y_0 is just dummy. since we calculating cov of y, so we should go from 1 to k. In for loop (1,k+1) means we will go until k.
            cov_true_on_train_data[i, :, :] = np.cov(y_measured_for_cov_calc[:, i, :], rowvar=False)

    # </editor-fold>


    ##########################################
    #####  Build the GPflow model/graph  #####
    ##########################################
    factored = False  # whether or not to use a factored model
    n_factors = None  # number of factors in a factored model (ignored if factored==False)
    heavy_tail = False  # whether to use the heavy-tailed emission distribution
    model_inverse = False  # if True, then use an inverse Wishart process; if False, use a Wishart process
    approx_wishart = True  # if True, use the additive white noise model


    # initilize the variational inducing points
    x_min = x_true.min()
    x_max = x_true.max()
    N_in, X_dim = x_true.shape
    N_in, D = y_measured.shape
    # Zu = x_min + np.random.rand(n_inducing, X_dim) * (x_max - x_min) #when Q will be considered
    Zv = x_min + np.random.rand(n_inducing, D) * (x_max - x_min)

    # follow the gpflow monitor tutorial to log the optimization procedure
    with gpflow.defer_build():

        if not factored:
            likel = FullCovLikelihood(D, n_samples,
                                      heavy_tail=heavy_tail,
                                      model_inverse=model_inverse,
                                      approx_wishart=approx_wishart,
                                      nu=None)

            model = FullCovarianceRegression(target_motion_type, cov_test_location_type, cov_test_re_run, dt, d_theta, add_Q_status, Q_value, x_true, Y_train, kern, likel, Zv, minibatch_size=minibatch_size) # S-VWP

    ####################################################
    #####  GP Monitor tasks for tracking progress  #####
    ####################################################

    # See GP Monitor's demo webpages for more information
    monitor_lag = 10  # how often GP Monitor should display training statistics
    save_lag = 100  # Don't make this too small. Saving is very I/O intensive

    # create the global step parameter tracking the optimization, if using GP monitor's 'create_global_step'
    # helper, this MUST be done before creating GP monitor tasks
    session = model.enquire_session()
    global_step = gpmon.create_global_step(session)

    # create the gpmonitor tasks
    print_task = gpmon.PrintTimingsTask().with_name('print') \
        .with_condition(gpmon.PeriodicIterationCondition(monitor_lag)) \
        .with_exit_condition(True)

    savedir = os.path.join(root_savedir, 'monitor-saves')
    saver_task = gpmon.CheckpointTask(savedir).with_name('saver') \
        .with_condition(gpmon.PeriodicIterationCondition(save_lag)) \
        .with_exit_condition(True)

    file_writer = gpmon.LogdirWriter(root_logdir, session.graph)

    model_tboard_task = gpmon.ModelToTensorBoardTask(file_writer, model).with_name('model_tboard') \
        .with_condition(gpmon.PeriodicIterationCondition(monitor_lag)) \
        .with_exit_condition(True)


    # <editor-fold desc="S-VWP log-likel tensorboard stuffs">
    train_tboard_task = LoglikelTensorBoardTask(file_writer, model, Y_train, minibatch_size=k, summary_name='train_ll').with_name('train_tboard') \
        .with_condition(gpmon.PeriodicIterationCondition(monitor_lag)) \
        .with_exit_condition(True)

    # put the tasks together in a monitor
    # monitor_tasks = [print_task, model_tboard_task, train_tboard_task, saver_task]
    monitor_tasks = [print_task, model_tboard_task, train_tboard_task]


    # add one more if there is a validation set+
    """ Y_valid is currently None """
    if Y_valid is not None:
        test_tboard_task = LoglikelTensorBoardTask(file_writer, model, Y_valid, minibatch_size=k, summary_name='test_ll').with_name('test_tboard') \
            .with_condition(gpmon.PeriodicIterationCondition(monitor_lag)) \
            .with_exit_condition(True)
        monitor_tasks.append(test_tboard_task)
    # </editor-fold>


    ##################################
    #####  Run the optimization  #####
    ##################################

    model.qx0_mean.trainable = qx0_mean_trainable_status
    model.qx0_cov.trainable = qx0_cov_trainable_status
    model.q_mu.trainable = q_mu_of_V_trainable_status
    model.q_sqrt.trainable = q_var_of_V_trainable_status

    # print('\nTrainables: \n\n' , model.read_trainables())

    print('\nlog-likelihood before optimization:')
    print(model.compute_log_likelihood())
    # create the optimizer
    optimiser = gpflow.train.AdamOptimizer(learning_rate)  # create the optimizer



    # run optimization steps in the GP Monitor context
    with gpmon.Monitor(monitor_tasks, session, global_step, print_summary=True) as monitor:
        optimiser.minimize(model, step_callback=monitor, maxiter=n_iterations, global_step=global_step)

    print('log-likelihood after optimization:')
    print(model.compute_log_likelihood())

    #############################
    #####  Prediction  #####
    #############################

    # <editor-fold desc="cov pred on train data">
    sess_cov_pred_on_train_data = model.enquire_session()
    G_mean, G_cov = sess_cov_pred_on_train_data.run([model.G_mean_new, model.G_var_new], feed_dict={model.x_states_new: x_true})
    Ar_scale_diag = model.likelihood.scale_diag.read_value(sess_cov_pred_on_train_data)  # (D,)
    qV_mu = model.q_mu.read_value(sess_cov_pred_on_train_data)
    qV_sqrt = model.q_sqrt.read_value(sess_cov_pred_on_train_data)

    N_test, D, nu = G_mean.shape
    G_samps = np.random.randn(n_samples, N_test, D, nu) * (G_cov ** 0.5) + G_mean  # (n_samples, N_test, D, nu)
    ArG = Ar_scale_diag * G_samps  # (n_samples, N_test, D, nu)
    ArGGAr = np.matmul(ArG, np.transpose(ArG, [0, 1, 3, 2]))  # (n_samples, N_test, D, D)
    cov_predicted_on_train_data = np.average(ArGGAr, axis=0)
    # </editor-fold>





    # <editor-fold desc="new_measurement_in_diff_locations">
    x_true = np.zeros((k + 1, 4))  # x is k+1 amount, since it goes from 0 to k. And 4 states: x,y,x_dot,y_dot
    y_measured = np.zeros((k + 1,2))  # y is k amount, since it goes from 1 to k. y_0 is just dummy. keeping it for indexing ease. And 2 measurements: range and bearing, or, x and y cartesian positions
    y1y2_stored = np.zeros((k + 1, 2))
    y1y2_stored_with_noise = np.zeros((k + 1, 2))
    if target_motion_type == 'consnant linear':
        x_true[0, :] = [0, 0, V, V]
        A_dynamics = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])  # A for constant linear motion
    if target_motion_type == 'consnant circular':
        # x_true[0, :] = [-r, 0, 0, V] #for clockwise rotation, circle center in origin
        # x_true[0, :] = [r, 0, 0, V]  # for anti-clockwise rotation, circle center in origin
        x_true[0, :] = [init_x_state_for_new_test_location, init_y_state_for_new_test_location, 0, V]  # circle center NOT in origin
        A_dynamics = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, np.cos(d_theta), -np.sin(d_theta)],[0, 0, np.sin(d_theta), np.cos(d_theta)]])  # A for anti-clockwise circular motion
        # A_dynamics = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, np.cos(d_theta), np.sin(d_theta)],[0, 0, -np.sin(d_theta), np.cos(d_theta)]])  # A for clockwise circular motion

    Q_value_diag = np.diag(Q_value)
    for l in range(1, k + 1):  # loops from 1 to k
        if add_Q_status is True:
            w = np.random.normal(np.array([0, 0, 0, 0]), np.sqrt(Q_value_diag)).reshape((1, 4))
        else:
            w = np.array([0, 0, 0, 0]).reshape((1, 4))

        x_true[l, :] = np.matmul(A_dynamics, x_true[l - 1, :]) + w
        var_of_epsilon_1 = 0.01
        epsilon1_l = np.random.normal(0, np.sqrt(var_of_epsilon_1))
        r2_l = sigma_R(x_true[l, :])
        epsilon2_l = np.random.normal(0, np.sqrt(r2_l))
        y1_l = np.sqrt(x_true[l, 0] ** 2 + x_true[l, 1] ** 2) + epsilon1_l
        y2_l = np.arctan(x_true[l, 1] / x_true[l, 0]) + epsilon2_l
        y1_l_noiseless = np.sqrt(x_true[l, 0] ** 2 + x_true[l, 1] ** 2)
        y2_l_noiseless = np.arctan(x_true[l, 1] / x_true[l, 0])
        y1y2_stored[l, :] = [y1_l_noiseless, y2_l_noiseless]
        y1y2_stored_with_noise[l, :] = [y1_l, y2_l]
        y_measured[l, :] = [y1_l * np.cos(y2_l), y1_l * np.sin(y2_l)]
    # </editor-fold>


    # <editor-fold desc="true cov calc from (cos, sin)(r1 r2)(cos, sin) matrix">
    r2_stored = np.zeros((1,len(x_true)))
    if true_cov_from == 'from cos/sin matrix':
        k_for_true_cov_calc = len(x_test) - 1  # x_test is k+1 amount, since it goes from 0 to k. That's why k is len(x_test)-1.
        cov_true = np.zeros((k_for_true_cov_calc + 1, 2, 2))  # cov_true_0 is dummy
        for l in range(1, k_for_true_cov_calc + 1):  # KF update loop
            # <editor-fold desc="R_l calc">
            y1 = y1y2_stored[l, 0]
            y2 = y1y2_stored[l, 1]
            var_of_epsilon_1 = 0.01
            r1 = var_of_epsilon_1
            # r2 = sigma_R(x_true[l, :])
            r2 = sigma_R(np.array([y1, y2]))
            r2_stored[0,l]=r2
            interim_mat1 = np.array([[np.cos(y2), -y1 * np.sin(y2)], [np.sin(y2), y1 * np.cos(y2)]])
            interim_mat2 = np.array([[r1, 0], [0, r2]])
            R_l = np.matmul(np.matmul(interim_mat1, interim_mat2), np.transpose(interim_mat1))
            # </editor-fold>
            cov_true[l,:,:] = R_l
    # </editor-fold>
    # plt.figure()
    # plt.plot(np.linspace(1, len(x_true), len(x_true)), r2_stored[0, :])
    # plt.show()

    # <editor-fold desc="Cov_true from y noise samples">

    if true_cov_from == 'from y noise samples':
        k_for_true_cov_calc = len(x_test) - 1  # x_test is k+1 amount, since it goes from 0 to k. That's why k is len(x_test)-1.
        y_measured_for_cov_calc = np.zeros((samples_of_meas_for_true_cov_calc,k_for_true_cov_calc+1,2))  # y is k amount, since it goes from 1 to k. y_0 is just dummy. keeping it for indexing ease. And 2 measurements: range and bearing, or, x and y cartesian positions

        for n in range(samples_of_meas_for_true_cov_calc):
            for l in range(1, k_for_true_cov_calc + 1):  # loops from 1 to k
                var_of_epsilon_1 = 0.01
                epsilon1_l = np.random.normal(0, np.sqrt(var_of_epsilon_1))
                r_l = sigma_R(x_test[l, :])
                epsilon2_l = np.random.normal(0, np.sqrt(r_l))
                y1_l = np.sqrt(x_test[l, 0] ** 2 + x_test[l, 1] ** 2) + epsilon1_l
                y2_l = np.arctan(x_test[l, 1] / x_test[l, 0]) + epsilon2_l
                y_measured_for_cov_calc[n, l, :] = [y1_l * np.cos(y2_l), y1_l * np.sin(y2_l)]

        y_measured_for_cov_calc = y_measured_for_cov_calc - x_test[:, 0:2]
        cov_true = np.zeros((k_for_true_cov_calc + 1, 2, 2))  # cov_true_0 is dummy
        for i in range(1,k_for_true_cov_calc + 1):  # y is k amount, since it goes from 1 to k. y_0 is just dummy. since we calculating cov of y, so we should go from 1 to k. In for loop (1,k+1) means we will go until k.
            cov_true[i, :, :] = np.cov(y_measured_for_cov_calc[:, i, :], rowvar=False)

    # </editor-fold>







    if not factored:

        sess = model.enquire_session()


        #this one will run no matter what. If re-run is applied then the next one will run
        # <editor-fold desc="states propagation (nominal KF)">
        x0_mean = model.qx0_mean.read_value(sess)
        x0_cov = model.qx0_cov.read_value(sess)
        x0_sampled = np.random.normal(x0_mean, x0_cov)
        if new_measurement_in_diff_locations is True:
            # x0_sampled = x_true[0, :]
            x0_sampled[0,0:2] = y_measured[1, :]

        k = model.k_total
        dt = model.dt
        d_theta = model.d_theta
        target_motion_type = model.target_motion_type
        if target_motion_type == 'consnant linear':
            A_dynamics = np.array(
                [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])  # A for constant linear motion
        if target_motion_type == 'consnant circular':
            A_dynamics = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, np.cos(d_theta), -np.sin(d_theta)],
                                   [0, 0, np.sin(d_theta), np.cos(d_theta)]])  # A for anti-clockwise circular motion
            # A_dynamics = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, np.cos(d_theta), np.sin(d_theta)],[0, 0, -np.sin(d_theta), np.cos(d_theta)]])  # A for clockwise circular motion
        x_states = np.zeros((k, 4))
        x_states[0, :] = x0_sampled
        Q_value_diag = np.diag(model.Q_value)
        for l in range(1, k):
            if model.add_Q_status is True:
                w = np.random.normal(np.array([0, 0, 0, 0]), np.sqrt(Q_value_diag)).reshape((1, 4))
            else:
                w = np.array([0, 0, 0, 0]).reshape((1, 4))
            x_states[l, :] = np.matmul(A_dynamics, x_states[l - 1, :]) + w

        if model.cov_test_location_type == 'on x_true':
            x_states_sliced = x_states[1:, :]
            x_test = x_states_sliced
        else:
            x_test = x_test
        # </editor-fold>


        # <editor-fold desc="cov test re-run (not running now)">
        if cov_test_re_run == True:

            for cov_test_re_run_counter in range(no_of_cov_test_re_run):

                # mu, s2 = sess.run([model.G_mean_new, model.G_var_new], feed_dict={model.X_new: X_new})  # (N_new, D, nu), (N_new, D, nu)
                G_mean, G_cov = sess.run([model.G_mean_new, model.G_var_new], feed_dict={model.x_states_new: x_test})
                Ar_scale_diag = model.likelihood.scale_diag.read_value(sess)  # (D,)
                qV_mu = model.q_mu.read_value(sess)
                qV_sqrt = model.q_sqrt.read_value(sess)

                N_test, D, nu = G_mean.shape
                G_samps = np.random.randn(n_samples, N_test, D, nu) * (G_cov ** 0.5) + G_mean  # (n_samples, N_test, D, nu)
                ArG = Ar_scale_diag * G_samps  # (n_samples, N_test, D, nu)
                ArGGAr = np.matmul(ArG, np.transpose(ArG, [0, 1, 3, 2]))  # (n_samples, N_test, D, D)
                cov_predicted = np.average(ArGGAr, axis=0)


                # <editor-fold desc="(KF-LMC) regular Kalman filter with learned R">
                """KF-LMC starts here"""
                C_meas_eq = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
                x_state_KF_LMC = np.zeros((len(x_states), 4))
                x_state_KF_LMC[0, :] = x0_sampled
                P_KFLMC = np.zeros((len(x_states), 4, 4))

                for l in range(1, len(x_states)):  # KF update loop
                    if model.add_Q_status is True:
                        w = np.random.normal(np.array([0, 0, 0, 0]), np.sqrt(Q_value_diag)).reshape((1, 4))
                    else:
                        w = np.array([0, 0, 0, 0]).reshape((1, 4))

                    x_hat_predicted_state = np.matmul(A_dynamics, x_state_KF_LMC[l - 1, :]) + w
                    x_hat_predicted_state = x_hat_predicted_state.transpose()
                    P_hat_predicted_P = np.matmul(np.matmul(A_dynamics, P_KFLMC[l - 1, :, :]),np.transpose(A_dynamics)) + Q_value_diag

                    try:
                        R_l = cov_predicted[l - 1, :,:]  # although indexing stars from (l-1) it actually is k-th R, because R_l stars from k=1 to k. But since python index must start from 0 so I need to use l-1.
                    except:
                        R_l = np.zeros((2, 2))
                    y_tilde_innovation_residual = y_measured[l, :].reshape(2, 1) - np.matmul(C_meas_eq,x_hat_predicted_state)
                    S_innovation_cov = np.matmul(np.matmul(C_meas_eq, P_hat_predicted_P), np.transpose(C_meas_eq)) + R_l

                    K_kalman_gain = np.matmul(np.matmul(P_hat_predicted_P, np.transpose(C_meas_eq)),np.linalg.inv(S_innovation_cov))

                    x_updated_states = x_hat_predicted_state + np.matmul(K_kalman_gain, y_tilde_innovation_residual)
                    P_updated_cov = np.matmul(np.eye(4) - np.matmul(K_kalman_gain, C_meas_eq), P_hat_predicted_P)

                    x_state_KF_LMC[l, :] = x_updated_states.transpose()
                    P_KFLMC[l, :] = P_updated_cov


                x_state_KF_LMC_sliced = x_state_KF_LMC[1:,]
                x_test = x_state_KF_LMC_sliced.copy()
                # x_states = x_state_KF_LMC.copy() #commenting this out so that x_states is the unsmoothed states. we are passing the unsmoothed states because the unsmoothed states will be then passed in the preds. In results_read.py we will do KF-LMC states from the unsmoothed states.

            # </editor-fold>
        # </editor-fold>



        # <editor-fold desc="KF-LMC (VB-AKF)">
        C_meas_eq = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        x_state_KF_LMC = np.zeros((len(x_states), 4))
        x_state_KF_LMC[0, :] = x0_sampled
        P_KFLMC = np.zeros((len(x_states), 4, 4))
        cov_predicted = np.zeros((k_for_true_cov_calc+1, 2, 2))  # cov_pred_0 is dummy


        for l in range(1, len(x_states)):  # KF update loop

            if model.add_Q_status is True:
                w = np.random.normal(np.array([0, 0, 0, 0]), np.sqrt(Q_value_diag)).reshape((1, 4))
            else:
                w = np.array([0, 0, 0, 0]).reshape((1, 4))

            x_hat_predicted_state = np.matmul(A_dynamics, x_state_KF_LMC[l - 1, :]) + w
            x_hat_predicted_state = x_hat_predicted_state.transpose()
            P_hat_predicted_P = np.matmul(np.matmul(A_dynamics, P_KFLMC[l - 1, :, :]),np.transpose(A_dynamics)) + Q_value


            G_mean, G_cov = sess.run([model.G_mean_new, model.G_var_new],feed_dict={model.x_states_new: x_hat_predicted_state.transpose()}) #transposing because feed dict takes 1,4 and not 4,1
            Ar_scale_diag = model.likelihood.scale_diag.read_value(sess)  # (D,)
            N_test, D, nu = G_mean.shape
            G_samps = np.random.randn(n_samples, N_test, D, nu) * (G_cov ** 0.5) + G_mean  # (n_samples, N_test, D, nu)
            ArG = Ar_scale_diag * G_samps  # (n_samples, N_test, D, nu)
            ArGGAr = np.matmul(ArG, np.transpose(ArG, [0, 1, 3, 2]))  # (n_samples, N_test, D, D)
            cov_predicted[l,:,:] = np.average(ArGGAr, axis=0)

            try:
                R_l = cov_predicted[l,:,:] # although indexing stars from (l-1) it actually is k-th R, because R_l stars from k=1 to k. But since python index must start from 0 so I need to use l-1.
            except:
                R_l = np.zeros((2, 2))

            y_tilde_innovation_residual = y_measured[l, :].reshape(2, 1) - np.matmul(C_meas_eq,x_hat_predicted_state)
            S_innovation_cov = np.matmul(np.matmul(C_meas_eq, P_hat_predicted_P), np.transpose(C_meas_eq)) + R_l

            K_kalman_gain = np.matmul(np.matmul(P_hat_predicted_P, np.transpose(C_meas_eq)),np.linalg.inv(S_innovation_cov))

            x_updated_states = x_hat_predicted_state + np.matmul(K_kalman_gain, y_tilde_innovation_residual)
            P_updated_cov = np.matmul(np.eye(4) - np.matmul(K_kalman_gain, C_meas_eq), P_hat_predicted_P)

            x_state_KF_LMC[l, :] = x_updated_states.transpose()
            P_KFLMC[l, :] = P_updated_cov
        # </editor-fold>





        # <editor-fold desc="Monte Carlo - KF-LMC (VB-AKF)">
        C_meas_eq = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        x_state_KF_LMC_Monte_Carlo = np.zeros((no_of_MC_samples_for_state_RMSE, len(x_states), 4))
        P_KFLMC_Monte_Carlo = np.zeros((no_of_MC_samples_for_state_RMSE, len(x_states), 4, 4))
        x_state_KF_TMC_monte_carlo = np.zeros((no_of_MC_samples_for_state_RMSE, len(x_states), 4))
        P_KF_TMC_monte_carlo = np.zeros((no_of_MC_samples_for_state_RMSE, len(x_states), 4, 4))
        x_state_NOMINAL_monte_carlo = np.zeros((no_of_MC_samples_for_state_RMSE, len(x_states), 4))
        P_KF_NOMINAL_monte_carlo = np.zeros((no_of_MC_samples_for_state_RMSE, len(x_states), 4, 4))

        cov_predicted_Monte_Carlo = np.zeros((no_of_MC_samples_for_state_RMSE, k_for_true_cov_calc+1, 2, 2))  # cov_pred_0 is dummy


        x_true_Monte_Carlo = np.zeros((no_of_MC_samples_for_state_RMSE, k+1, 4))  # x is k+1 amount, since it goes from 0 to k. And 4 states: x,y,x_dot,y_dot
        y_measured_Monte_Carlo = np.zeros((no_of_MC_samples_for_state_RMSE, k+1, 2))  # y is k amount, since it goes from 1 to k. y_0 is just dummy. keeping it for indexing ease. And 2 measurements: range and bearing, or, x and y cartesian positions

        for n_monte in range(no_of_MC_samples_for_state_RMSE):
            x_state_KF_LMC_Monte_Carlo[n_monte, 0, :] = x0_sampled
            x_state_NOMINAL_monte_carlo[n_monte, 0, :] = x0_sampled
            x_state_KF_TMC_monte_carlo [n_monte, 0, :] = x0_sampled
            print('MC now:', n_monte)


            # <editor-fold desc="new_measurement_in_diff_locations">
            if target_motion_type == 'consnant linear':
                x_true_Monte_Carlo[n_monte, 0, :] = [0, 0, V, V]
                A_dynamics = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])  # A for constant linear motion
            if target_motion_type == 'consnant circular':
                # x_true[0, :] = [-r, 0, 0, V] #for clockwise rotation, circle center in origin
                # x_true[0, :] = [r, 0, 0, V]  # for anti-clockwise rotation, circle center in origin
                x_true_Monte_Carlo[n_monte, 0, :] = [init_x_state_for_new_test_location, init_y_state_for_new_test_location, 0,V]  # circle center NOT in origin
                A_dynamics = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, np.cos(d_theta), -np.sin(d_theta)],[0, 0, np.sin(d_theta),np.cos(d_theta)]])  # A for anti-clockwise circular motion
                # A_dynamics = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, np.cos(d_theta), np.sin(d_theta)],[0, 0, -np.sin(d_theta), np.cos(d_theta)]])  # A for clockwise circular motion

            Q_value_diag = np.diag(Q_value)
            for l in range(1, len(x_states)):  # loops from 1 to k
                # print('l now:',l)
                if add_Q_status is True:
                    w = np.random.normal(np.array([0, 0, 0, 0]), np.sqrt(Q_value_diag)).reshape((1, 4))
                else:
                    w = np.array([0, 0, 0, 0]).reshape((1, 4))

                x_true_Monte_Carlo[n_monte, l, :] = np.matmul(A_dynamics, x_true_Monte_Carlo[n_monte, l - 1, :]) + w
                var_of_epsilon_1 = 0.01
                epsilon1_l = np.random.normal(0, np.sqrt(var_of_epsilon_1))
                r2_l = sigma_R(x_true_Monte_Carlo[n_monte, l, :])
                epsilon2_l = np.random.normal(0, np.sqrt(r2_l))
                y1_l = np.sqrt(x_true[l, 0] ** 2 + x_true[l, 1] ** 2) + epsilon1_l
                y2_l = np.arctan(x_true[l, 1] / x_true[l, 0]) + epsilon2_l
                y_measured_Monte_Carlo[n_monte, l, :] = [y1_l * np.cos(y2_l), y1_l * np.sin(y2_l)]
            # </editor-fold>




            for l in range(1, len(x_states)):  # KF LMC (VB-AKF) and TMC loop

                # <editor-fold desc="Nominal KF monte carlo">
                """Nominal KF monte carlo here"""
                x_hat_predicted_state = np.matmul(A_dynamics, x_state_NOMINAL_monte_carlo[n_monte, l-1, :]) + w
                x_hat_predicted_state = x_hat_predicted_state.transpose()
                P_hat_predicted_P = np.matmul(np.matmul(A_dynamics, P_KF_NOMINAL_monte_carlo[n_monte, l-1, :, :]),np.transpose(A_dynamics)) + Q_value

                y1 = y1y2_stored_with_noise[l, 0]
                y2 = y1y2_stored_with_noise[l, 1]
                var_of_epsilon_1 = 0.01
                r1 = var_of_epsilon_1
                # r2 = sigma_R(x_true[l, :])
                r2 = r2_for_nominal
                interim_mat1 = np.array([[np.cos(y2), -y1 * np.sin(y2)], [np.sin(y2), y1 * np.cos(y2)]])
                interim_mat2 = np.array([[r1, 0], [0, r2]])
                R_l = np.matmul(np.matmul(interim_mat1, interim_mat2), np.transpose(interim_mat1))

                y_tilde_innovation_residual = y_measured_Monte_Carlo[n_monte, l, :].reshape(2, 1) - np.matmul(C_meas_eq,x_hat_predicted_state)
                S_innovation_cov = np.matmul(np.matmul(C_meas_eq, P_hat_predicted_P), np.transpose(C_meas_eq)) + R_l

                K_kalman_gain = np.matmul(np.matmul(P_hat_predicted_P, np.transpose(C_meas_eq)),np.linalg.inv(S_innovation_cov))

                x_updated_states = x_hat_predicted_state + np.matmul(K_kalman_gain, y_tilde_innovation_residual)
                P_updated_cov = np.matmul(np.eye(4) - np.matmul(K_kalman_gain, C_meas_eq), P_hat_predicted_P)

                x_state_NOMINAL_monte_carlo[n_monte, l, :] = x_updated_states.transpose()
                P_KF_NOMINAL_monte_carlo[n_monte, l, :, :] = P_updated_cov
                # </editor-fold>


                # <editor-fold desc="KF-LMC (VB-AKF) Monte-Carlo">
                """KF-LMC monte carlo"""

                if model.add_Q_status is True:
                    w = np.random.normal(np.array([0, 0, 0, 0]), np.sqrt(Q_value_diag)).reshape((1, 4))
                else:
                    w = np.array([0, 0, 0, 0]).reshape((1, 4))

                x_hat_predicted_state = np.matmul(A_dynamics, x_state_KF_LMC_Monte_Carlo[n_monte, l-1, :]) + w
                x_hat_predicted_state = x_hat_predicted_state.transpose()
                P_hat_predicted_P = np.matmul(np.matmul(A_dynamics, P_KFLMC_Monte_Carlo[n_monte, l-1, :, :]),np.transpose(A_dynamics)) + Q_value


                G_mean, G_cov = sess.run([model.G_mean_new, model.G_var_new],feed_dict={model.x_states_new: x_hat_predicted_state.transpose()}) #transposing because feed dict takes 1,4 and not 4,1
                Ar_scale_diag = model.likelihood.scale_diag.read_value(sess)  # (D,)
                N_test, D, nu = G_mean.shape
                G_samps = np.random.randn(n_samples, N_test, D, nu) * (G_cov ** 0.5) + G_mean  # (n_samples, N_test, D, nu)
                ArG = Ar_scale_diag * G_samps  # (n_samples, N_test, D, nu)
                ArGGAr = np.matmul(ArG, np.transpose(ArG, [0, 1, 3, 2]))  # (n_samples, N_test, D, D)
                cov_predicted_Monte_Carlo[n_monte, l, :, :] = np.average(ArGGAr, axis=0)

                try:
                    R_l = cov_predicted_Monte_Carlo[n_monte, l, :, :] # although indexing stars from (l-1) it actually is k-th R, because R_l stars from k=1 to k. But since python index must start from 0 so I need to use l-1.
                except:
                    R_l = np.zeros((2, 2))

                y_tilde_innovation_residual = y_measured_Monte_Carlo[n_monte, l, :].reshape(2, 1) - np.matmul(C_meas_eq,x_hat_predicted_state)
                S_innovation_cov = np.matmul(np.matmul(C_meas_eq, P_hat_predicted_P), np.transpose(C_meas_eq)) + R_l

                K_kalman_gain = np.matmul(np.matmul(P_hat_predicted_P, np.transpose(C_meas_eq)),np.linalg.inv(S_innovation_cov))

                x_updated_states = x_hat_predicted_state + np.matmul(K_kalman_gain, y_tilde_innovation_residual)
                P_updated_cov = np.matmul(np.eye(4) - np.matmul(K_kalman_gain, C_meas_eq), P_hat_predicted_P)

                x_state_KF_LMC_Monte_Carlo[n_monte, l, :] = x_updated_states.transpose()
                P_KFLMC_Monte_Carlo[n_monte, l, :] = P_updated_cov
                # </editor-fold>


                # <editor-fold desc="KF-TMC Monte_carlo">
                """KF-TMC monte carlo"""

                if model.add_Q_status is True:
                    w = np.random.normal(np.array([0, 0, 0, 0]), np.sqrt(Q_value_diag)).reshape((1, 4))
                else:
                    w = np.array([0, 0, 0, 0]).reshape((1, 4))

                x_hat_predicted_state = np.matmul(A_dynamics, x_state_KF_TMC_monte_carlo[n_monte, l - 1, :]) + w
                x_hat_predicted_state = x_hat_predicted_state.transpose()
                P_hat_predicted_P = np.matmul(np.matmul(A_dynamics, P_KF_TMC_monte_carlo[n_monte, l - 1, :, :]),np.transpose(A_dynamics)) + Q_value

                R_l = cov_true[l, :, :]

                y_tilde_innovation_residual = y_measured_Monte_Carlo[n_monte, l, :].reshape(2, 1) - np.matmul(C_meas_eq,x_hat_predicted_state)
                S_innovation_cov = np.matmul(np.matmul(C_meas_eq, P_hat_predicted_P), np.transpose(C_meas_eq)) + R_l

                K_kalman_gain = np.matmul(np.matmul(P_hat_predicted_P, np.transpose(C_meas_eq)),np.linalg.inv(S_innovation_cov))

                x_updated_states = x_hat_predicted_state + np.matmul(K_kalman_gain, y_tilde_innovation_residual)
                P_updated_cov = np.matmul(np.eye(4) - np.matmul(K_kalman_gain, C_meas_eq), P_hat_predicted_P)

                x_state_KF_TMC_monte_carlo[n_monte, l, :] = x_updated_states.transpose()
                P_KF_TMC_monte_carlo[n_monte, l, :] = P_updated_cov
                # </editor-fold>
        # </editor-fold>



        cov_predicted = cov_predicted[1:, :,:]  # slicing since this will be passed in the result read, and that code has this set up



        preds = model.predict(x_test, x_states) # the states here in x_states are the unsmoothed states. we are passing unsmoothed states because the unsmoothed states will then be passed in the preds. In results_read.py we will do KF-LMC states from the unsmoothed states.
        # G_mean, G_cov = preds['G_mean'], preds['G_cov']  # (N_test, D, nu)
        # Ar_scale_diag = preds['Ar_scale_diag'][:, None]
        # x0_mean = preds['x0_mean']
        # x0_cov = preds['x0_cov']
        qV_mu = preds['qV_mu']
        qV_sqrt = preds['qV_sqrt']
        # x_states = preds['x_states']
        logP = preds['logP']
        # N_test, D, nu = G_mean.shape
        # G_samps = np.random.randn(n_samples, N_test, D, nu) * (G_cov ** 0.5) + G_mean  # (n_samples, N_test, D, nu)
        # ArG = Ar_scale_diag * G_samps  # (n_samples, N_test, D, nu)
        # ArGGAr = np.matmul(ArG, np.transpose(ArG, [0, 1, 3, 2]))  # (n_samples, N_test, D, D)
        # cov_predicted = np.average(ArGGAr, axis=0)

        if not approx_wishart:
            additive_part = np.diag(np.ones(D) * 1e-5)[None, :, :]  # at least add some small jitter, like with normal GPs
        else:
            sigma2inv_conc = preds['sigma2inv_conc']  # (D,)
            sigma2inv_rate = preds['sigma2inv_rate']
            sigma2inv_samps = np.random.gamma(sigma2inv_conc, scale=1.0 / sigma2inv_rate, size=[n_samples, D])  # (n_samples, D)

            if model_inverse:
                # inverse Wishart process variants
                additive_part = np.apply_along_axis(np.diag, axis=1, arr=sigma2inv_samps)  # (n_samples, D, D)
            else:
                # Wishart process variants
                additive_part = np.apply_along_axis(np.diag, axis=1, arr=sigma2inv_samps ** -1.0)  # (n_samples, D, D)


        preds.update(dict(x_true=x_true))



    """
    Training and prediction upto here
    Printing and saving in file after here 
    """




    npz_file_name = './savedir/' + specific_diff_name + ',mc-' + str(monte_carlo_sim_no) + ',k-' + str(k-1) + ',lr-' + str(learning_rate) + ',i-' + str(n_iterations) + '.npz'


    # <editor-fold desc="Not saving JSON - save predictions in json format">
    for key in preds.keys():
        if isinstance(preds[key], np.ndarray):
            preds[key] = preds[key].tolist()


    np.savez(npz_file_name, x_states=x_states, x_true=x_true, x_state_KF_LMC=x_state_KF_LMC, P_KFLMC=P_KFLMC, G_mean=G_mean, G_cov=G_cov,
             x0_mean=x0_mean, x0_cov=x0_cov, cov_predicted=cov_predicted, no_of_cov_test_re_run=no_of_cov_test_re_run, qV_mu=qV_mu, qV_sqrt=qV_sqrt,
             Ar_scale_diag=Ar_scale_diag, logP=logP, y_measured=y_measured, y1y2_stored=y1y2_stored, y1y2_stored_with_noise= y1y2_stored_with_noise,
             target_motion_type=target_motion_type, V=V, r=r,
             N_test=N_test, x_test=x_test, cov_true=cov_true, cov_test_location_type=cov_test_location_type ,add_Q_status=add_Q_status,
             Q_value=Q_value, cov_test_re_run=cov_test_re_run, no_of_MC_samples_for_state_RMSE=no_of_MC_samples_for_state_RMSE,
             x_state_KF_LMC_Monte_Carlo=x_state_KF_LMC_Monte_Carlo, P_KFLMC_Monte_Carlo=P_KFLMC_Monte_Carlo,
             x_state_KF_TMC_monte_carlo=x_state_KF_TMC_monte_carlo, P_KF_TMC_monte_carlo=P_KF_TMC_monte_carlo,
             y_measured_Monte_Carlo=y_measured_Monte_Carlo, cov_true_on_train_data=cov_true_on_train_data,
             cov_predicted_on_train_data=cov_predicted_on_train_data, x_state_NOMINAL_monte_carlo=x_state_NOMINAL_monte_carlo)





    print('\nsaved npz file name: ', npz_file_name)

    print('\nlogP optimized: ', preds['logP'])



if __name__ == '__main__':
    lrs = None
    n_iterations = None
    inducing_points_array = None
    # lrs = np.array([0.01, 0.05, 0.1, 0.15, 0.2 , 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
    # lrs = np.array([0.0001, 0.0005])
    # lrs = np.array([0.01, 0.05, 0.15, 0.2])
    # lrs = np.array([0.25, 0.3, 0.35, 0.4])

    # n_iterations = np.array([20000,14000, 16000])
    # n_iterations = np.array([10000,14000,12000])

    # inducing_points_array = np.array([50, 100, 110, 120, 150, 200, 250])


    if lrs is not None:
        for lr in lrs:
            main_func_demo2(None, lr, None, None) #if this code run then No monte-carlo. And no Mn goes from here. Mn is defined inside.

    elif n_iterations is not None:
        for iter_no in n_iterations:
            main_func_demo2(None, None, iter_no, None) #if this code run then No monte-carlo. And no Mn goes from here. Mn is defined inside.

    elif inducing_points_array is not None:
        for n_inducing in inducing_points_array:
            main_func_demo2(None, None, None, n_inducing)

    else:
        main_func_demo2(None, None, None, None)


    print('\n\nlrs:\n', lrs)
    print('\nIters:\n', n_iterations)
    print('\nInducing points:\n', inducing_points_array)



    plt.plot(np.linspace(1,10),np.linspace(1,10))
    plt.show()