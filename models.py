import gpflow.transforms
import numpy as np
from scipy.special import logsumexp
import tensorflow as tf
from gpflow.params import Minibatch, Parameter
from gpflow.params import DataHolder

from gpflow import settings, kullback_leiblers, transforms
from gpflow.decors import params_as_tensors, params_as_tensors_for
from gpflow.models.svgp import SVGP
import gpflow.training.monitor as gpmon
from gpflow.conditionals import _expand_independent_outputs

tf.random.set_random_seed(1)
np.random.seed(1)

class DynamicCovarianceRegression(SVGP):
    """
    Effectively a helper wrapping call to SVGP.
    """
    def __init__(self, target_motion_type, cov_test_location_type, cov_test_re_run, dt, d_theta, add_Q_status, Q_value, x_true, Y, kern, likelihood, Z_v, minibatch_size=None, whiten=True):
        """

        :param X:
        :param Y:
        :param kern:
        :param likelihood:
        :param Z_v:
        :param minibatch_size:
        :param whiten:
        """
        cov_dim = likelihood.cov_dim
        nu = likelihood.nu

        Y_mean = np.mean(Y, axis=0)  # (D,)
        X = x_true #providing this just because I have to provide a X for SVGP. This will not be used. The estimated X by the KF will be used in the optimization process.
        super().__init__(X, Y, kern, likelihood,
                         feat=None,
                         mean_function=None,
                         num_latent=cov_dim * nu,
                         q_diag=False,
                         whiten=whiten,
                         minibatch_size=minibatch_size,
                         Z=Z_v,
                         name='SVGP')  # must provide a name space when delaying build for GPflow opt functionality


        self.minibatch_size = minibatch_size
        self.cov_test_location_type = cov_test_location_type
        self.cov_test_re_run = cov_test_re_run
        self.add_Q_status = add_Q_status
        self.Q_value = Q_value



        # <editor-fold desc="q(x0) variational dist initialization">
        v_init = 2
        v_prior = 0
        qx0_mean_init = np.array([Y[1,0],Y[1,1],v_init,v_init]).reshape((1,4))
        self.qx0_mean = Parameter(qx0_mean_init, dtype=settings.float_type)
        qx0_cov_param_init = 0.1 * np.ones((1, 4)) #only need to learn 4 variances for 4 initial states.
        self.qx0_cov = Parameter(qx0_cov_param_init, dtype=settings.float_type, transform=transforms.positive)
        # </editor-fold>

        # <editor-fold desc="prior P(x0)">
        init = Y[1,:]
        self.Px0_mean = np.array([Y[1,0], Y[1,1],v_prior,v_prior]).reshape((1,4))
        self.Px0_cov = 0.1 * np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
        # </editor-fold>

        # self.Zu = Parameter(Zu, dtype=settings.float_type) #when Q will be considered

        self.dt = dt
        self.d_theta = d_theta
        self.target_motion_type = target_motion_type

        self.k_total = len(Y)


        self.compile()  # if I don't compile now then there's a weird error when trying to construct the prediction
        self.Y_mean = Y_mean  # used when constructing the predictions
        self.construct_predictive_density()


    @params_as_tensors
    def construct_predictive_density(self):
        tf.random.set_random_seed(1)
        D = self.likelihood.D

        # self.Cx_new = tf.placeholder(dtype=settings.float_type, shape=[None, 2, 2])
        # self.Mn_new = tf.placeholder(dtype=settings.float_type, shape=[None, 2, 2])
        # self.X_new = tf.placeholder(dtype=settings.float_type, shape=[None, 2])

        self.Y_new = tf.placeholder(dtype=settings.float_type, shape=[None, D])
        self.x_states_new = tf.placeholder(dtype=settings.float_type, shape=[None, 4]) # because we have 4 states
        self.n_samples = tf.placeholder(dtype=settings.int_type, shape=[])
        Y_new = self.Y_new
        x_states = self.x_states_new

        G_mean, G_var = self._build_predict(x_states)

        N_new = tf.shape(G_mean)[0]
        cov_dim = self.likelihood.cov_dim
        self.G_mean_new = tf.reshape(G_mean, [N_new, cov_dim, -1])  # (N_new, cov_dim, nu)
        self.G_var_new = tf.reshape(G_var, [N_new, cov_dim, -1])

        nu = tf.shape(self.G_mean_new)[-1]
        G_samps = tf.random.normal([self.n_samples, N_new, cov_dim, nu], dtype=settings.float_type) \
                  * (self.G_var_new ** 0.5) + self.G_mean_new

        log_det_cov, yt_inv_y = self.likelihood.make_gaussian_components(G_samps, Y_new, x_states)

        # compute the Gaussian metrics
        D_ = tf.cast(self.likelihood.D, settings.float_type)
        self.logp_gauss_data = - 0.5 * yt_inv_y
        self.logp_gauss = - 0.5 * D_ * np.log(2 * np.pi) - 0.5 * log_det_cov + self.logp_gauss_data  # (S, N)

        if not self.likelihood.heavy_tail:
            self.logp_data = self.logp_gauss_data
            self.logp = self.logp_gauss
        else:
            dof = tf.cast(self.likelihood.dof, settings.float_type)
            self.logp_data = - 0.5 * (dof + D_) * tf.log(1.0 + yt_inv_y / dof)
            self.logp = tf.lgamma(0.5 * (dof + D_)) - tf.lgamma(0.5 * dof) - 0.5 * D_ * tf.log(np.pi * dof) \
                        - 0.5 * log_det_cov + self.logp_data  # (S, N)

    def mcmc_predict_density(self, Y_new, x_states_new, n_samples=100):
        sess = self.enquire_session()
        outputs = sess.run([self.logp, self.logp_data, self.logp_gauss, self.logp_gauss_data],
                           feed_dict={self.Y_new: Y_new, self.x_states_new: x_states_new, self.n_samples: n_samples})
        log_S = np.log(n_samples)
        return tuple(map(lambda x: logsumexp(x, axis=0) - log_S, outputs))

class FullCovarianceRegression(DynamicCovarianceRegression):

    @params_as_tensors
    def build_prior_KL(self):
        if self.whiten:
            K = None
        else:
            Z_mu = self.feature.Z
            Z_len, Z_dim = Z_mu.get_shape().as_list()
            Z_cov = 0.0000001 * tf.eye(2, batch_shape=[Z_len])
            Z_cov = tf.cast(Z_cov, settings.float_type)
            K = self.Kuu(self.feature, Z_mu, Z_cov, self.kern, jitter=settings.numerics.jitter_level)  # (P x) x M x M

        KL = kullback_leiblers.gauss_kl(self.q_mu, self.q_sqrt, K)

        if self.likelihood.approx_wishart:
            p_dist = tf.distributions.Gamma(self.likelihood.p_sigma2inv_conc, rate=self.likelihood.p_sigma2inv_rate)
            q_dist = tf.distributions.Gamma(self.likelihood.q_sigma2inv_rate, rate=self.likelihood.q_sigma2inv_rate)
            self.KL_gamma = tf.reduce_sum(q_dist.kl_divergence(p_dist))
            KL += self.KL_gamma
        return KL


    def mcmc_predict_matrix(self, X_new, n_samples):        #no use OF THIS
        # sample of x here, maybe??
        params, Beta = self.predict(X_new)
        mu, s2 = params['mu'], params['s2']
        scale_diag = params['scale_diag']

        N_new, D, nu = mu.shape
        F_samps = np.random.randn(n_samples, N_new, D, nu) * np.sqrt(s2) + mu  # (n_samples, N_new, D, nu)
        AF = scale_diag[:, None] * F_samps  # (n_samples, N_new, D, nu)
        affa = np.matmul(AF, np.transpose(AF, [0, 1, 3, 2]))  # (n_samples, N_new, D, D)

        if self.likelihood.approx_wishart:
            sigma2inv_conc = params['sigma2inv_conc']
            sigma2inv_rate = params['sigma2inv_rate']
            sigma2inv_samps = np.random.gamma(sigma2inv_conc, scale=1.0 / sigma2inv_rate, size=[n_samples, D])  # (n_samples, D)

            if self.likelihood.model_inverse:
                lam = np.apply_along_axis(np.diag, axis=0, arr=sigma2inv_samps)  # (n_samples, D, D)
            else:
                lam = np.apply_along_axis(np.diag, axis=0, arr=sigma2inv_samps ** -1.0)
            affa = affa + lam[:, None, :, :]  # (n_samples, N_new, D, D)
        return affa

    def predict(self, x_test, x_states):

        np.random.seed(1)
        sess = self.enquire_session()
        x0_mean = self.qx0_mean.read_value(sess)
        x0_cov = self.qx0_cov.read_value(sess)


        # mu, s2 = sess.run([self.G_mean_new, self.G_var_new], feed_dict={self.X_new: X_new})  # (N_new, D, nu), (N_new, D, nu)
        G_mean, G_cov = sess.run([self.G_mean_new, self.G_var_new], feed_dict={self.x_states_new: x_test})
        Ar_scale_diag = self.likelihood.scale_diag.read_value(sess)  # (D,)
        qV_mu = self.q_mu.read_value(sess)
        qV_sqrt = self.q_sqrt.read_value(sess)



        params = dict(x_states=x_states, G_mean=G_mean, G_cov=G_cov, Ar_scale_diag=Ar_scale_diag, qV_mu=qV_mu, qV_sqrt=qV_sqrt,
                      logP=self.compute_log_likelihood(), x0_mean=x0_mean, x0_cov=x0_cov)

        if self.likelihood.approx_wishart:
            sigma2inv_conc = self.likelihood.q_sigma2inv_conc.read_value(sess)  # (D,)
            sigma2inv_rate = self.likelihood.q_sigma2inv_rate.read_value(sess)
            params.update(dict(sigma2inv_conc=sigma2inv_conc, sigma2inv_rate=sigma2inv_rate))

        return params


    @params_as_tensors
    def _build_likelihood(self): #overwrie the SVGP method #ELBO
        """
        This gives a variational bound on the model likelihood.
        """
        KL_for_qV_given_pV = self.build_prior_KL()
        mean_for_KL_qx0_Px0 = tf.transpose(self.qx0_mean - self.Px0_mean)
        qx0_cov_for_KL = tf.transpose(self.qx0_cov)
        Px0_cov = tf.constant(self.Px0_cov)
        KL_for_qx0_given_Px0 = kullback_leiblers.gauss_kl(mean_for_KL_qx0_Px0, qx0_cov_for_KL**0.5, Px0_cov)


        tf.random.set_random_seed(1)

        # <editor-fold desc="states propagation">
        k_total = self.k_total
        qx0_mean_param_tensor = self.qx0_mean
        qx0_cov_param_tensor = self.qx0_cov

        x0_sampled = tf.random_normal((1,4), dtype=settings.float_type) * (qx0_cov_param_tensor ** 0.5) + qx0_mean_param_tensor #this tensor's output shape is (1,4) because it's only x0. x0's length is 1 with 4 states.

        dt = self.dt
        d_theta = self.d_theta
        target_motion_type = self.target_motion_type
        if target_motion_type == 'consnant linear':
            A_dynamics = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])  # A for constant linear motion
        if target_motion_type == 'consnant circular':
            A_dynamics = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, np.cos(d_theta), -np.sin(d_theta)],[0, 0, np.sin(d_theta), np.cos(d_theta)]])  # A for anti-clockwise circular motion
            # A_dynamics = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, np.cos(d_theta), np.sin(d_theta)],[0, 0, -np.sin(d_theta), np.cos(d_theta)]])  # A for clockwise circular motion

        x_states = tf.TensorArray(settings.float_type, size=0, dynamic_size=True, clear_after_read=False)
        x_states = x_states.write(0, x0_sampled)
        m_x_states = tf.TensorArray(settings.float_type, size=0, dynamic_size=True, clear_after_read=False)
        m_x_states = m_x_states.write(0, qx0_mean_param_tensor)
        P_x_states = tf.TensorArray(settings.float_type, size=0, dynamic_size=True, clear_after_read=False)
        P_x_states = P_x_states.write(0, qx0_cov_param_tensor)

        Q_value_diag = np.diag(self.Q_value)

        k = tf.constant(0)
        def loop_cond(k_now, dummy, dummy2, dummy3):
            return k_now < k_total-1 #for some reason I have to do the -1, otherwise I have 1 more states than there actually is.

        def loop_body(k_now, states, m_x_states, P_x_states):
            state_now = tf.transpose(states.read(k_now)) #transposing because state_k is (1,4) but need state_k need to be (4,1) because A is (4,4) and we need A*state_k
            state_next = tf.matmul(A_dynamics,state_now)
            if self.add_Q_status is True:
                w = tf.random_normal((1, 4), dtype=settings.float_type) * (Q_value_diag ** 0.5)  # + mean which is 0. #gives w with shape (1,4) which is same as state_next's current shape.
                state_next = state_next + w
            state_next = tf.transpose(state_next)  # transposing back (4,1) to (1,4). Because we want to stack time steps k in the rows and states in the columns.


            m_x_states_now = tf.transpose(m_x_states.read(k_now)) #transposing because state_k is (1,4) but need state_k need to be (4,1) because A is (4,4) and we need A*state_k
            m_x_states_next = tf.matmul(A_dynamics, m_x_states_now)
            m_x_states_next = tf.transpose(m_x_states_next)  # transposing back (4,1) to (1,4). Because we want to stack time steps k in the rows and states in the columns.

            P_x_states_now = tf.transpose(P_x_states.read(k_now)) #transposing because state_k is (1,4) but need state_k need to be (4,1) because A is (4,4) and we need A*state_k
            P_x_states_now_matrixed = tf.matrix_diag(tf.transpose(P_x_states_now)) #transposing here again because here (1,4) needed. The first transpose could actually be avoided, but I am keeping things consistent with previous lines
            P_x_states_now_matrixed = P_x_states_now_matrixed[0,:,:]
            Q_value_matrixed = tf.matrix_diag(Q_value_diag)
            P_x_states_next_matrixed = tf.matmul(tf.matmul(A_dynamics, P_x_states_now_matrixed)  , A_dynamics  , transpose_b=True) + Q_value_matrixed
            P_x_states_next = tf.linalg.diag_part(P_x_states_next_matrixed)
            P_x_states_next = tf.reshape(P_x_states_next, [4,1])
            P_x_states_next = tf.transpose(P_x_states_next)  # transposing back (4,1) to (1,4). Because we want to stack time steps k in the rows and states in the columns.

            state_next = tf.random_normal((1, 4), dtype=settings.float_type) * (P_x_states_next ** 0.5) + m_x_states_next  #gives w with shape (1,4) which is same as state_next's current shape.


            states = states.write(k_now+1, state_next)
            m_x_states = m_x_states.write(k_now+1, m_x_states_next)
            P_x_states = P_x_states.write(k_now+1, P_x_states_next)
            return k_now+1, states, m_x_states, P_x_states

        k_final, x_states, m_x_states, P_x_states = tf.while_loop(loop_cond, loop_body, loop_vars=[k, x_states, m_x_states, P_x_states],
                                                                  parallel_iterations=1)
        x_states = x_states.stack()
        x_states = tf.reshape(x_states,[-1,4]) #reshaping so that x_states has dimension (k,4)
        # </editor-fold>


        x_states_sliced = tf.slice(x_states,[1,0],[k_total-1,4])
        Gmean, Gvar = self._build_predict(x_states_sliced, full_cov=False, full_output_cov=False) #need to use X_sliced to get G, since G has k-1 elements and x has k.

        Y_sliced = tf.slice(self.Y, [1, 0], [k_total-1, 2]) #removing the Y_0, which was dummy index and should not be used.
        var_exp = self.likelihood.variational_expectations(Gmean, Gvar, Y_sliced, x_states_sliced) #here we give all X_states (all k elements). slcicing of C*x_states will be done in likelihood (removing the cx_0)

        # re-scale for minibatch size
        scale = tf.cast(self.num_data, settings.float_type) / tf.cast(tf.shape(self.X)[0], settings.float_type)


        return tf.reduce_sum(var_exp) * scale - KL_for_qV_given_pV - KL_for_qx0_given_Px0


def get_loglikel(model, Yt, minibatch_size):
    loglikel_ = 0.0
    loglikel_data_ = 0.0
    gauss_ll_ = 0.0
    gauss_ll_data_ = 0.0

    k = model.k_total

    np.random.seed(1)
    sess2 = model.enquire_session()
    x0_mean = model.qx0_mean.read_value(sess2)
    x0_cov = model.qx0_cov.read_value(sess2)

    x0_sampled = np.random.normal(x0_mean,x0_cov) #std dev is needed
    # x0_sampled = np.random.normal(x0_mean, np.sqrt(x0_cov))  # std dev is needed

    dt = model.dt
    d_theta = model.d_theta
    target_motion_type = model.target_motion_type
    if target_motion_type == 'consnant linear':
        A_dynamics = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])  # A for constant linear motion
    if target_motion_type == 'consnant circular':
        A_dynamics = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, np.cos(d_theta), -np.sin(d_theta)],[0, 0, np.sin(d_theta), np.cos(d_theta)]])  # A for anti-clockwise circular motion
        # A_dynamics = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, np.cos(d_theta), np.sin(d_theta)],[0, 0, -np.sin(d_theta), np.cos(d_theta)]])  # A for clockwise circular motion
    x_states = np.zeros((k,4))
    x_states[0,:] = x0_sampled
    m_x_states = np.zeros((k, 4))
    m_x_states[0, :] = x0_mean
    P_x_states = np.zeros((k, 4,4))
    P_x_states[0, 0,0] = x0_cov[0,0]
    P_x_states[0, 1,1] = x0_cov[0,1]
    P_x_states[0, 2,2] = x0_cov[0,2]
    P_x_states[0, 3,3] = x0_cov[0,3]
    Q_value_diag = np.diag(model.Q_value)
    for l in range(1,k):
        m_x_states[l,:] = np.matmul(A_dynamics,m_x_states[l-1,:])
        P_x_states[l,:,:] = np.matmul( np.matmul(A_dynamics,P_x_states[l-1,:,:]) , np.transpose(A_dynamics)) + model.Q_value
        x_states[l, :] = np.random.multivariate_normal(m_x_states[l,:],P_x_states[l,:,:])

    x_states = x_states[1:,:] #slicing and removing x_0, because x is used to get G and G should have same lenght as Y. Y has length k-1 and x has length k, so we need to remove x_0 so that G has the same length. It should be noted that Y is also initialzied as k values but y_0 is actually dummy and we dont need it. It was created as such for indexing ease.

    aa = (-(-len(x_states) // minibatch_size))
    for mb in range(-(-len(x_states) // minibatch_size)):
        mb_start = mb * minibatch_size
        mb_finish = (mb + 1) * minibatch_size
        x0_mean_mb = x0_mean[mb_start:mb_finish, :]
        x0_cov_mb = x0_cov[mb_start:mb_finish, :]
        x_states_mb = x_states[mb_start:mb_finish, :] #x_states and Y should not have same index. X has k+1 and y is k. Need to figure this out.
        Yt_mb = Yt[mb_start:mb_finish, :]
        logp, logp_data, logp_gauss, logp_gauss_data = model.mcmc_predict_density(Yt_mb, x_states_mb)  # (N_new,), (N_new,)
        loglikel_ += np.sum(logp)  # simply summing over the log p(Y_n, X_n | F_n^)
        loglikel_data_ += np.sum(logp_data)
        gauss_ll_ += np.sum(logp_gauss)
        gauss_ll_data_ += np.sum(logp_gauss_data)
    return loglikel_, loglikel_data_, gauss_ll_, gauss_ll_data_


#################################################################
#####  custom GP Monitor tasks to track metrics and params  #####
#################################################################


class LoglikelTensorBoardTask(gpmon.BaseTensorBoardTask):
    def __init__(self, file_writer, model, Yt, minibatch_size, summary_name):
        super().__init__(file_writer, model)
        self.Yt = Yt
        self.minibatch_size= minibatch_size
        self._full_ll = tf.placeholder(settings.float_type, shape=())
        self._full_ll_data = tf.placeholder(settings.float_type, shape=())
        self._full_gauss_ll = tf.placeholder(settings.float_type, shape=())
        self._full_gauss_ll_data = tf.placeholder(settings.float_type, shape=())
        self._summary = tf.summary.merge([tf.summary.scalar(summary_name + '_full', self._full_ll),
                                          tf.summary.scalar(summary_name + '_data', self._full_ll_data),
                                          tf.summary.scalar(summary_name + '_gauss_full', self._full_gauss_ll),
                                          tf.summary.scalar(summary_name + '_gauss_data', self._full_gauss_ll_data),
                                          ])

    def run(self, context: gpmon.MonitorContext, *args, **kwargs) -> None:
        loglikel_, loglikel_data_, gauss_ll_, gauss_ll_data_ = get_loglikel(self.model, self.Yt, self.minibatch_size)
        self._eval_summary(context, {self._full_ll: loglikel_, self._full_ll_data: loglikel_data_,
                                     self._full_gauss_ll: gauss_ll_, self._full_gauss_ll_data: gauss_ll_data_})
