import abc
import numpy as np
import tensorflow as tf

from gpflow.params import Parameter
from gpflow import likelihoods, settings, transforms
from gpflow.decors import params_as_tensors


class DynamicCovarianceBaseLikelihood(likelihoods.Likelihood):
    """
    Abstract class for all Wishart process likelihoods.
    """
    def __init__(self, D, cov_dim, nu, n_mc_samples, model_inverse, heavy_tail, dof=2.5):
        """
        IMPORTANT: No concrete class should directly inherit this Base class. Instead, construct concrete likelihoods
        using the component abstract likelihood classes.

        :param D: int - The number of variables in the data. In theory, this is unrelated to the dimension of the
            constructed covariance matrix.
        :param cov_dim: int - Will not be exposed to the user
        :param nu: int - Will not be exposed to user
        :param n_mc_samples: int - Number of Monte Carlo samples used to approximate the reparameterized gradients.
        :param model_inverse: boolean - If True, use the inverse Wishart process, otherwise, the Wishart process.
        :param heavy_tail: boolean - If True, use the multivariate-t emission distribution, otherwise, use the Gaussian.
        :param dof: float; optional initial value for degrees of freedom parameter for a multivariate-t emission
            distribution. This is ignored if heavy_tail==False.
        """
        super().__init__()
        self.n_mc_samples = n_mc_samples

        self.D = D
        self.cov_dim = cov_dim
        self.nu = nu
        self.model_inverse = model_inverse
        self.heavy_tail = heavy_tail

        if heavy_tail:
            # create degrees of freedom; must be > 2!
            # self.dof = 2.5
            self.dof = Parameter(dof, transform=transforms.positive, dtype=settings.float_type)

    @abc.abstractmethod
    def make_gaussian_components(self, G, Y, x):
        """
        The workhorse function that needs to be computed by concrete classes. This method returns the components used
        in the Gaussian density kernels.

        :param F: (S, N, cov_dim, __) - The (samples of the) matrix of GP outputs.
        :param Y: (N, D) -
        :return: (log_det_term, yt_inv_y)
            log_det_cov: (S, N, D, __)
            yt_inv_y: (S, N)
        """
        raise NotImplementedError("Method not implemented.")

    @params_as_tensors
    def logp_(self, G, Y, x):
        """
        Compute the (Monte Carlo estimate of) the log likelihood given samples of the GPs.

        :param F: (S, N, num_latent) -
        :param Y: (S, N) -
        :return:
        """
        log_det_cov, yt_inv_y = self.make_gaussian_components(G, Y, x)  # (S, N), (S, N)

        D_ = tf.cast(self.D, settings.float_type)

        if not self.heavy_tail:
            logp = - 0.5 * D_ * np.log(2 * np.pi) - 0.5 * yt_inv_y  # (S, N)
        else:
            dof = tf.cast(self.dof, settings.float_type)
            logp = tf.lgamma(0.5 * (dof + D_)) - tf.lgamma(0.5 * dof) - 0.5 * D_ * tf.log(np.pi * dof) \
                   - 0.5 * (dof + D_) * tf.log(1.0 + yt_inv_y / dof)  # (S, N)

        logp = logp - 0.5 * log_det_cov
        return tf.reduce_mean(logp, axis=0)  # (N,)

    @params_as_tensors
    def variational_expectations(self, G_mean, G_cov, Y, x):
        """
        logp. Models inheriting SVGP are required to have this signature.
        Compute log p(Y | variational parameters).

        :param G_mean: (N, cov_dim * nu), the parameters of the latent GP points F
        :param G_cov: (N, cov_dim * nu), the parameters of the latent GP points F
        :param Y: (N, D)
        :return: logp
            logp: (N,) - Log probability density of the data.
        """
        N = tf.shape(Y)[0]
        n_latent = tf.shape(G_mean)[1]

        # produce a sample of F, the latent GP points at the input locations X
        G_sample = tf.random_normal((self.n_mc_samples, N, n_latent), dtype=settings.float_type) * (G_cov ** 0.5) \
                   + G_mean  # (S, N, D * nu)
        G_sample = tf.reshape(G_sample, (self.n_mc_samples, N, self.cov_dim, -1))  # (S, N, cov_dim, nu)

        # finally, the likelihood variant will use these to compute the appropriate log density
        logp = self.logp_(G_sample, Y, x)

        return logp


class FullCovLikelihood(DynamicCovarianceBaseLikelihood):
    """
    Concrete class for full covariance models.
    """
    def __init__(self, D, n_mc_samples, heavy_tail, model_inverse, approx_wishart,
                 nu=None, dof=2.5):
        """

        :param D: The dimensionality of the covariance matrix being constructed with the multi-output GPs.
        :param n_mc_samples: The number of Monte Carlo samples to use to approximate the reparameterized gradients.
        :param heavy_tail: bool - If True, use the multivariate-t distribution emission model.
        :param model_inverse: bool - If True, we are modeling the inverse of the Covariance matrix with a Wishart
            distribution, i.e., this corresponds to an inverse Wishart process model.
        :param approx_wishart: bool - If True, use the additive noise model.
        :param nu:
        :param dof:
        """
        nu = D if nu is None else nu
        if nu < D:
            raise Exception("Wishart DOF must be >= D.")

        super().__init__(D, cov_dim=D, nu=nu, n_mc_samples=n_mc_samples, model_inverse=model_inverse,
                         heavy_tail=heavy_tail, dof=dof)

        self.model_inverse = model_inverse
        self.approx_wishart = approx_wishart

        # this case assumes a square scale matrix, and it must lead with dimension D
        self.scale_diag = Parameter(np.ones(self.D), transform=transforms.positive, dtype=settings.float_type)

        if approx_wishart:
            # create additional noise param; should be positive; conc=0.1 and rate=0.0001 initializes sigma2inv=1000 and
            # thus initializes sigma2=0.001
            self.p_sigma2inv_conc = Parameter(0.1, transform=transforms.positive, dtype=settings.float_type)
            self.p_sigma2inv_rate = Parameter(0.0001, transform=transforms.positive, dtype=settings.float_type)
            self.q_sigma2inv_conc = Parameter(0.1 * np.ones(self.D), transform=transforms.positive, dtype=settings.float_type)
            self.q_sigma2inv_rate = Parameter(0.0001 * np.ones(self.D), transform=transforms.positive, dtype=settings.float_type)

    @params_as_tensors
    def make_gaussian_components(self, G, Yd, x):
        """
        An auxiliary function for logp that returns the components of a Gaussian kernel.

        :param G: (S, N, D, __) - the (samples of the) matrix of GP outputs. It must have leading dimensions like
            (S, N, D, ...), where S is the number of Monte Carlo samples.
        :param Y: (N, D)
        :return:
        """

        C = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        C = tf.convert_to_tensor(C,dtype=settings.float_type)

        Cx = tf.einsum('ij,nj->ni',C,x)

        Y = Yd - Cx  # (Y-Cx) for the likelihood. likelihood usees (Y-Cx)^T (affa) (Y-Cx). can now directly use y.

        Ar_G = self.scale_diag[:, None] * G  # (S, N, D, nu)
        Ar_GG_Ar = tf.matmul(Ar_G, Ar_G, transpose_b=True)  # (S, N, D, D)

        if self.approx_wishart:
            n_samples = tf.shape(G)[0]  # could be 1 if making predictions
            dist = tf.distributions.Gamma(self.q_sigma2inv_conc, self.q_sigma2inv_rate)
            sigma2_inv = dist.sample([n_samples])  # (S, D)
            sigma2_inv = tf.clip_by_value(sigma2_inv, 1e-8, np.inf)
            sigma2 = sigma2_inv ** -1.0

            if self.model_inverse:
                additive_part = sigma2_inv[:, None, :]
            else:
                additive_part = sigma2[:, None, :]
        else:
            additive_part = 1e-5  # at the very least add some small fixed noise

        Ar_GG_Ar = tf.matrix_set_diag(Ar_GG_Ar, tf.matrix_diag_part(Ar_GG_Ar) + additive_part)

        L = tf.cholesky(Ar_GG_Ar)  # (S, N, D, D)
        log_det_cov = 2 * tf.reduce_sum(tf.log(tf.matrix_diag_part(L)), axis=2)  # (S, N)
        if self.model_inverse:
            log_det_cov = - log_det_cov

        Y.set_shape([None, self.D])  # in GPflow 1.0 I didn't need to do this

        if self.model_inverse:
            # avoid the Cholesky decomposition altogether, more computation, but probably more accurate
            y_prec = tf.einsum('jk,ijkl->ijl', Y, Ar_GG_Ar)  # (S, N, D)
            yt_inv_y = tf.reduce_sum(y_prec * Y, axis=2)  # (S, N)

        else:
            # this case can happen if approx_wishart is True
            n_samples = tf.shape(G)[0]  # could be 1 when computing MAP test metric
            Ys = tf.tile(Y[None, :, :, None], [n_samples, 1, 1, 1])  # this is inefficient, but can't get the shapes to play well with cholesky_solve otherwise
            L_solve_y = tf.matrix_triangular_solve(L, Ys, lower=True)  # (S, N, D, 1)
            yt_inv_y = tf.reduce_sum(L_solve_y ** 2.0, axis=(2, 3))  # (S, N)

        return log_det_cov, yt_inv_y

