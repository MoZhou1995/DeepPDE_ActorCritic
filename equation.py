import numpy as np
import tensorflow as tf
from scipy.stats import multivariate_normal as normal


class Equation(object):
    """Base class for defining PDE related function."""

    def __init__(self, eqn_config):
        self.dim = eqn_config.dim
        self.total_time = eqn_config.total_time
        self.num_time_interval = eqn_config.num_time_interval
        self.delta_t = self.total_time / self.num_time_interval
        self.sqrt_delta_t = np.sqrt(self.delta_t)

    def sample(self, num_sample):
        """Sample forward SDE."""
        raise NotImplementedError

    def w_tf(self, x, u):
        """Running cost in control problems."""
        raise NotImplementedError

    def Z_tf(self, x):
        """Terminal cost in control problems."""
        raise NotImplementedError
        
    def b_tf(self, x):
        """A function to specify the boundary."""
        raise NotImplementedError
        
    def V_true(self, x):
        """True value function"""
        raise NotImplementedError
    
    def u_true(self, x):
        """Optimal control"""
        raise NotImplementedError

class LQR(Equation):
    """linear quadratic regulator"""
    def __init__(self, eqn_config):
        super(LQR, self).__init__(eqn_config)
        self.sigma = np.sqrt(2.0)
        self.p = eqn_config.p
        self.q = eqn_config.q
        self.beta = eqn_config.beta
        self.R = eqn_config.R
        self.sqrtpqoverbeta = np.sqrt(self.p * self.q) / self.beta
        print("define LQR")

    def sample(self, num_sample, control_fcn):
        print("call sample")
        # uniformly sample x0 in a ball, rejection sampling, to be improved later
        x0 = np.zeros(shape=[0, self.dim])
        while np.shape(x0)[0] < num_sample:
            x_sample = np.random.uniform(low=-self.R, high=self.R, size=[num_sample,self.dim])
            index = np.where(self.b_np(x_sample) < 0)
            x0 = np.concatenate([x0, x_sample[index[0],:]], axis=0)
            if np.shape(x0)[0] > num_sample:
                x0 = x0[0:num_sample,:]
        print("sample initialization finished")
        x_tau = np.zeros([num_sample, self.dim])        
        tau = self.total_time * np.ones(num_sample)
        # ExitIndex=i means X_i is in Omega but X_{i+1} is not
        ExitIndex = np.ones(num_sample) * self.num_time_interval
        dw_sample = normal.rvs(size=[num_sample,
                                     self.dim,
                                     self.num_time_interval]) * self.sqrt_delta_t
        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1])
        x_sample[:, :, 0] = x0
        print("sample start propagation")
        for i in range(self.num_time_interval):
            print("sample time step", i)
            x_sample[:, :, i + 1] = x_sample[:, :, i] + self.beta * control_fcn(x_sample[:, :, i]) * self.delta_t + self.sigma * dw_sample[:, :, i]
            Exit = self.b_np(x_sample[:, :, i + 1])
            # using loop currently, to be improved later
            for j in range(num_sample):
                print("sample index", j)
                if Exit[j] >= 0 and ExitIndex[j] == self.num_time_interval:
                    print("sample find exit")
                    ExitIndex[j] = i
                    delta_x = np.sqrt(np.sum((x_sample[j, :, i + 1]-x_sample[j, :, i])**2))
                    mu = (self.R - np.sqrt(np.sum(x_sample[j, :, i]**2)) ) / delta_x
                    tau[j] = (i + mu) * self.delta_t
                    x_tau[j,:] = (1-mu) * x_sample[j, :, i] + mu * x_sample[j, :, i+1]
        print("sample assigning x_tau in Omega")
        for j in range(num_sample):
            if ExitIndex[j] == self.num_time_interval:
                x_tau[j,:] = x_sample[j, :, self.num_time_interval]
        print("sample finish")
        return dw_sample, x_sample, x_tau, tau, ExitIndex

    def w_tf(self, x, u):
        print("call w_tf")
        return tf.reduce_sum(self.p*tf.square(x) + self.q * tf.square(u) - 2*self.sqrtpqoverbeta
                             , 1, keepdims=True)

    def Z_tf(self, x):
        print("call Z_tf")
        return 0 * tf.reduce_sum(x, 1, keepdims=False) + self.sqrtpqoverbeta * (self.R ** 2)

    def b_tf(self, x):
        print("call b_tf")
        return tf.reduce_sum(tf.square(x), 1, keepdims=True) - (self.R ** 2)
    
    def b_np(self, x):
        print("call b_np")
        return np.sum(x**2, 1, keepdims=True) - (self.R ** 2)
    
    def V_true(self, x):
        print("call V_true")
        return tf.reduce_sum(tf.square(x), 1, keepdims=True) * self.sqrtpqoverbeta

    def u_true(self, x):
        print("call u_true")
        return -np.sqrt(self.p / self.q) * x