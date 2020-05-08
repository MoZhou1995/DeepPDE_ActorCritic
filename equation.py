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
        #self.x_sample = tf.Variable(initial_value=0, trainable=False, shape=None)

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
        
    def sample_tf(self, num_sample):
        #x0 = np.zeros(shape=[num_sample, self.dim])# + [0.7, 0.1]
        x0 = np.zeros(shape=[0, self.dim])
        while np.shape(x0)[0] < num_sample:
            x_Sample = np.random.uniform(low=-self.R, high=self.R, size=[num_sample,self.dim])
            index = np.where(self.b_np(x_Sample) < 0)
            x0 = np.concatenate([x0, x_Sample[index[0],:]], axis=0)
            if np.shape(x0)[0] > num_sample:
                x0 = x0[0:num_sample,:]
        dw_sample = normal.rvs(size=[num_sample,
                                     self.dim,
                                     self.num_time_interval]) * self.sqrt_delta_t
        x_bdry = normal.rvs(size=[num_sample, self.dim])
        norm = np.sqrt(np.sum(np.square(x_bdry), 1, keepdims=True))
        x_bdry = self.R * x_bdry / norm
        return x0, dw_sample, x_bdry
    
    #def propagate_tf(self, num_sample, x0, dw_sample, NN_control, training):
    def propagate_tf(self, num_sample, x0, dw_sample, lmbd, training):
        #x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1])
        #x_sample[:, :, 0] = x0
        x_smp = tf.reshape(x0, [num_sample, self.dim, 1])
        x_i = x0
        #coef = np.ones([num_sample, self.num_time_interval])
        flag = np.ones([num_sample])
        for i in range(self.num_time_interval):
            #delta_x = self.beta * NN_control(x_sample[:, :, i], training, need_grad=False)\
            #    * self.delta_t + self.sigma * dw_sample[:, :, i]
            #delta_x = self.beta * NN_control(x_i, training, need_grad=False)\
            #    * self.delta_t + self.sigma * dw_sample[:, :, i]
            delta_x = self.beta * lmbd * self.u_true(x_i) * self.delta_t + self.sigma * dw_sample[:, :, i]
            #x_iPlus1_temp = x_sample[:, :, i] + delta_x
            x_iPlus1_temp = x_i + delta_x
            Exit = self.b_tf(x_iPlus1_temp) #Exit>=0 means out
            Exit = tf.reshape(tf.math.ceil((tf.math.sign(Exit)+1)/2), [num_sample]) #1 for Exit>=0, 0 for Exit<0
            delta_x_sqrnorm = tf.reduce_sum(delta_x**2, 1, keepdims=False)
            #inner_product = tf.reduce_sum(delta_x * x_sample[:,:,i], 1, keepdims=False)
            inner_product = tf.reduce_sum(delta_x * x_i, 1, keepdims=False)
            #discriminant = inner_product ** 2 - delta_x_sqrnorm * (tf.reduce_sum(x_sample[:, :, i]**2,1,keepdims=False)- self.R ** 2)
            discriminant = inner_product ** 2 - delta_x_sqrnorm * (tf.reduce_sum(x_i**2,1,keepdims=False)- self.R ** 2)
            #coef[:,i] = flag*(1-Exit) + flag * Exit * (tf.sqrt(tf.abs(discriminant)) - inner_product) / delta_x_sqrnorm
            coef_i = flag*(1-Exit) + flag * Exit * (tf.sqrt(tf.abs(discriminant)) - inner_product) / delta_x_sqrnorm
            if i==0:
                coef = tf.reshape(coef_i, [num_sample, 1])
            else:
                coef = tf.concat([coef, tf.reshape(coef_i, [num_sample, 1])], axis=1)
            #x_sample[:, :, i + 1] = x_sample[:, :, i] + delta_x * tf.reshape(coef[:,i], [num_sample,1])
            x_i = x_i + delta_x * tf.reshape(coef_i, [num_sample,1])
            x_smp = tf.concat([x_smp, tf.reshape(x_i, [num_sample, self.dim, 1])], axis=2)
            flag = flag * (1 - Exit)
        #return x_sample, coef
        return x_smp, coef
    
    def sample(self, num_sample, control_fcn):
        # uniformly sample x0 in a ball, rejection sampling, maybe to be improved later
        x0 = np.zeros(shape=[0, self.dim])
        while np.shape(x0)[0] < num_sample:
            x_Sample = np.random.uniform(low=-self.R, high=self.R, size=[num_sample,self.dim])
            index = np.where(self.b_np(x_Sample) < 0)
            x0 = np.concatenate([x0, x_Sample[index[0],:]], axis=0)
            if np.shape(x0)[0] > num_sample:
                x0 = x0[0:num_sample,:]
        coef = np.ones([num_sample, self.num_time_interval])
        dw_sample = normal.rvs(size=[num_sample,
                                     self.dim,
                                     self.num_time_interval]) * self.sqrt_delta_t
        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1])
        x_sample[:, :, 0] = x0
        # flag is used to denote stopping time
        flag = np.ones([num_sample]) #1 for in, 0 for out
        for i in range(self.num_time_interval):
            delta_x = self.beta * control_fcn(x_sample[:, :, i])\
                * self.delta_t + self.sigma * dw_sample[:, :, i]
            x_iPlus1_temp = x_sample[:, :, i] + delta_x
            Exit = self.b_np(x_iPlus1_temp) #Exit>=0 means out
            Exit = np.reshape(np.ceil((np.sign(Exit)+1)/2), [num_sample]) #1 for Exit>=0, 0 for Exit<0
            #coef[:,i] = flag * coef[:,i]
            # the following is to compute x_tau and corresponding coefficients
            #index_out = np.where(flag + Exit == 2)[0] # flag=Exit=1, out at this step
            delta_x_sqrnorm = tf.reduce_sum(delta_x**2, 1, keepdims=False)
            inner_product = tf.reduce_sum(delta_x * x_sample[:,:,i], 1, keepdims=False)
            discriminant = inner_product ** 2 - delta_x_sqrnorm * (tf.reduce_sum(x_sample[:, :, i]**2,1,keepdims=False)- self.R ** 2)
            coef[:,i] = flag*(1-Exit) + flag * Exit * (tf.sqrt(tf.abs(discriminant)) - inner_product) / delta_x_sqrnorm
            # dx_out = delta_x[index_out,:] #1*49*dim
            # dx_out_sqrnorm = np.sum(dx_out ** 2, 1, keepdims=False)
            # x_i_out = x_sample[index_out, :, i]
            # inner_product = np.sum(dx_out * x_i_out, 1, keepdims=False)
            # discriminant = inner_product ** 2 - dx_out_sqrnorm * (np.sum(x_i_out**2,1,keepdims=False)- self.R ** 2)
            # coef[index_out,i] = (np.sqrt(discriminant) - inner_product) / dx_out_sqrnorm
            x_sample[:, :, i + 1] = x_sample[:, :, i] + delta_x * tf.reshape(coef[:,i], [num_sample,1])
            flag = flag * (1 - Exit)
        #sample on the boundary
        x_bdry = normal.rvs(size=[num_sample, self.dim])
        norm = np.sqrt(np.sum(np.square(x_bdry), 1, keepdims=True))
        x_bdry = self.R * x_bdry / norm
        return dw_sample, x_sample, coef, x_bdry

    def w_tf(self, x, u): #num_sample * 1
        return tf.reduce_sum(self.p * tf.square(x) + self.q * tf.square(u) - 2*self.sqrtpqoverbeta, 1, keepdims=True)

    def Z_tf(self, x): #num_sample * 1
        return 0 * tf.reduce_sum(x, 1, keepdims=True) + self.sqrtpqoverbeta * (self.R ** 2)

    def b_np(self, x): #num_sample * 1
        return np.sum(x**2, 1, keepdims=True) - (self.R ** 2)
    
    def b_tf(self, x): #num_sample * 1
        return tf.reduce_sum(x**2, 1, keepdims=True) - (self.R ** 2)
    
    def V_true(self, x): #num_sample * 1
        return tf.reduce_sum(tf.square(x), 1, keepdims=True) * self.sqrtpqoverbeta

    def u_true(self, x): #num_sample * dim
        return -np.sqrt(self.p / self.q) * x