import numpy as np
import tensorflow as tf
from scipy.stats import multivariate_normal as normal


class Equation(object):
    """Base class for defining PDE related function."""

    def __init__(self, eqn_config):
        self.dim = eqn_config.dim
        self.gamma = eqn_config.discount
        # self.total_time = eqn_config.total_time
        # self.num_time_interval = eqn_config.num_time_interval
        # self.delta_t = self.total_time / self.num_time_interval
        # self.sqrt_delta_t = np.sqrt(self.delta_t)
        
    def sample_tf(self, num_sample, T, N):
        """Sample forward SDE."""
        raise NotImplementedError
        
    def propagate_tf(self, num_sample, x0, dw_sample, NN_control, training, T, N):
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
        self.k = np.sqrt((self.gamma**2) * (self.q**2) + 4 * self.p * self.q * (self.beta**2)) / (self.beta**2) / 2
        
    def sample_tf(self, num_sample, T, N): #normal sample for BM
        delta_t = T / N
        sqrt_delta_t = np.sqrt(delta_t)
        #x0 = np.zeros(shape=[num_sample, self.dim]) + [0.99, 0.0]
        x0 = np.zeros(shape=[0, self.dim])
        while np.shape(x0)[0] < num_sample:
            x_Sample = np.random.uniform(low=-self.R, high=self.R, size=[num_sample*self.dim,self.dim])
            index = np.where(self.b_np(x_Sample) < 0)
            x0 = np.concatenate([x0, x_Sample[index[0],:]], axis=0)
            if np.shape(x0)[0] > num_sample:
                x0 = x0[0:num_sample,:]
        dw_sample = normal.rvs(size=[num_sample,
                                     self.dim,
                                     N]) * sqrt_delta_t
        x_bdry = normal.rvs(size=[num_sample, self.dim])
        norm = np.sqrt(np.sum(np.square(x_bdry), 1, keepdims=True))
        x_bdry = self.R * x_bdry / norm
        return x0, dw_sample, x_bdry
    
    def sample2_tf(self, num_sample, T, N): #bdd sample for BM
        delta_t = T / N
        sqrt_delta_t = np.sqrt(delta_t)
        #x0 = np.zeros(shape=[num_sample, self.dim]) + [0.99, 0.0]
        x0 = np.zeros(shape=[0, self.dim])
        while np.shape(x0)[0] < num_sample:
            x_Sample = np.random.uniform(low=-self.R, high=self.R, size=[num_sample,self.dim])
            index = np.where(self.b_np(x_Sample) < 0)
            x0 = np.concatenate([x0, x_Sample[index[0],:]], axis=0)
            if np.shape(x0)[0] > num_sample:
                x0 = x0[0:num_sample,:]
        dw_sample = np.random.randint(6,size=[num_sample, self.dim, N])
        dw_sample = np.floor((dw_sample - 1)/4) * (np.sqrt(3.0) * sqrt_delta_t)
        x_bdry = normal.rvs(size=[num_sample, self.dim])
        norm = np.sqrt(np.sum(np.square(x_bdry), 1, keepdims=True))
        x_bdry = self.R * x_bdry / norm
        return x0, dw_sample, x_bdry
    
    def propagate_tf(self, num_sample, x0, dw_sample, NN_control, training, T, N, cheat):
        delta_t = T / N
        x_smp = tf.reshape(x0, [num_sample, self.dim, 1])
        x_i = x0
        flag = np.ones([num_sample])
        for i in range(N):
            if cheat:
                delta_x = self.beta * self.u_true(x_i) * delta_t + self.sigma * dw_sample[:, :, i]
            else:
                delta_x = self.beta * NN_control(x_i, training, need_grad=False) * delta_t + self.sigma * dw_sample[:, :, i]
            #delta_x = self.beta * lmbd * self.u_true(x_i) * delta_t + self.sigma * dw_sample[:, :, i]
            x_iPlus1_temp = x_i + delta_x
            Exit = self.b_tf(x_iPlus1_temp) #Exit>=0 means out
            Exit = tf.reshape(tf.math.ceil((tf.math.sign(Exit)+1)/2), [num_sample]) #1 for Exit>=0, 0 for Exit<0
            delta_x_sqrnorm = tf.reduce_sum(delta_x**2, 1, keepdims=False)
            inner_product = tf.reduce_sum(delta_x * x_i, 1, keepdims=False)
            discriminant = inner_product ** 2 - delta_x_sqrnorm * (tf.reduce_sum(x_i**2,1,keepdims=False)- self.R ** 2)
            coef_i = flag*(1-Exit) + flag * Exit * (tf.sqrt(tf.abs(discriminant)) - inner_product) / delta_x_sqrnorm
            if i==0:
                coef = tf.reshape(coef_i, [num_sample, 1])
            else:
                coef = tf.concat([coef, tf.reshape(coef_i, [num_sample, 1])], axis=1)
            x_i = x_i + delta_x * tf.reshape(coef_i, [num_sample,1])
            x_smp = tf.concat([x_smp, tf.reshape(x_i, [num_sample, self.dim, 1])], axis=2)
            flag = flag * (1 - Exit)
        return x_smp, coef
    
    def propagate2_tf(self, num_sample, x0, dw_sample, NN_control, training, T, N, cheat):
        delta_t = T / N
        sqrt_delta_t = np.sqrt(delta_t)
        x_smp = tf.reshape(x0, [num_sample, self.dim, 1])
        x_i = x0
        x0_norm = tf.sqrt(tf.reduce_sum(x0**2,1))
        #temp: 2 for inside (inner); 0 (and 1) for middle layer; -2 (and -1) for boundary or outside
        temp = tf.sign(self.R - x0_norm - np.sqrt(6 * self.dim * delta_t)) + tf.sign(self.R - x0_norm - (delta_t**2))
        #flag: 2 for inside; 1 for middle layer, which means step size need modification; 0 means boundary, but we will move for at least a first step.
        flag = np.ones([num_sample]) + tf.math.floor(temp/2)
        for i in range(N):
            xi_norm = tf.sqrt(tf.reduce_sum(x_i**2,1))
            dt_i = (2*flag - (flag**2)) * ((self.R - xi_norm)**2) / (3*self.dim) + (flag**2 - 2*flag + 1) * delta_t
            if cheat:
                delta_x = self.beta * self.u_true(x_i) * tf.reshape(dt_i, [num_sample,1]) + self.sigma * dw_sample[:, :, i] * tf.reshape(tf.sqrt(dt_i), [num_sample,1]) / sqrt_delta_t
            else:
                delta_x = self.beta * NN_control(x_i, training, need_grad=False) * tf.reshape(dt_i, [num_sample,1]) + self.sigma * dw_sample[:, :, i] * tf.reshape(tf.sqrt(dt_i), [num_sample,1]) / sqrt_delta_t
            # delta_x = self.beta * self.u_true(x_i) * tf.reshape(dt_i, [num_sample,1]) + self.sigma * dw_sample[:, :, i] * tf.reshape(tf.sqrt(dt_i), [num_sample,1]) / sqrt_delta_t
            x_iPlus1_temp = x_i + delta_x
            x_iPlus1_temp_norm = tf.sqrt(tf.reduce_sum(x_iPlus1_temp**2,1,keepdims=False))
            temp = tf.sign(self.R - x_iPlus1_temp_norm - np.sqrt(6 * self.dim * delta_t)) + tf.sign(self.R - x_iPlus1_temp_norm - (delta_t**2))
            new_flag = (np.ones([num_sample]) + tf.math.floor(temp/2)) * tf.sign(flag)
            delta_x_sqrnorm = tf.reduce_sum(delta_x**2, 1, keepdims=False)
            inner_product = tf.reduce_sum(delta_x * x_i, 1, keepdims=False)
            discriminant = inner_product ** 2 - delta_x_sqrnorm * (tf.reduce_sum(x_i**2,1,keepdims=False)- self.R ** 2)
            # if flag=0, then new_flag=0, coef=0, outside; if new_flag>0, then coef=1; else, coef is in (0,1) 
            coef_i = tf.sign(new_flag) + tf.sign(flag) * (1 - tf.sign(new_flag) ) * (tf.sqrt(tf.abs(discriminant)) - inner_product) / delta_x_sqrnorm
            if i==0:
                coef = tf.reshape(coef_i, [num_sample, 1])
                dt = tf.reshape(dt_i, [num_sample, 1])
            else:
                coef = tf.concat([coef, tf.reshape(coef_i, [num_sample, 1])], axis=1)
                dt = tf.concat([dt, tf.reshape(dt_i, [num_sample, 1])], axis=1)
            x_i = x_i + delta_x * tf.reshape(coef_i, [num_sample,1])
            x_smp = tf.concat([x_smp, tf.reshape(x_i, [num_sample, self.dim, 1])], axis=2)
            flag = new_flag
        return x_smp, dt, coef

    def w_tf(self, x, u): #num_sample * 1
        return tf.reduce_sum(self.p * tf.square(x) + self.q * tf.square(u), 1, keepdims=True) - 2*self.k*self.dim

    def Z_tf(self, x): #num_sample * 1
        return 0 * tf.reduce_sum(x, 1, keepdims=True) + self.k * (self.R ** 2)

    def b_np(self, x): #num_sample * 1
        return np.sum(x**2, 1, keepdims=True) - (self.R ** 2)
    
    def b_tf(self, x): #num_sample * 1
        return tf.reduce_sum(x**2, 1, keepdims=True) - (self.R ** 2)
    
    def V_true(self, x): #num_sample * 1
        return tf.reduce_sum(tf.square(x), 1, keepdims=True) * self.k

    def u_true(self, x): #num_sample * dim
        return -self.beta * self.k / self.q * x
    

class LQtest(Equation):
    """linear quadratic regulator"""
    def __init__(self, eqn_config):
        super(LQtest, self).__init__(eqn_config)
        self.sigma = np.sqrt(2.0)
        self.A = np.array(eqn_config.A)
        self.B = eqn_config.B
        self.lmbd = eqn_config.lmbd
        self.R = eqn_config.R
        self.sqrt_lmbd = np.sqrt(self.lmbd)
        
    def sample_tf(self, num_sample, T, N): #normal sample for BM
        delta_t = T / N
        sqrt_delta_t = np.sqrt(delta_t)
        #x0 = np.zeros(shape=[num_sample, self.dim]) + [0.99, 0.0]
        x0 = np.zeros(shape=[0, self.dim])
        while np.shape(x0)[0] < num_sample:
            x_Sample = np.random.uniform(low=-self.R, high=self.R, size=[num_sample*self.dim,self.dim])
            index = np.where(self.b_np(x_Sample) < 0)
            x0 = np.concatenate([x0, x_Sample[index[0],:]], axis=0)
            if np.shape(x0)[0] > num_sample:
                x0 = x0[0:num_sample,:]
        dw_sample = normal.rvs(size=[num_sample,
                                     self.dim,
                                     N]) * sqrt_delta_t
        x_bdry = normal.rvs(size=[num_sample, self.dim])
        norm = np.sqrt(np.sum(np.square(x_bdry), 1, keepdims=True))
        x_bdry = self.R * x_bdry / norm
        return x0, dw_sample, x_bdry
    
    def sample2_tf(self, num_sample, T, N): #bdd sample for BM
        delta_t = T / N
        sqrt_delta_t = np.sqrt(delta_t)
        #x0 = np.zeros(shape=[num_sample, self.dim]) + [0.99, 0.0]
        x0 = np.zeros(shape=[0, self.dim])
        while np.shape(x0)[0] < num_sample:
            x_Sample = np.random.uniform(low=-self.R, high=self.R, size=[num_sample*self.dim,self.dim])
            index = np.where(self.b_np(x_Sample) < 0)
            x0 = np.concatenate([x0, x_Sample[index[0],:]], axis=0)
            if np.shape(x0)[0] > num_sample:
                x0 = x0[0:num_sample,:]
        dw_sample = np.random.randint(6,size=[num_sample, self.dim, N])
        dw_sample = np.floor((dw_sample - 1)/4) * (np.sqrt(3.0) * sqrt_delta_t)
        x_bdry = normal.rvs(size=[num_sample, self.dim])
        norm = np.sqrt(np.sum(np.square(x_bdry), 1, keepdims=True))
        x_bdry = self.R * x_bdry / norm
        return x0, dw_sample, x_bdry
    
    def sample3_tf(self, num_sample, T, N): #bdd sample for BM, sample more x0 near boundary
        # use r^dim as the radius distribution
        delta_t = T / N
        sqrt_delta_t = np.sqrt(delta_t)
        a = 4
        r_Sample = np.random.uniform(low=0, high=self.R, size=[num_sample,1])
        r = r_Sample**(1 / (self.dim + a)) * (self.R**((self.dim + a - 1) / (self.dim + a)))
        angle = normal.rvs(size=[num_sample, self.dim])
        norm = np.sqrt(np.sum(angle**2, 1, keepdims=True))
        x0 = r * angle / norm
        dw_sample = np.random.randint(6,size=[num_sample, self.dim, N])
        dw_sample = np.floor((dw_sample - 1)/4) * (np.sqrt(3.0) * sqrt_delta_t)
        x_bdry = normal.rvs(size=[num_sample, self.dim])
        norm = np.sqrt(np.sum(np.square(x_bdry), 1, keepdims=True))
        x_bdry = self.R * x_bdry / norm
        return x0, dw_sample, x_bdry
    
    def propagate_tf(self, num_sample, x0, dw_sample, NN_control, training, T, N):
        delta_t = T / N
        x_smp = tf.reshape(x0, [num_sample, self.dim, 1])
        x_i = x0
        flag = np.ones([num_sample])
        for i in range(N):
            # delta_x = 2 * self.sqrt_lmbd * self.u_true(x_i) * delta_t + self.sigma * dw_sample[:, :, i]
            delta_x = 2 * self.sqrt_lmbd * NN_control(x_i, training, need_grad=False) * delta_t + self.sigma * dw_sample[:, :, i]
            #delta_x = 2 * self.sqrt_lmbd * lmbd * self.u_true(x_i) * delta_t + self.sigma * dw_sample[:, :, i]
            x_iPlus1_temp = x_i + delta_x
            Exit = self.b_tf(x_iPlus1_temp) #Exit>=0 means out
            Exit = tf.reshape(tf.math.ceil((tf.math.sign(Exit)+1)/2), [num_sample]) #1 for Exit>=0, 0 for Exit<0
            delta_x_sqrnorm = tf.reduce_sum(delta_x**2, 1, keepdims=False)
            inner_product = tf.reduce_sum(delta_x * x_i, 1, keepdims=False)
            discriminant = inner_product ** 2 - delta_x_sqrnorm * (tf.reduce_sum(x_i**2,1,keepdims=False)- self.R ** 2)
            coef_i = flag*(1-Exit) + flag * Exit * (tf.sqrt(tf.abs(discriminant)) - inner_product) / delta_x_sqrnorm
            if i==0:
                coef = tf.reshape(coef_i, [num_sample, 1])
            else:
                coef = tf.concat([coef, tf.reshape(coef_i, [num_sample, 1])], axis=1)
            x_i = x_i + delta_x * tf.reshape(coef_i, [num_sample,1])
            x_smp = tf.concat([x_smp, tf.reshape(x_i, [num_sample, self.dim, 1])], axis=2)
            flag = flag * (1 - Exit)
        return x_smp, coef
    
    def propagate2_tf(self, num_sample, x0, dw_sample, NN_control, training, T, N):
        delta_t = T / N
        sqrt_delta_t = np.sqrt(delta_t)
        x_smp = tf.reshape(x0, [num_sample, self.dim, 1])
        x_i = x0
        x0_norm = tf.sqrt(tf.reduce_sum(x0**2,1))
        #temp: 2 for inside (inner); 0 (and 1) for middle layer; -2 (and -1) for boundary or outside
        temp = tf.sign(self.R - x0_norm - np.sqrt(6 * self.dim * delta_t)) + tf.sign(self.R - x0_norm - (delta_t**2))
        #flag: 2 for inside; 1 for middle layer, which means step size need modification; 0 means boundary, but we will move for at least a first step.
        flag = np.ones([num_sample]) + tf.math.floor(temp/2)
        for i in range(N):
            xi_norm = tf.sqrt(tf.reduce_sum(x_i**2,1))
            dt_i = (2*flag - (flag**2)) * ((self.R - xi_norm)**2) / (3*self.dim) + (flag**2 - 2*flag + 1) * delta_t
            delta_x = 2 * self.sqrt_lmbd * NN_control(x_i, training, need_grad=False) * tf.reshape(dt_i, [num_sample,1]) + self.sigma * dw_sample[:, :, i] * tf.reshape(tf.sqrt(dt_i), [num_sample,1]) / sqrt_delta_t
            # delta_x = 2 * self.sqrt_lmbd * self.u_true(x_i) * tf.reshape(dt_i, [num_sample,1]) + self.sigma * dw_sample[:, :, i] * tf.reshape(tf.sqrt(dt_i), [num_sample,1]) / sqrt_delta_t
            x_iPlus1_temp = x_i + delta_x
            x_iPlus1_temp_norm = tf.sqrt(tf.reduce_sum(x_iPlus1_temp**2,1,keepdims=False))
            temp = tf.sign(self.R - x_iPlus1_temp_norm - np.sqrt(6 * self.dim * delta_t)) + tf.sign(self.R - x_iPlus1_temp_norm - (delta_t**2))
            new_flag = (np.ones([num_sample]) + tf.math.floor(temp/2)) * tf.sign(flag)
            delta_x_sqrnorm = tf.reduce_sum(delta_x**2, 1, keepdims=False)
            inner_product = tf.reduce_sum(delta_x * x_i, 1, keepdims=False)
            discriminant = inner_product ** 2 - delta_x_sqrnorm * (tf.reduce_sum(x_i**2,1,keepdims=False)- self.R ** 2)
            # if flag=0, then new_flag=0, coef=0, outside; if new_flag>0, then coef=1; else, coef is in (0,1) 
            coef_i = tf.sign(new_flag) + tf.sign(flag) * (1 - tf.sign(new_flag) ) * (tf.sqrt(tf.abs(discriminant)) - inner_product) / delta_x_sqrnorm
            if i==0:
                coef = tf.reshape(coef_i, [num_sample, 1])
                dt = tf.reshape(dt_i, [num_sample, 1])
            else:
                coef = tf.concat([coef, tf.reshape(coef_i, [num_sample, 1])], axis=1)
                dt = tf.concat([dt, tf.reshape(dt_i, [num_sample, 1])], axis=1)
            x_i = x_i + delta_x * tf.reshape(coef_i, [num_sample,1])
            x_smp = tf.concat([x_smp, tf.reshape(x_i, [num_sample, self.dim, 1])], axis=2)
            flag = new_flag
        return x_smp, dt, coef

    def w_tf(self, x, u): #num_sample * 1
        return tf.reduce_sum(u**2, 1, keepdims=True)

    def Z_tf(self, x): #num_sample * 1, actually we can just use V_true
        return -tf.math.log(self.A * (x**2) + self.B) / self.lmbd

    def b_np(self, x): #num_sample * 1 contour the boundary
        return np.sum(x**2, 1, keepdims=True) - (self.R ** 2)
    
    def b_tf(self, x): #num_sample * 1 contour the boundary
        return tf.reduce_sum(x**2, 1, keepdims=True) - (self.R ** 2)
    
    def V_true(self, x): #num_sample * 1 true value
        return -tf.math.log(self.A * (x**2) + self.B) / self.lmbd

    def u_true(self, x): #num_sample * dim true control
        temp = self.A * (x**2) + self.B
        return 2 * x * self.A / temp / self.sqrt_lmbd