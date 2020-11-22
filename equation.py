import numpy as np
import tensorflow as tf
from scipy.optimize import fsolve
from scipy.optimize import bisect
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
        self.k = ( ((self.gamma**2) * (self.q**2) + 4 * self.p * self.q * (self.beta**2))**0.5 - self.q*self.gamma )/ (self.beta**2) / 2
        
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
    
    def sample0_tf(self, num_sample, T, N): #bdd sample for BM
        delta_t = T / N
        sqrt_delta_t = np.sqrt(delta_t)
        x0 = np.zeros(shape=[num_sample, self.dim]) + 0.01
        dw_sample = np.random.randint(6,size=[num_sample, self.dim, N])
        dw_sample = np.floor((dw_sample - 1)/4) * (np.sqrt(3.0) * sqrt_delta_t)
        x_bdry = normal.rvs(size=[num_sample, self.dim])
        norm = np.sqrt(np.sum(np.square(x_bdry), 1, keepdims=True))
        x_bdry = self.R * x_bdry / norm
        return x0, dw_sample, x_bdry
    
    def propagate0_tf(self, num_sample, x0, dw_sample, NN_control, training, T, N, cheat):
        # the most naive scheme, just stop where next step is out
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
            # delta_x_sqrnorm = tf.reduce_sum(delta_x**2, 1, keepdims=False)
            # inner_product = tf.reduce_sum(delta_x * x_i, 1, keepdims=False)
            # discriminant = inner_product ** 2 - delta_x_sqrnorm * (tf.reduce_sum(x_i**2,1,keepdims=False)- self.R ** 2)
            # coef_i = flag*(1-Exit) + flag * Exit * (tf.sqrt(tf.abs(discriminant)) - inner_product) / delta_x_sqrnorm
            coef_i = flag * (1 - Exit)
            if i==0:
                coef = tf.reshape(coef_i, [num_sample, 1])
            else:
                coef = tf.concat([coef, tf.reshape(coef_i, [num_sample, 1])], axis=1)
            x_i = x_i + delta_x * tf.reshape(coef_i, [num_sample,1])
            x_smp = tf.concat([x_smp, tf.reshape(x_i, [num_sample, self.dim, 1])], axis=2)
            flag = flag * (1 - Exit)
        return x_smp, coef
    
    def propagate_tf(self, num_sample, x0, dw_sample, NN_control, training, T, N, cheat):
        # the original scheme
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
    
    def propagate1_tf(self, num_sample, x0, dw_sample, NN_control, training, T, N, cheat):
        # the most complicated scheme, need to solve a quartic equation to get the coefficient
        delta_t = T / N
        x_smp = tf.reshape(x0, [num_sample, self.dim, 1])
        x_i = x0
        flag = np.ones([num_sample])
        for i in range(N):
            if cheat:
                delta_x_drift = self.beta * self.u_true(x_i) * delta_t
            else:
                delta_x_drift = self.beta * NN_control(x_i, training, need_grad=False) * delta_t
            delta_x_diffusion = self.sigma * dw_sample[:, :, i]
            delta_x = delta_x_drift + delta_x_diffusion
            x_iPlus1_temp = x_i + delta_x
            Exit = self.b_tf(x_iPlus1_temp) #Exit>=0 means out
            Exit = tf.reshape(tf.math.ceil((tf.math.sign(Exit)+1)/2), [num_sample]) #1 for Exit>=0, 0 for Exit<0
            coef_i_temp = (1 - Exit) * flag
            # find the index for boundary
            exit_index = tf.where(flag*Exit==1) # care the dim, num_exit x 1
            x_exit =tf.gather_nd(x_i,exit_index)
            delta_x_drift_exit = tf.gather_nd(delta_x_drift,exit_index)
            delta_x_diffusion_exit = tf.gather_nd(delta_x_diffusion,exit_index)
             # the following code is to explicitly solve a quartic equation [a,b,c,d,e]*[x^4,x^3,x^2,x,1]=0 to get sqrt_rho
            # we want to find rho s.t. x_i + drift * rho + diffusion * sqrt_rho is on the boundary
            a = tf.reduce_sum(delta_x_drift_exit**2, axis=1, keepdims=False) # shape = [num_sample] for a b c d e p q Delta_0 Delta_1
            b = 2 * tf.reduce_sum(delta_x_drift_exit * delta_x_diffusion_exit, axis=1, keepdims=False)
            c = tf.reduce_sum(2 * delta_x_drift_exit * x_exit + (delta_x_diffusion_exit**2), axis=1, keepdims=False)
            d = 2 * tf.reduce_sum(delta_x_diffusion_exit * x_exit, axis=1, keepdims=False)
            e = tf.reduce_sum(x_exit**2, axis=1, keepdims=False) - self.R**2
            p = (8*a*c - 3*(b**2)) / (8*(a**2))
            q = (b**3 - 4*a*b*c + 8*(a**2)*d) / (8 * (a**3))
            sign_q = tf.sign(q)
            Delta_0 = c**2 - 3*b*d + 12*a*e
            Delta_1 = 2*(c**3) - 9*b*c*d + 27*(b**2)*e + 27*a*(d**2) - 72*a*c*e
            Delta_2 = Delta_1**2 - 4*(Delta_0**3)
            sign_Delta_2 = tf.sign(Delta_2)
            signal_Delta_2 = tf.math.ceil((sign_Delta_2+1)/2)
            # this is for sign_Delta >= 0
            QQ = (Delta_1 + ( tf.abs(Delta_2) )**0.5 ) / 2 # the absolute value here is to make sure that no nan
            Q = tf.sign(QQ) * tf.abs(QQ)**(1/3) #I don't know why python cannot compute the cubic root of negative number
            S_plus = 0.5 * tf.abs((Q + Delta_0/Q) / (3*a) - 2*p/3)**0.5
            # This is for sign_Delta < 0
            phi = tf.math.acos( tf.clip_by_value( Delta_1 / 2/ tf.abs(Delta_0)**1.5,-1,1) )
            S_minus = 0.5 * tf.abs(2 * (tf.abs(Delta_0)**0.5) * tf.math.cos(phi/3) / (3*a) - 2*p/3)**0.5
            S = signal_Delta_2 * S_plus + (1-signal_Delta_2) * S_minus
            # either root1 or root3 is what we want.
            # root1 = 0.5 * (q/S - 4*(S**2) - 2*p)**0.5 - b/4/a - S
            # root2 = - 0.5 * (q/S - 4*(S**2) - 2*p)**0.5 - b/4/a - S
            # root3 = 0.5 * (-q/S - 4*(S**2) - 2*p)**0.5 - b/4/a + S
            # root4 = - 0.5 * (-q/S - 4*(S**2) - 2*p)**0.5 - b/4/a + S
            # root = tf.concat([[root1],[root2],[root3],[root4]],axis=0)
            # root = tf.transpose(root)
            temp = -4*(S**2) -2*p + tf.abs(q/S)
            sqrt_rho = 0.5 * tf.abs(temp)**0.5 - b/4/a - sign_q*S
            check_index = tf.where( (1-sqrt_rho) * sqrt_rho < 0) # same format as Exit_index
            new_temp = -4*(S**2) -2*p - tf.abs(q/S)
            new_sqrt_rho = 0.5 * tf.abs(new_temp)**0.5 - b/4/a + sign_q*S # new_temp should be positive at check_index
            new_sqrt_rho = tf.gather(new_sqrt_rho, check_index[:,0])
            sqrt_rho_final = tf.tensor_scatter_nd_update(sqrt_rho, check_index, new_sqrt_rho)
            # now we get the solutiob of the quartic equation and we can use the next line to check
            # result = a*(sqrt_rho**4) + b*(sqrt_rho**3) + c*(sqrt_rho**2) + d*sqrt_rho+e
            rho = sqrt_rho_final**2
            # rho = rho*0 + 0.000001
            # rho = rho * 0.000001
            coef_i = tf.tensor_scatter_nd_update(coef_i_temp, exit_index, rho)
            if i==0:
                coef = tf.reshape(coef_i, [num_sample, 1])
            else:
                coef = tf.concat([coef, tf.reshape(coef_i, [num_sample, 1])], axis=1)
            x_i = x_i + delta_x_drift * tf.reshape(coef_i, [num_sample,1]) + delta_x_diffusion * tf.reshape(coef_i**0.5, [num_sample,1])
            x_smp = tf.concat([x_smp, tf.reshape(x_i, [num_sample, self.dim, 1])], axis=2)
            flag = flag * (1 - Exit)
        # tf.print("x_smp", tf.reduce_prod(x_smp))
        # tf.print("coef", tf.reduce_sum(coef))
        return x_smp, coef
    
    def propagate2_tf(self, num_sample, x0, dw_sample, NN_control, training, T, N, cheat):
        # the new scheme
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
            # # if flag=0, then new_flag=0, coef=0, outside; if new_flag>0, then coef=1; else, coef is in (0,1) 
            coef_i = tf.sign(new_flag) + tf.sign(flag) * (1 - tf.sign(new_flag) ) * (tf.sqrt(tf.abs(discriminant)) - inner_product) / delta_x_sqrnorm
            # coef_i = tf.sign(flag) * tf.sign(new_flag)
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

    def b_tf(self, x): #num_sample * 1
        return tf.reduce_sum(x**2, 1, keepdims=True) - (self.R ** 2)

    def b_np(self, x): #num_sample * 1
        return np.sum(x**2, 1, keepdims=True) - (self.R ** 2)
    
    def V_true(self, x): #num_sample * 1
        return tf.reduce_sum(tf.square(x), 1, keepdims=True) * self.k

    def u_true(self, x): #num_sample * dim
        return -self.beta * self.k / self.q * x
    
    def V_grad_true(self, x): #num_sample * dim
        return 2 * self.k * x

    
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
    
    def sample0_tf(self, num_sample, T, N): #bdd sample for BM
        delta_t = T / N
        sqrt_delta_t = np.sqrt(delta_t)
        x0 = np.zeros(shape=[num_sample, self.dim])# + [0.99, 0.0]
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
                delta_x = 2 * self.sqrt_lmbd * self.u_true(x_i) * tf.reshape(dt_i, [num_sample,1]) + self.sigma * dw_sample[:, :, i] * tf.reshape(tf.sqrt(dt_i), [num_sample,1]) / sqrt_delta_t
            else:
                delta_x = 2 * self.sqrt_lmbd * NN_control(x_i, training, need_grad=False) * tf.reshape(dt_i, [num_sample,1]) + self.sigma * dw_sample[:, :, i] * tf.reshape(tf.sqrt(dt_i), [num_sample,1]) / sqrt_delta_t
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
        return -tf.math.log(tf.reduce_sum(self.A * (x**2), 1, keepdims=True) + self.B) / self.lmbd

    def b_np(self, x): #num_sample * 1 contour the boundary
        return np.sum(x**2, 1, keepdims=True) - (self.R ** 2)
    
    def V_true(self, x): #num_sample * 1 true value
        return -tf.math.log(tf.reduce_sum(self.A * (x**2), 1, keepdims=True) + self.B) / self.lmbd

    def u_true(self, x): #num_sample * dim true control
        temp = tf.reduce_sum(self.A * (x**2), 1, keepdims=True) + self.B
        return 2 * x * self.A / temp / self.sqrt_lmbd
    
class VDP1(Equation):
    """Van Der Pol oscillator"""
    def __init__(self, eqn_config):
        super(VDP1, self).__init__(eqn_config)
        self.sigma = np.sqrt(2.0)
        self.q = eqn_config.q
        self.R = eqn_config.R
        
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
    
    def sample0_tf(self, num_sample, T, N): #bdd sample for BM
        delta_t = T / N
        sqrt_delta_t = np.sqrt(delta_t)
        x0 = np.zeros(shape=[num_sample, self.dim])# + [0.99, 0.0]
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
                u = self.u_true(x_i)
            else:
                u = NN_control(x_i, training, need_grad=False)
            delta_x = tf.concat([x_i[:,1:2], (1 - x_i[:,0:1]**2)*x_i[:,1:2] - x_i[:,0:1] + u],axis=1) * delta_t + self.sigma * dw_sample[:, :, i]
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
                u = self.u_true(x_i)
            else:
                u = NN_control(x_i, training, need_grad=False)
            delta_x = tf.concat([x_i[:,1:2], (1 - x_i[:,0:1]**2)*x_i[:,1:2] - x_i[:,0:1] + u],axis=1) * tf.reshape(dt_i, [num_sample,1]) + self.sigma * dw_sample[:, :, i] * tf.reshape(tf.sqrt(dt_i), [num_sample,1]) / sqrt_delta_t
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
        return ((x[:,0:1])**2) * (self.gamma + x[:,1:2]) - 2*x[:,0:1]*x[:,1:2] + x[:,0:1] + (self.gamma-1)*x[:,1:2] + 1/4/(self.q**2) -2 + self.q*(u**2)

    def Z_tf(self, x): #num_sample * 1
        return (x[:,0:1])**2 + x[:,1:2]

    def b_np(self, x): #num_sample * 1
        return np.sum(x**2, 1, keepdims=True) - (self.R ** 2)
    
    def V_true(self, x): #num_sample * 1
        return (x[:,0:1])**2 + x[:,1:2]

    def u_true(self, x): #num_sample * 1
        return tf.reduce_sum(x*0,1,keepdims=True) - 1/2/self.q
    
    
class VDP2(Equation):
    """Van Der Pol oscillator"""
    def __init__(self, eqn_config):
        super(VDP2, self).__init__(eqn_config)
        self.sigma = np.sqrt(2.0)
        self.q = eqn_config.q
        self.R = eqn_config.R
        self.d = eqn_config.control_dim
        
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
    
    def sample0_tf(self, num_sample, T, N): #bdd sample for BM
        delta_t = T / N
        sqrt_delta_t = np.sqrt(delta_t)
        x0 = np.zeros(shape=[num_sample, self.dim]) + 0.1
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
                u = self.u_true(x_i)
            else:
                u = NN_control(x_i, training, need_grad=False)
            x_i1 = x_i[:,0:self.d] #num_sample * d
            x_i2 = x_i[:,self.d:self.dim]
            delta_x = tf.concat([x_i2, (1 - x_i1**2)*x_i2 - x_i1 + u],axis=1) * delta_t + self.sigma * dw_sample[:, :, i]
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
                u = self.u_true(x_i)
            else:
                u = NN_control(x_i, training, need_grad=False)
            x_i1 = x_i[:,0:self.d] #num_sample * d
            x_i2 = x_i[:,self.d:self.dim]
            delta_x = tf.concat([x_i2, (1 - x_i1**2)*x_i2 - x_i1 + u],axis=1) * tf.reshape(dt_i, [num_sample,1]) + self.sigma * dw_sample[:, :, i] * tf.reshape(tf.sqrt(dt_i), [num_sample,1]) / sqrt_delta_t
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
        d = self.d # dim/2
        x1 = x[:,0:d] #num_sample * d
        x2 = x[:,d:self.dim] #num_sample * d
        px1 = tf.concat([x1[:,1:d],x1[:,0:1]],1) #num_sample * d
        px2 = tf.concat([x2[:,1:d],x2[:,0:1]],1) #num_sample * d
        nx1 = tf.concat([x1[:,d-1:d],x1[:,0:d-1]],1)
        nx2 = tf.concat([x2[:,d-1:d],x2[:,0:d-1]],1)
        dv1 = px1 + nx1 #num_sample * d
        dv2 = px2 + nx2 #num_sample * d
        temp = self.gamma*(x1*px1 + x2*px2) + (dv2**2)/4/self.q - x2*dv1 - ((1-x1**2)*x2 - x1)*dv2
        return tf.reduce_sum(temp + self.q*(u**2), 1, keepdims=True)

    def Z_tf(self, x): #num_sample * 1
        return self.V_true(x)

    def b_np(self, x): #num_sample * 1
        return np.sum(x**2, 1, keepdims=True) - (self.R ** 2)
    
    def V_true(self, x): #num_sample * 1
        d = self.d # dim/2
        x1 = x[:,0:d] #num_sample * d
        x2 = x[:,d:self.dim] #num_sample * d
        px1 = tf.concat([x1[:,1:d],x1[:,0:1]],1) #num_sample * d
        px2 = tf.concat([x2[:,1:d],x2[:,0:1]],1) #num_sample * d
        return tf.reduce_sum(x1*px1 + x2*px2, 1, keepdims=True)

    def u_true(self, x): #num_sample * 1
        d = self.d
        x2 = x[:,d:self.dim] #num_sample * d
        px2 = tf.concat([x2[:,1:d],x2[:,0:1]],1)
        nx2 = tf.concat([x2[:,d-1:d],x2[:,0:d-1]],1)
        return -(px2 + nx2)/2/self.q
    
    
class VDP3(Equation):
    """Van Der Pol oscillator"""
    def __init__(self, eqn_config):
        super(VDP3, self).__init__(eqn_config)
        self.sigma = np.sqrt(2.0)
        self.a = eqn_config.a
        self.epsl = eqn_config.epsilon
        self.q = eqn_config.q
        self.R = eqn_config.R
        self.d = eqn_config.control_dim
        
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
    
    def sample0_tf(self, num_sample, T, N): #bdd sample for BM
        delta_t = T / N
        sqrt_delta_t = np.sqrt(delta_t)
        x0 = np.zeros(shape=[num_sample, self.dim]) + 0.1
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
                u = self.u_true(x_i)
            else:
                u = NN_control(x_i, training, need_grad=False)
            x_i1 = x_i[:,0:self.d] #num_sample * d
            x_i2 = x_i[:,self.d:self.dim]
            delta_x = tf.concat([x_i2, (1 - x_i1**2)*x_i2 - x_i1 + u],axis=1) * delta_t + self.sigma * dw_sample[:, :, i]
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
                u = self.u_true(x_i)
            else:
                u = NN_control(x_i, training, need_grad=False)
            x_i1 = x_i[:,0:self.d] #num_sample * d
            x_i2 = x_i[:,self.d:self.dim]
            delta_x = tf.concat([x_i2, (1 - x_i1**2)*x_i2 - x_i1 + u],axis=1) * tf.reshape(dt_i, [num_sample,1]) + self.sigma * dw_sample[:, :, i] * tf.reshape(tf.sqrt(dt_i), [num_sample,1]) / sqrt_delta_t
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
        d = self.d # dim/2
        x1 = x[:,0:d] #num_sample * d
        x2 = x[:,d:self.dim] #num_sample * d
        px1 = tf.concat([x1[:,1:d],x1[:,0:1]],1) #num_sample * d
        px2 = tf.concat([x2[:,1:d],x2[:,0:1]],1) #num_sample * d
        nx1 = tf.concat([x1[:,d-1:d],x1[:,0:d-1]],1)
        nx2 = tf.concat([x2[:,d-1:d],x2[:,0:d-1]],1)
        dv1 = 2*self.a*x1 - self.epsl*(px1 + nx1) #num_sample * d
        dv2 = 2*self.a*x2 - self.epsl*(px2 + nx2) #num_sample * d
        temp = -self.gamma*self.epsl*(x1*px1 + x2*px2) + (dv2**2)/4/self.q - x2*dv1 - ((1-x1**2)*x2 - x1)*dv2
        return tf.reduce_sum(temp + self.q*(u**2), 1, keepdims=True) + self.gamma*self.a*tf.reduce_sum(x**2, 1, keepdims=True) - 2*self.a*self.dim

    def Z_tf(self, x): #num_sample * 1
        return self.V_true(x)

    def b_np(self, x): #num_sample * 1
        return np.sum(x**2, 1, keepdims=True) - (self.R ** 2)
    
    def V_true(self, x): #num_sample * 1
        d = self.d # dim/2
        x1 = x[:,0:d] #num_sample * d
        x2 = x[:,d:self.dim] #num_sample * d
        px1 = tf.concat([x1[:,1:d],x1[:,0:1]],1) #num_sample * d
        px2 = tf.concat([x2[:,1:d],x2[:,0:1]],1) #num_sample * d
        return self.a*tf.reduce_sum(x**2, 1, keepdims=True) - self.epsl*tf.reduce_sum(x1*px1 + x2*px2, 1, keepdims=True)

    def u_true(self, x): #num_sample * 1
        d = self.d
        x2 = x[:,d:self.dim] #num_sample * d
        px2 = tf.concat([x2[:,1:d],x2[:,0:1]],1)
        nx2 = tf.concat([x2[:,d-1:d],x2[:,0:d-1]],1)
        return -(2*self.a*x2 - self.epsl*(px2 + nx2))/2/self.q
    

class ekn(Equation):
    """Diffusive Eikonal equation"""
    def __init__(self, eqn_config):
        super(ekn, self).__init__(eqn_config)
        self.a2 = eqn_config.a2
        self.a3 = eqn_config.a3
        self.epsl = 1/2/self.a2/self.dim
        self.sigma = np.sqrt(self.epsl) * np.sqrt(2.0)
        self.R = eqn_config.R
        
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
    
    def sample0_tf(self, num_sample, T, N): #bdd sample for BM
        delta_t = T / N
        sqrt_delta_t = np.sqrt(delta_t)
        x0 = np.zeros(shape=[num_sample, self.dim]) + 0.1
        dw_sample = np.random.randint(6,size=[num_sample, self.dim, N])
        dw_sample = np.floor((dw_sample - 1)/4) * (np.sqrt(3.0) * sqrt_delta_t)
        x_bdry = normal.rvs(size=[num_sample, self.dim])
        norm = np.sqrt(np.sum(np.square(x_bdry), 1, keepdims=True))
        x_bdry = self.R * x_bdry / norm
        return x0, dw_sample, x_bdry
    
    def propagate0_tf(self, num_sample, x0, dw_sample, NN_control, training, T, N, cheat):
        delta_t = T / N
        x_smp = tf.reshape(x0, [num_sample, self.dim, 1])
        x_i = x0
        flag = np.ones([num_sample])
        for i in range(N):
            if cheat:
                u = self.u_true(x_i)
            else:
                u = NN_control(x_i, training, need_grad=False) #*0.9 + 0.1*self.u_true(x_i)
            xi_norm_square = tf.reduce_sum(x_i**2,1,keepdims=True)
            xi_norm = tf.sqrt(xi_norm_square)
            c = 3 * (self.dim+1) * self.a3 / 2/self.a2 / self.dim / (2*self.a2 - 3*self.a3*xi_norm)
            delta_x = c*u * delta_t + self.sigma * dw_sample[:, :, i]
            x_iPlus1_temp = x_i + delta_x
            Exit = self.b_tf(x_iPlus1_temp) #Exit>=0 means out
            Exit = tf.reshape(tf.math.ceil((tf.math.sign(Exit)+1)/2), [num_sample]) #1 for Exit>=0, 0 for Exit<0
            # delta_x_sqrnorm = tf.reduce_sum(delta_x**2, 1, keepdims=False)
            # inner_product = tf.reduce_sum(delta_x * x_i, 1, keepdims=False)
            # discriminant = inner_product ** 2 - delta_x_sqrnorm * (tf.reduce_sum(x_i**2,1,keepdims=False)- self.R ** 2)
            coef_i = flag*(1-Exit) #+ flag * Exit * (tf.sqrt(tf.abs(discriminant)) - inner_product) / delta_x_sqrnorm
            if i==0:
                coef = tf.reshape(coef_i, [num_sample, 1])
            else:
                coef = tf.concat([coef, tf.reshape(coef_i, [num_sample, 1])], axis=1)
            x_i = x_i + delta_x * tf.reshape(coef_i, [num_sample,1])
            x_smp = tf.concat([x_smp, tf.reshape(x_i, [num_sample, self.dim, 1])], axis=2)
            flag = flag * (1 - Exit)
        return x_smp, coef
    
    def propagate1_tf(self, num_sample, x0, dw_sample, NN_control, training, T, N, cheat):
        # the most complicated scheme, need to solve a quartic equation to get the coefficient
        delta_t = T / N
        x_smp = tf.reshape(x0, [num_sample, self.dim, 1])
        x_i = x0
        flag = np.ones([num_sample])
        for i in range(N):
            if cheat:
                u = self.u_true(x_i)
            else:
                u = NN_control(x_i, training, need_grad=False)
            xi_norm_square = tf.reduce_sum(x_i**2,1,keepdims=True)
            xi_norm = tf.sqrt(xi_norm_square)
            co = 3 * (self.dim+1) * self.a3 / 2/self.a2 / self.dim / (2*self.a2 - 3*self.a3*xi_norm)
            delta_x_drift = co * u * delta_t
            delta_x_diffusion = self.sigma * dw_sample[:, :, i]
            delta_x = delta_x_drift + delta_x_diffusion
            x_iPlus1_temp = x_i + delta_x
            Exit = self.b_tf(x_iPlus1_temp) #Exit>=0 means out
            Exit = tf.reshape(tf.math.ceil((tf.math.sign(Exit)+1)/2), [num_sample]) #1 for Exit>=0, 0 for Exit<0
            coef_i_temp = (1 - Exit) * flag
            # find the index for boundary
            exit_index = tf.where(flag*Exit==1) # care the dim, num_exit x 1
            x_exit =tf.gather_nd(x_i,exit_index)
            delta_x_drift_exit = tf.gather_nd(delta_x_drift,exit_index)
            delta_x_diffusion_exit = tf.gather_nd(delta_x_diffusion,exit_index)
             # the following code is to explicitly solve a quartic equation [a,b,c,d,e]*[x^4,x^3,x^2,x,1]=0 to get sqrt_rho
            # we want to find rho s.t. x_i + drift * rho + diffusion * sqrt_rho is on the boundary
            a = tf.reduce_sum(delta_x_drift_exit**2, axis=1, keepdims=False) # shape = [num_sample] for a b c d e p q Delta_0 Delta_1
            b = 2 * tf.reduce_sum(delta_x_drift_exit * delta_x_diffusion_exit, axis=1, keepdims=False)
            c = tf.reduce_sum(2 * delta_x_drift_exit * x_exit + (delta_x_diffusion_exit**2), axis=1, keepdims=False)
            d = 2 * tf.reduce_sum(delta_x_diffusion_exit * x_exit, axis=1, keepdims=False)
            e = tf.reduce_sum(x_exit**2, axis=1, keepdims=False) - self.R**2
            p = (8*a*c - 3*(b**2)) / (8*(a**2))
            q = (b**3 - 4*a*b*c + 8*(a**2)*d) / (8 * (a**3))
            sign_q = tf.sign(q)
            Delta_0 = c**2 - 3*b*d + 12*a*e
            Delta_1 = 2*(c**3) - 9*b*c*d + 27*(b**2)*e + 27*a*(d**2) - 72*a*c*e
            Delta_2 = Delta_1**2 - 4*(Delta_0**3)
            sign_Delta_2 = tf.sign(Delta_2)
            signal_Delta_2 = tf.math.ceil((sign_Delta_2+1)/2)
            # this is for sign_Delta >= 0
            QQ = (Delta_1 + ( tf.abs(Delta_2) )**0.5 ) / 2 # the absolute value here is to make sure that no nan
            Q = tf.sign(QQ) * tf.abs(QQ)**(1/3) #I don't know why python cannot compute the cubic root of negative number
            S_plus = 0.5 * tf.abs((Q + Delta_0/Q) / (3*a) - 2*p/3)**0.5
            # This is for sign_Delta < 0
            phi = tf.math.acos( tf.clip_by_value( Delta_1 / 2/ tf.abs(Delta_0)**1.5,-1,1) )
            S_minus = 0.5 * tf.abs(2 * (tf.abs(Delta_0)**0.5) * tf.math.cos(phi/3) / (3*a) - 2*p/3)**0.5
            S = signal_Delta_2 * S_plus + (1-signal_Delta_2) * S_minus
            # either root1 or root3 is what we want.
            # root1 = 0.5 * (q/S - 4*(S**2) - 2*p)**0.5 - b/4/a - S
            # root2 = - 0.5 * (q/S - 4*(S**2) - 2*p)**0.5 - b/4/a - S
            # root3 = 0.5 * (-q/S - 4*(S**2) - 2*p)**0.5 - b/4/a + S
            # root4 = - 0.5 * (-q/S - 4*(S**2) - 2*p)**0.5 - b/4/a + S
            # root = tf.concat([[root1],[root2],[root3],[root4]],axis=0)
            # root = tf.transpose(root)
            temp = -4*(S**2) -2*p + tf.abs(q/S)
            sqrt_rho = 0.5 * tf.abs(temp)**0.5 - b/4/a - sign_q*S
            check_index = tf.where( (1-sqrt_rho) * sqrt_rho < 0) # same format as Exit_index
            new_temp = -4*(S**2) -2*p - tf.abs(q/S)
            new_sqrt_rho = 0.5 * tf.abs(new_temp)**0.5 - b/4/a + sign_q*S # new_temp should be positive at check_index
            new_sqrt_rho = tf.gather(new_sqrt_rho, check_index[:,0])
            sqrt_rho_final = tf.tensor_scatter_nd_update(sqrt_rho, check_index, new_sqrt_rho)
            # now we get the solutiob of the quartic equation and we can use the next line to check
            # result = a*(sqrt_rho**4) + b*(sqrt_rho**3) + c*(sqrt_rho**2) + d*sqrt_rho+e
            rho = sqrt_rho_final**2
            # rho = rho*0 + 0.000001
            # rho = rho * 0.000001
            coef_i = tf.tensor_scatter_nd_update(coef_i_temp, exit_index, rho)
            if i==0:
                coef = tf.reshape(coef_i, [num_sample, 1])
            else:
                coef = tf.concat([coef, tf.reshape(coef_i, [num_sample, 1])], axis=1)
            x_i = x_i + delta_x_drift * tf.reshape(coef_i, [num_sample,1]) + delta_x_diffusion * tf.reshape(coef_i**0.5, [num_sample,1])
            x_smp = tf.concat([x_smp, tf.reshape(x_i, [num_sample, self.dim, 1])], axis=2)
            flag = flag * (1 - Exit)
        # tf.print("x_smp", tf.reduce_prod(x_smp))
        # tf.print("coef", tf.reduce_sum(coef))
        return x_smp, coef
    
    def propagate2_tf(self, num_sample, x0, dw_sample, NN_control, training, T, N, cheat):
        delta_t = T / N
        sqrt_delta_t = np.sqrt(delta_t)
        x_smp = tf.reshape(x0, [num_sample, self.dim, 1])
        x_i = x0
        x0_norm = tf.sqrt(tf.reduce_sum(x0**2,1,keepdims=True))
        #temp: 2 for inside (inner); 0 (and 1) for middle layer; -2 (and -1) for boundary or outside
        temp = tf.sign(self.R - x0_norm - np.sqrt(6 * self.dim * delta_t)) + tf.sign(self.R - x0_norm - (delta_t**2))
        #flag: 2 for inside; 1 for middle layer, which means step size need modification; 0 means boundary, but we will move for at least a first step.
        flag = np.ones([num_sample,1]) + tf.math.floor(temp/2)
        for i in range(N):
            xi_norm_square = tf.reduce_sum(x_i**2,1,keepdims=True)
            xi_norm = tf.sqrt(xi_norm_square)
            dt_i = (2*flag - (flag**2)) * ((self.R - xi_norm)**2) / (3*self.dim) + (flag**2 - 2*flag + 1) * delta_t
            if cheat:
                u = self.u_true(x_i)
            else:
                u = NN_control(x_i, training, need_grad=False)
            c = 3 * (self.dim+1) * self.a3 / 2/self.a2 / self.dim / (2*self.a2 - 3*self.a3*xi_norm)
            delta_x = c*u*dt_i + self.sigma * dw_sample[:, :, i] * tf.sqrt(dt_i) / sqrt_delta_t
            x_iPlus1_temp = x_i + delta_x
            x_iPlus1_temp_norm = tf.sqrt(tf.reduce_sum(x_iPlus1_temp**2,1,keepdims=True))
            temp = tf.sign(self.R - x_iPlus1_temp_norm - np.sqrt(6 * self.dim * delta_t)) + tf.sign(self.R - x_iPlus1_temp_norm - (delta_t**2))
            new_flag = (np.ones([num_sample,1]) + tf.math.floor(temp/2)) * tf.sign(flag)
            # delta_x_sqrnorm = tf.reduce_sum(delta_x**2, 1, keepdims=True)
            # inner_product = tf.reduce_sum(delta_x * x_i, 1, keepdims=True)
            # discriminant = inner_product ** 2 - delta_x_sqrnorm * (xi_norm_square - self.R ** 2)
            # # if flag=0, then new_flag=0, coef=0, outside; if new_flag>0, then coef=1; else, coef is in (0,1) 
            # coef_i = tf.sign(new_flag) + tf.sign(flag) * (1 - tf.sign(new_flag) ) * (tf.sqrt(tf.abs(discriminant)) - inner_product) / delta_x_sqrnorm
            coef_i = tf.sign(flag) * tf.sign(new_flag)
            if i==0:
                coef = coef_i
                dt = dt_i
            else:
                coef = tf.concat([coef, coef_i], axis=1)
                dt = tf.concat([dt, dt_i], axis=1)
            x_i = x_i + delta_x * coef_i
            x_smp = tf.concat([x_smp, tf.reshape(x_i, [num_sample, self.dim, 1])], axis=2)
            flag = new_flag
        return x_smp, dt, coef

    def w_tf(self, x, u): #num_sample * 1
        return 0*tf.reduce_sum(x, 1, keepdims=True) + 1

    def Z_tf(self, x): #num_sample * 1
        return self.V_true(x)

    def b_np(self, x): #num_sample * 1
        return np.sum(x**2, 1, keepdims=True) - (self.R ** 2)
    
    def b_tf(self, x): #num_sample * 1
        return tf.reduce_sum(x**2, 1, keepdims=True) - (self.R ** 2)
    
    def V_true(self, x): #num_sample * 1
        x_norm = tf.reduce_sum(x**2, axis=1, keepdims=True)**0.5
        # x_norm = tf.norm(x, ord='euclidean', axis=1, keepdims=True)
        return self.a3*x_norm**3 - self.a2 * x_norm**2# + self.a2 - self.a3

    def u_true(self, x): #num_sample * 1
        x_norm = tf.reduce_sum(x**2, axis=1, keepdims=True)**0.5
        # x_norm = tf.norm(x, ord='euclidean', axis=1, keepdims=True)
        return x/x_norm
    
    def V_grad_true(self, x):
        x_norm = tf.reduce_sum(x**2, axis=1, keepdims=True)**0.5
        # x_norm = tf.norm(x, ord='euclidean', axis=1, keepdims=True)
        return (3*self.a3*x_norm - 2*self.a2) * x