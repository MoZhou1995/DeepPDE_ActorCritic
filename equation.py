import numpy as np
import tensorflow as tf
from scipy.stats import multivariate_normal as normal

class Equation(object):
    """Base class for defining PDE related function."""
    def __init__(self, eqn_config):
        self.dim = eqn_config.dim
        self.gamma = eqn_config.discount
        self.R = eqn_config.R
        self.control_dim = eqn_config.control_dim
        
    def sample_normal(self, num_sample, N): #normal sample for BM
        r_Sample = np.random.uniform(low=0, high=self.R, size=[num_sample,1])
        r = r_Sample**(1 / self.dim) * (self.R**((self.dim- 1) / self.dim ))
        angle = normal.rvs(size=[num_sample, self.dim])
        norm = np.sqrt(np.sum(angle**2, 1, keepdims=True))
        x0 = r * angle / norm
        dw_sample = normal.rvs(size=[num_sample, self.dim, N])# * sqrt_delta_t
        x_bdry = normal.rvs(size=[num_sample, self.dim])
        norm = np.sqrt(np.sum(np.square(x_bdry), 1, keepdims=True))
        x_bdry = self.R * x_bdry / norm
        return x0, dw_sample, x_bdry
    
    def sample_bounded(self, num_sample, N): #bdd sample for BM
        r_Sample = np.random.uniform(low=0, high=self.R, size=[num_sample,1])
        r = r_Sample**(1 / self.dim) * (self.R**((self.dim- 1) / self.dim ))
        angle = normal.rvs(size=[num_sample, self.dim])
        norm = np.sqrt(np.sum(angle**2, 1, keepdims=True))
        x0 = r * angle / norm
        dw_sample = np.random.randint(6,size=[num_sample, self.dim, N])
        dw_sample = np.floor((dw_sample - 1)/4) * np.sqrt(3.0) 
        x_bdry = normal.rvs(size=[num_sample, self.dim])
        norm = np.sqrt(np.sum(np.square(x_bdry), 1, keepdims=True))
        x_bdry = self.R * x_bdry / norm
        return x0, dw_sample, x_bdry
    
    def sample0(self, num_sample, N):
        x0 = np.zeros(shape=[num_sample, self.dim]) + 0.01
        dw_sample = normal.rvs(size=[num_sample, self.dim, N])
        x_bdry = normal.rvs(size=[num_sample, self.dim])
        norm = np.sqrt(np.sum(np.square(x_bdry), 1, keepdims=True))
        x_bdry = self.R * x_bdry / norm
        return x0, dw_sample, x_bdry
        
    def propagate_naive(self, num_sample, x0, dw_sample, NN_control, training, T, N, cheat):
        # the most naive scheme, just stop where next step is out
        delta_t = T / N
        sqrt_delta_t = np.sqrt(delta_t)
        x_smp = tf.reshape(x0, [num_sample, self.dim, 1])
        x_i = x0
        flag = np.ones([num_sample])
        for i in range(N):
            if cheat:
                u_i = self.u_true(x_i)
            else:
                u_i = NN_control(x_i, training, need_grad=False)
            delta_x = self.drift(x_i, u_i) * delta_t + self.diffusion(x_i, dw_sample[:, :, i]) * sqrt_delta_t
            x_iPlus1_temp = x_i + delta_x
            Exit = self.b_tf(x_iPlus1_temp) #Exit>=0 means out
            Exit = tf.reshape(tf.math.ceil((tf.math.sign(Exit)+1)/2), [num_sample]) #1 for Exit>=0, 0 for Exit<0
            coef_i = flag * (1 - Exit)
            if i==0:
                coef = tf.reshape(coef_i, [num_sample, 1])
            else:
                coef = tf.concat([coef, tf.reshape(coef_i, [num_sample, 1])], axis=1)
            x_i = x_i + delta_x * tf.reshape(coef_i, [num_sample,1])
            x_smp = tf.concat([x_smp, tf.reshape(x_i, [num_sample, self.dim, 1])], axis=2)
            flag = flag * (1 - Exit)
        dt = np.ones([num_sample,N]) * delta_t
        return x_smp, dt, coef
    
    def propagate_adaptive(self, num_sample, x0, dw_sample, NN_control, training, T, N, cheat):
        # the new scheme
        delta_t = T / N
        sqrt_delta_t = np.sqrt(delta_t)
        x_smp = tf.reshape(x0, [num_sample, self.dim, 1])
        x_i = x0
        x0_norm = tf.sqrt(tf.reduce_sum(x0**2,1))
        #temp: 2 for inside (inner); 0 (and 1) for boundary layer; -2 (and -1) for outside
        temp = tf.sign(self.R - x0_norm - self.sigma*np.sqrt(3 * self.dim * delta_t)) + tf.sign(self.R - x0_norm)
        #flag: 2 for inside; 1 means step size need modification; 0 means boundary, but we will move for at least a first step.
        flag = np.ones([num_sample]) + tf.math.floor(temp/2)
        for i in range(N):
            xi_norm = tf.sqrt(tf.reduce_sum(x_i**2,1))
            dt_i = (2*flag - (flag**2)) * ((self.R - xi_norm)**2) / (3 * self.dim * self.sigma**2) + (flag**2 - 2*flag + 1) * delta_t
            dt_i = tf.maximum(dt_i, delta_t*1e-4)
            if cheat:
                u_i = self.u_true(x_i)
            else:
                u_i = NN_control(x_i, training, need_grad=False)
            delta_x = self.drift(x_i, u_i) * tf.reshape(dt_i, [num_sample,1]) + self.diffusion(x_i, dw_sample[:, :, i]) * tf.reshape(tf.sqrt(dt_i), [num_sample,1])
            x_iPlus1_temp = x_i + delta_x
            x_iPlus1_temp_norm = tf.sqrt(tf.reduce_sum(x_iPlus1_temp**2,1,keepdims=False))
            temp = tf.sign(self.R - x_iPlus1_temp_norm - self.sigma*np.sqrt(3 * self.dim * delta_t)) + tf.sign(self.R - x_iPlus1_temp_norm)
            new_flag = (np.ones([num_sample]) + tf.math.floor(temp/2)) * tf.sign(flag)
            coef_i = tf.sign(flag) * tf.sign(new_flag)
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

    def w_tf(self, x, u):
        """Running cost in control problems."""
        raise NotImplementedError

    def Z_tf(self, x):
        """Terminal cost in control problems."""
        raise NotImplementedError
        
    def b_np(self, x): #num_sample * 1
        """a function whose level set is the boundary, in numpy"""
        return np.sum(x**2, 1, keepdims=True) - (self.R ** 2)
    
    def b_tf(self, x): #num_sample * 1
        """a function whose level set is the boundary, in tensorflow"""
        return tf.reduce_sum(x**2, 1, keepdims=True) - (self.R ** 2)
        
    def V_true(self, x):
        """True value function"""
        raise NotImplementedError
    
    def u_true(self, x):
        """Optimal control"""
        raise NotImplementedError
        
    def drift(self, x, u):
        """drift in the SDE"""
        raise NotImplementedError
    
    def diffusion(self, x):
        """diffusion in the SDE"""
        raise NotImplementedError

class LQR(Equation):
    """linear quadratic regulator"""
    def __init__(self, eqn_config):
        super(LQR, self).__init__(eqn_config)
        self.sigma = np.sqrt(2.0)
        self.p = eqn_config.p
        self.q = eqn_config.q
        self.beta = eqn_config.beta
        self.k = ( ((self.gamma**2) * (self.q**2) + 4 * self.p * self.q * (self.beta**2))**0.5 - self.q*self.gamma )/ (self.beta**2) / 2
    
    def w_tf(self, x, u): #num_sample * 1
        return tf.reduce_sum(self.p * tf.square(x) + self.q * tf.square(u), 1, keepdims=True) - 2*self.k*self.dim

    def Z_tf(self, x): #num_sample * 1
        return 0 * tf.reduce_sum(x, 1, keepdims=True) + self.k * (self.R ** 2)

    def V_true(self, x): #num_sample * 1
        return tf.reduce_sum(tf.square(x), 1, keepdims=True) * self.k

    def u_true(self, x): #num_sample * dim
        return -self.beta * self.k / self.q * x
    
    def V_grad_true(self, x): #num_sample * dim
        return 2 * self.k * x
    
    def drift(self, x, u):
        return self.beta * u
    
    def diffusion(self, x, dw):
        return self.sigma * dw
    
    
class VDP(Equation):
    """Van Der Pol oscillator"""
    def __init__(self, eqn_config):
        super(VDP, self).__init__(eqn_config)
        self.sigma = np.sqrt(2.0)
        self.a = eqn_config.a
        self.epsl = eqn_config.epsilon
        self.q = eqn_config.q
    
    def w_tf(self, x, u): #num_sample * 1
        d = self.control_dim # dim/2
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
    
    def V_true(self, x): #num_sample * 1
        d = self.control_dim # dim/2
        x1 = x[:,0:d] #num_sample * d
        x2 = x[:,d:self.dim] #num_sample * d
        px1 = tf.concat([x1[:,1:d],x1[:,0:1]],1) #num_sample * d
        px2 = tf.concat([x2[:,1:d],x2[:,0:1]],1) #num_sample * d
        return self.a*tf.reduce_sum(x**2, 1, keepdims=True) - self.epsl*tf.reduce_sum(x1*px1 + x2*px2, 1, keepdims=True)

    def u_true(self, x): #num_sample * 1
        d = self.control_dim
        x2 = x[:,d:self.dim] #num_sample * d
        px2 = tf.concat([x2[:,1:d],x2[:,0:1]],1)
        nx2 = tf.concat([x2[:,d-1:d],x2[:,0:d-1]],1)
        return -(2*self.a*x2 - self.epsl*(px2 + nx2))/2/self.q
    
    def V_grad_true(self, x): #num_sample * dim
        d = self.control_dim # dim/2
        x1 = x[:,0:d] #num_sample * d
        x2 = x[:,d:self.dim] #num_sample * d
        px1 = tf.concat([x1[:,1:d],x1[:,0:1]],1) #num_sample * d
        px2 = tf.concat([x2[:,1:d],x2[:,0:1]],1) #num_sample * d
        nx1 = tf.concat([x1[:,d-1:d],x1[:,0:d-1]],1)
        nx2 = tf.concat([x2[:,d-1:d],x2[:,0:d-1]],1)
        return tf.concat([2*self.a*x1 - self.epsl*(px1+nx1), 2*self.a*x2 - self.epsl*(px2+nx2)],axis=1)
    
    def drift(self, x, u):
        x_1 = x[:,0:self.control_dim] #num_sample * d
        x_2 = x[:,self.control_dim:self.dim]
        return tf.concat([x_2, (1 - x_1**2)*x_2 - x_1 + u],axis=1)
    
    def diffusion(self, x, dw):
        return self.sigma * dw

class EKN(Equation):
    """Diffusive Eikonal equation"""
    def __init__(self, eqn_config):
        super(ekn, self).__init__(eqn_config)
        self.a2 = eqn_config.a2
        self.a3 = eqn_config.a3
        self.epsl = 1/2/self.a2/self.dim
        self.sigma = np.sqrt(self.epsl) * np.sqrt(2.0)

    def w_tf(self, x, u): #num_sample * 1
        return 0*tf.reduce_sum(x, 1, keepdims=True) + 1

    def Z_tf(self, x): #num_sample * 1
        return self.V_true(x)

    def V_true(self, x): #num_sample * 1
        x_norm = tf.reduce_sum(x**2, axis=1, keepdims=True)**0.5
        return self.a3*x_norm**3 - self.a2 * x_norm**2# + self.a2 - self.a3

    def u_true(self, x): #num_sample * 1
        x_norm = tf.reduce_sum(x**2, axis=1, keepdims=True)**0.5
        return x/x_norm
    
    def V_grad_true(self, x):
        x_norm = tf.reduce_sum(x**2, axis=1, keepdims=True)**0.5
        return (3*self.a3*x_norm - 2*self.a2) * x
    
    def drift(self, x, u):
        x_norm = tf.reduce_sum(x**2, axis=1, keepdims=True)**0.5
        c = 3 * (self.dim+1) * self.a3 / 2/self.a2 / self.dim / (2*self.a2 - 3*self.a3*x_norm)
        return c * u
    
    def diffusion(self, x, dw):
        return self.sigma * dw
