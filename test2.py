"""
This code is to test a controled SDE dX_t = u*X_t dt + sqrt(2) d W_t in 2d
on the unit circle. Let tau be the stopping time that X_t exit the 
circle. We want to compute the stopping time with killed diffusion
algorithm, which means that we need to solve a quartic equation. 
Then we want to compute the integration of a cost function. u is 
considered as a trainable parameter and we want to use gradient based
method to train it. 

@author: Mo Zhou
"""
import numpy as np
import tensorflow as tf
from scipy.stats import multivariate_normal as normal
tf.keras.backend.set_floatx("float64")
# parameters
total_time = 0.2
Dim = 2
num_interval = 50
Num_sample = 256
R = 1.0
gamma = 1.0
sigma = np.sqrt(2.0)

# function for boundary
def b_np(x): #num_sample * 1
    return np.sum(x**2, 1, keepdims=True) - (R**2)
def b_tf(x): #num_sample * 1
    return tf.reduce_sum(x**2, 1, keepdims=True) - (R**2)

# sample x0 and dw, 
def sample(num_sample, dim, T, N):
    #print("call sample")
    delta_t = T / N
    sqrt_delta_t = np.sqrt(delta_t)
    x0 = np.zeros(shape=[0, dim])
    while np.shape(x0)[0] < num_sample:
        x_Sample = np.random.uniform(low=-R, high=R, size=[num_sample*dim,dim])
        index = np.where(b_np(x_Sample) < 0)
        x0 = np.concatenate([x0, x_Sample[index[0],:]], axis=0)
        # #print(np.shape(x0)[0])
        if np.shape(x0)[0] > num_sample:
            x0 = x0[0:num_sample,:]
    dw_sample = normal.rvs(size=[num_sample,dim,N]) * sqrt_delta_t
    return x0, dw_sample

# SDE solver, return x_smp and coef, this is the key point to test
def propagate(x0, dw_sample, T, N):
    num_sample = np.shape(x0)[0]
    dim = np.shape(x0)[1]
    delta_t = T / N
    x_smp = tf.reshape(x0, [num_sample, dim, 1])
    x_i = x0
    flag = np.ones([num_sample])
    for i in range(N):
        delta_x_drift = Solver.model.u * x_i * delta_t
        delta_x_diffusion = sigma * dw_sample[:, :, i]
        delta_x = delta_x_drift + delta_x_diffusion
        x_iPlus1_temp = x_i + delta_x
        Exit = b_tf(x_iPlus1_temp) #Exit>=0 means out
        #print("Exit",Exit)#,tf.strings.to_hash_bucket_fast(Exit,1))
        Exit = tf.reshape(tf.math.ceil((tf.math.sign(Exit)+1)/2), [num_sample])
        coef_i_temp = (1 - Exit) * flag
        #print("coef_i_temp",coef_i_temp)
        exit_index = tf.where(flag*Exit==1) # care the dim, num_exit x 1
        num_exit = tf.shape(exit_index)[0]
        # def no_exit():
        #     return coef_i_temp
        def have_exit():
            x_exit =tf.gather_nd(x_i,exit_index)
            delta_x_drift_exit = tf.gather_nd(delta_x_drift,exit_index)
            delta_x_diffusion_exit = tf.gather_nd(delta_x_diffusion,exit_index)
             # the following code is to explicitly solve a quartic equation [a,b,c,d,e]*[x^4,x^3,x^2,x,1]=0 to get sqrt_rho
            # we want to find rho s.t. x_i + drift * rho + diffusion * sqrt_rho is on the boundary
            a = tf.reduce_sum(delta_x_drift_exit**2, axis=1, keepdims=False) # shape = [num_exit] for a b c d e p q Delta_0 Delta_1
            b = 2 * tf.reduce_sum(delta_x_drift_exit * delta_x_diffusion_exit, axis=1, keepdims=False)
            c = tf.reduce_sum(2 * delta_x_drift_exit * x_exit + (delta_x_diffusion_exit**2), axis=1, keepdims=False)
            d = 2 * tf.reduce_sum(delta_x_diffusion_exit * x_exit, axis=1, keepdims=False)
            e = tf.reduce_sum(x_exit**2, axis=1, keepdims=False) - R**2
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
            phi = tf.math.acos( tf.math.minimum( tf.abs(Delta_1**2 / 4/ Delta_0**3)**0.5,1) )
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
            check_index = tf.where( (1-sqrt_rho) * sqrt_rho < 0)
            #print("sqrt_rho", sqrt_rho)
            # rho = sqrt_rho**2
            def every_point_good():
                return sqrt_rho
            def some_point_not_good():
                new_temp = -4*(S**2) -2*p - tf.abs(q/S)
                new_sqrt_rho = 0.5 * tf.abs(new_temp)**0.5 - b/4/a + sign_q*S # new_temp should be positive at check_index
                new_sqrt_rho = tf.gather(new_sqrt_rho, check_index[:,0])
                result_some_point_not_good = tf.tensor_scatter_nd_update(sqrt_rho, check_index, new_sqrt_rho)
                #print("result_some_point_not_good", result_some_point_not_good)
                return result_some_point_not_good
            sqrt_rho_final = tf.cond(tf.shape(check_index)[0] > 0, some_point_not_good, every_point_good)
            rho = sqrt_rho_final**2
            result_have_exit = tf.tensor_scatter_nd_update(coef_i_temp, exit_index, rho)
            #print("result_have_exit",result_have_exit)
            return result_have_exit
        #coef_i = tf.cond(num_exit>0, have_exit, no_exit)
        coef_i = have_exit()
        #print("coef_i", coef_i)
        if i==0:
            coef = tf.reshape(coef_i, [num_sample, 1])
        else:
            coef = tf.concat([coef, tf.reshape(coef_i, [num_sample, 1])], axis=1)
            # #print(tf.reduce_sum(coef_i))
            # #print(tf.reduce_sum(x_i))
            x_i = x_i + delta_x_drift * tf.reshape(coef_i, [num_sample,1]) + delta_x_diffusion * tf.reshape(coef_i**0.5, [num_sample,1])
            x_smp = tf.concat([x_smp, tf.reshape(x_i, [num_sample, dim, 1])], axis=2)
            flag = flag * (1 - Exit)
    return x_smp, coef

class solver(object):
    def __init__(self, total_time, dim):
        self.T = total_time
        self.dim = Dim
        self.lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay([100], [1e-1,1e-2])
        # define the model
        self.model = model(total_time, dim)
        self.num_sample = Num_sample
        # define Adam optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule, epsilon=1e-8)
    
    def train(self, steps):
        for step in range(steps):
            #print("step ", step)
            x0, dw_sample = sample(self.num_sample, self.dim, self.T, num_interval)
            loss = tf.reduce_mean(self.model(x0, dw_sample)).numpy()
            self.train_step(self.num_sample, x0, dw_sample, self.T, num_interval)
            print("step ",step,"u", self.model.u.numpy(), "loss", loss)

    def grad(self, num_sample, x0, dw_sample, T, N):
        # #print("call grad")
        with tf.GradientTape(persistent=True) as tape:
            loss = tf.reduce_mean(self.model(x0, dw_sample)) #+ self.model.u**2
        # #print("trainable variable", self.model.trainable_variables)
        Grad = tape.gradient(loss, self.model.trainable_variables)
        # #print("Grad",Grad)
        del tape
        return Grad
    #define training
    @tf.function
    def train_step(self, num_sample, x0, dw_sample, T, N):
        # #print("call train_step")
        gradient = self.grad(num_sample, x0, dw_sample, T, N)
        self.optimizer.apply_gradients(zip(gradient, self.model.trainable_variables))
    
class model(tf.keras.Model):
    def __init__(self, total_time, dim):
        super(model, self).__init__()
        # the trainable variable u
        self.u = tf.Variable(initial_value = -1.0, trainable=True, dtype="float64")
        #print(self.u)
        self.T = total_time
        self.dim = Dim
        self.N = num_interval
        self.num_sample = Num_sample
        self.gamma = gamma
        
    def call(self, x0, dw):
        x, coef = propagate(x0, dw, self.T, self.N)
        # compute the running cost, which is also the loss function
        delta_t = self.T / self.N
        y = 0
        discount = 1 #broadcast to num_sample x 1
        for t in range(self.N):
            # running cost
            w = (self.u**2 + 2) * tf.reduce_sum((x[:,:,t])**2, 1, keepdims=True) - 2 * self.dim
            y = y + coef[:,t:t+1] * w * delta_t * discount
            discount *= tf.math.exp(-self.gamma * delta_t * coef[:,t:t+1])
        y = y + discount * tf.reduce_sum((x[:,:,-1])**2, 1, keepdims=True) # add the true value
        return y #num_sample x dim
    
# the following is the main function
Solver = solver(total_time, Dim)    
Solver.train(40)







