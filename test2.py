"""
This code is to test a controled SDE dX_t = u*X_t dt + d W_t in 2d
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

# parameters
total_time = 1.0
dim = 2
num_interval = 2
num_sample = 32

# sample x0 and dw, 
def sample(num_sample, dim, T, N):
    print("call sample")
    return 0, 0

# SDE solver, return x_smp and coef, this is the key point to test
def propagate(num_sample, dim, x0, dw_sample, T, N):
    return 0, 0

class solver(object):
    def __init__(self, total_time, dim):
        self.T = total_time
        self.dim = dim
        self.lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay([10], [1e-2,1e-3])
        # define the model
        self.model = model(total_time, dim)
        # define Adam optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule, epsilon=1e-8)
    
    def train(self, steps):
        for step in range(steps):
            print("step ", step)
            x0, dw_sample = sample(num_sample, self.dim, self.T, num_interval)
            self.train_step(num_sample, x0, dw_sample, self.T, num_interval)

    def grad(self, num_sample, x0, dw_sample, T, N):
        print("call grad")
        with tf.GradientTape(persistent=True) as tape:
            loss = self.model(x0, dw_sample) + self.model.u**2
        print("trainable variable", self.model.trainable_variables)
        Grad = tape.gradient(loss, self.model.trainable_variables)
        print("Grad",Grad)
        del tape
        return Grad
    #define training
    @tf.function
    def train_step(self, num_sample, x0, dw_sample, T, N):
        print("call train_step")
        gradient = self.grad(num_sample, x0, dw_sample, T, N)
        self.optimizer.apply_gradients(zip(gradient, self.model.trainable_variables))
    
class model(tf.keras.Model):
    def __init__(self, total_time, dim):
        super(model, self).__init__()
        # the trainable variable u
        self.u = tf.Variable(initial_value = [1.0], trainable=True)
        self.T = total_time
        self.dim = dim
        
    def call(self, x0, dw):
        print(self.u)
        # compute the running cost, which is also the loss function
        return 0
    
# the following is the main function
Solver = solver(total_time, dim)    
Solver.train(3)







