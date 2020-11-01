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

# the trainable variable u
u = tf.Variable(initial_value = 1.0, trainable=True)

# sample x0 and dw, 
def sample(num_sample, T, N):
    return 0, 0

# SDE solver, return x_smp and coefficient
def propagate(num_sample, x0, dw_sample, T, N):
    return 0, 0

# compute the running cost, which is also the loss function
def integration(num_sample, x0, dw_sample, T, N):
    x_smp, coef = propagate(num_sample, x0, dw_sample, T, N)
    return 0

def grad(num_sample, x0, dw_sample, T, N):
    with tf.GradientTape(persistent=True) as tape:
        loss = integration(num_sample, x0, dw_sample, T, N) + u**2
    Grad = tape.gradient(loss, u)
    del tape
    return Grad

# define Adam optimizer
lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay([10], [1e-2,1e-3])
optimizer = tf.keras.optimizers.Adam(learning_rate=[1e-2, 1e-3], epsilon=1e-8)

# training
@tf.function
def train(num_sample, x0, dw_sample, T, N):
    gradient = grad(num_sample, x0, dw_sample, T, N)
    optimizer.apply_gradients(zip(gradient, u))
    
for i in range(2):
    x0, dw_sample = sample(num_sample, total_time, num_interval)
    #train(num_sample, x0, dw_sample, total_time, num_interval)
    gradient = grad(num_sample, x0, dw_sample, total_time, num_interval)
    u = u - 0.01 * gradient







