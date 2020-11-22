"""
The main file to run BSDE solver to solve parabolic partial differential equations (PDEs).

"""

import json
import munch
import os
import logging

from absl import app
from absl import flags
from absl import logging as absl_logging
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import equation as eqn
from solver import ActorCriticSolver

flags.DEFINE_string('config_path', 'configs/ekn_d10.json',
                    """The path to load json file.""")
flags.DEFINE_string('exp_name', 'ekn_10dTD3',
                    """The name of numerical experiments, prefix for logging""")
FLAGS = flags.FLAGS
FLAGS.log_dir = './logs'  # directory where to write event logs and output array

def main(argv):
    del argv
    with open(FLAGS.config_path) as json_data_file:
        config = json.load(json_data_file)
    config = munch.munchify(config)
    bsde = getattr(eqn, config.eqn_config.eqn_name)(config.eqn_config)
    tf.keras.backend.set_floatx(config.net_config.dtype)
    dim = config.eqn_config.dim
    control_dim = config.eqn_config.control_dim
    T = config.eqn_config.total_time_critic
    N = config.eqn_config.num_time_interval_critic
    R = config.eqn_config.R
    
    if not os.path.exists(FLAGS.log_dir):
        os.mkdir(FLAGS.log_dir)
    path_prefix = os.path.join(FLAGS.log_dir, FLAGS.exp_name)
    with open('{}_config.json'.format(path_prefix), 'w') as outfile:
        json.dump(dict((name, getattr(config, name))
                       for name in dir(config) if not name.startswith('__')),
                  outfile, indent=2)

    absl_logging.get_absl_handler().setFormatter(logging.Formatter('%(levelname)-6s %(message)s'))
    absl_logging.set_verbosity('info')

    logging.info('Begin to solve %s ' % config.eqn_config.eqn_name)
    ActorCritic_solver = ActorCriticSolver(config, bsde)
    training_history,x,y, true_y, z, true_z, grad_y = ActorCritic_solver.train()
    
    # r = np.sqrt(np.sum(np.square(x), 1, keepdims=False))
    # theta = np.arctan(x[:,1]/x[:,0])
    # theta2 = np.arctan(z[:,1]/z[:,0])
    # theta3 = np.arctan(true_z[:,1]/true_z[:,0])
    # theta4 = np.arctan(grad_y[:,1]/grad_y[:,0])
    # u_norm = np.sqrt(np.sum(np.square(z), 1, keepdims=False))
    # grad_y_norm = np.sqrt(np.sum(np.square(grad_y), 1, keepdims=False))
    # f1 = plt.figure()
    # ax1 = f1.add_subplot(111)
    # ax1.plot(r,y,'ro',label='value_r_V')
    # ax1.plot(r,true_y,'bo', label='true_value')
    # plt.legend()
    # f2 = plt.figure()
    # ax2 = f2.add_subplot(111)
    # ax2.plot(r,u_norm,'ro', label='control_r_|u|')
    # plt.legend()
    # f3 = plt.figure()#for angle of actor
    # ax3 = f3.add_subplot(111)
    # ax3.plot(theta,theta2,'ro', label='control_angle')
    # ax3 = f3.add_subplot(111)
    # ax3.plot(theta,theta3,'bo', label='true_angle')
    # plt.legend()
    # f4 = plt.figure()
    # plt.quiver(x[:,0],x[:,1],z[:,0],z[:,1])
    # plt.legend(["control field"])
    # f5 = plt.figure()
    # plt.quiver(x[:,0],x[:,1],true_z[:,0],true_z[:,1])
    # plt.legend(["true control field"])
    
    np.savetxt('{}_T{}_N{}_R{}.csv'.format(path_prefix,T,N,R),
               training_history,
               fmt=['%d', '%.5e', '%.5e', '%.5e', '%.5e', '%.5e', '%.5e', '%.5e', '%d'],
               delimiter=",",
               header='step, loss_critic, loss_actor,true_loss_actor, err_value, err_control, error_cost,error_cost2, elapsed_time',
               comments='')
    figure_data = np.concatenate([x,y, true_y, z, true_z], axis=1)
    head = ("x,")*dim + "y_NN,y_true," + ("Z_NN,")*control_dim + "z_true" + (",z_true")*(control_dim-1)
    np.savetxt('{}_T{}_N{}_R{}_hist.csv'.format(path_prefix,T,N,R), figure_data, delimiter=",",
               header=head, comments='')
if __name__ == '__main__':
    app.run(main)
