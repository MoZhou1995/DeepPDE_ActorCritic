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


flags.DEFINE_string('config_path', 'configs/VDP3_d10.json',
                    """The path to load json file.""")
flags.DEFINE_string('exp_name', 'VDP3_10dADMM3NN',
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
    training_history,x,y, true_y, z, true_z = ActorCritic_solver.train()
    # r = np.sqrt(np.sum(np.square(x), 1, keepdims=False))
    # theta = np.arctan(x[:,1]/x[:,0])
    # theta2 = np.arctan(z[:,1]/z[:,0])
    # theta3 = np.arctan(true_z[:,1]/true_z[:,0])
    # u_norm = np.sqrt(np.sum(np.square(z), 1, keepdims=False))
    # # print(x,y)
    # f1 = plt.figure()
    # ax1 = f1.add_subplot(111)
    # ax1.plot(r,y,'ro',label='value_r_V')
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
    #print(np.sign(x[:,0]/u[:,0]))
    # f4 = plt.figure() # for critic
    # temp = (x[:,0])**2 - (x[:,1])**2
    # ax4 = f4.add_subplot(111)
    # ax4.plot(temp,y,'ro', label='critic')
    # plt.legend()
    # f5 = plt.figure()
    # ax5 = f5.add_subplot(111)
    # ax5.plot(x[:,0]**2 - x[:,1]**2,z[:,0]**2 - z[:,1]**2,'ro')
    # ax5_2 = f5.add_subplot(111)
    # ax5_2.plot(x[:,0]**2 - x[:,1]**2,true_z[:,0]**2 - true_z[:,1]**2,'bo')
    # plt.legend()
    # f6 = plt.figure() # for critic
    # ax6 = f6.add_subplot(111)
    # ax6.plot(x[:,1],y,'ro', label='critic')
    # ax6_2 = f6.add_subplot(111)
    # ax6_2.plot(x[:,1],true_y,'bo', label='true')
    # plt.legend()
    # f7 = plt.figure()
    # ax7 = f7.add_subplot(111)
    # ax7.plot(x[:,1],z[:,0],'ro')
    # ax7_2 = f7.add_subplot(111)
    # ax7_2.plot(x[:,1],true_z[:,0],'bo')
    # plt.legend()
    # f8 = plt.figure()
    # ax8 = f8.add_subplot(111)
    # ax8.plot(x[:,1],z[:,1],'ro',label='NN')
    # ax8_2 = f8.add_subplot(111)
    # ax8_2.plot(x[:,1],true_z[:,1],'bo',label='true')
    # plt.legend()
    # plt.hist(true_y.numpy(), bins='auto')
    
    np.savetxt('{}_T{}_N{}_R{}.csv'.format(path_prefix,T,N,R),
               training_history,
               fmt=['%d', '%.5e', '%.5e', '%.5e', '%.5e', '%.5e', '%.5e', '%.5e', '%d'],
               delimiter=",",
               header='step, loss_critic, loss_actor,true_loss_actor, err_value, err_control, error_cost,error_cost2, elapsed_time',
               comments='')
    figure_data = np.concatenate([x,y, true_y, z, true_z], axis=1)
    head = "x" + (",")*dim + "y_NN,y_true,z_NN" + (",")*control_dim + "z_true"
    np.savetxt('{}_T{}_N{}_R{}_hist.csv'.format(path_prefix,T,N,R), figure_data, delimiter=",",
               header=head, comments='')
if __name__ == '__main__':
    app.run(main)
