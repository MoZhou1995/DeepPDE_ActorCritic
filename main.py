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


flags.DEFINE_string('config_path', 'configs/lqr_d2.json',
                    """The path to load json file.""")
flags.DEFINE_string('exp_name', 'test',
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
    num = 100
    for lmbd in [0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4]:
        loss_act=np.zeros([num])
        for index in range(num):
            #ActorCritic_solver = ActorCriticSolver(config, bsde)
            ActorCritic_solver = ActorCriticSolver(config, bsde, lmbd)
            training_history,x,y,u,loss_actor = ActorCritic_solver.train()
            loss_act[index] = loss_actor
        print("lmbd", lmbd, "loss_actor", np.mean(loss_act))
    # r = np.sqrt(np.sum(np.square(x), 1, keepdims=False))
    # theta = np.arctan(x[:,1]/x[:,0])
    # theta2 = np.arctan(u[:,1]/u[:,0])
    # u_norm = np.sqrt(np.sum(np.square(u), 1, keepdims=False))
    #print(x,y)
    # f1 = plt.figure()
    # ax1 = f1.add_subplot(111)
    # ax1.plot(r,y,'ro',label='value_r_V')
    # plt.legend()
    # f2 = plt.figure()
    # ax2 = f2.add_subplot(111)
    # ax2.plot(r,u_norm,'ro', label='control_r_|u|')
    # plt.legend()
    # f3 = plt.figure()
    # ax3 = f3.add_subplot(111)
    # ax3.plot(theta,theta2,'ro', label='control_angle')
    # plt.legend()
    #print(np.sign(x[:,0]/u[:,0]))
    np.savetxt('{}_training_history.csv'.format(path_prefix),
               training_history,
               fmt=['%d', '%.5e', '%.5e', '%.5e', '%.5e', '%d'],
               delimiter=",",
               header='step, loss_critic, loss_actor, err_value, err_control, elapsed_time',
               comments='')

if __name__ == '__main__':
    app.run(main)
