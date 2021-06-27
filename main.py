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

flags.DEFINE_string('config_path', 'configs/lqr_d5.json',
                    """The path to load json file.""")
flags.DEFINE_string('exp_name', None,
                    """The name of numerical experiments, prefix for logging""")
FLAGS = flags.FLAGS
FLAGS.log_dir = './logs'  # directory where to write event logs and output array

def main(argv):
    del argv
    if FLAGS.exp_name is None: # use config name as exp_name
        FLAGS.exp_name = os.path.splitext(os.path.basename(FLAGS.config_path))[0]
    with open(FLAGS.config_path) as json_data_file:
        config = json.load(json_data_file)
    config = munch.munchify(config)
    bsde = getattr(eqn, config.eqn_config.eqn_name)(config.eqn_config)
    tf.keras.backend.set_floatx(config.net_config.dtype)
    dim = config.eqn_config.dim
    control_dim = config.eqn_config.control_dim
    sample = config.train_config.sample_type
    scheme = config.train_config.scheme
    TD = config.train_config.TD_type
    train = config.train_config.train
    
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
    config.eqn_config.total_time_critic = 0.04
    config.eqn_config.total_time_actor = 0.04
    ActorCritic_solver = ActorCriticSolver(config, bsde)
    training_history,x,y, true_y, z, true_z, grad_y = ActorCritic_solver.train()
    
    
    char = "T0.04"
    np.savetxt('{}_{}.csv'.format(path_prefix,char),
               training_history,
               fmt=['%d', '%.5e', '%.5e', '%.5e', '%.5e', '%.5e', '%.5e', '%.5e', '%d'],
               delimiter=",",
               header='step, loss_critic, loss_actor, err_value, error_value_infty, err_control, err_value_grad,error_cost2, elapsed_time',
               comments='')
    figure_data = np.concatenate([x,y, true_y, z, true_z], axis=1)
    head = ("x,")*dim + "y_NN,y_true," + ("Z_NN,")*control_dim + "z_true" + (",z_true")*(control_dim-1)
    np.savetxt('{}_{}_hist.csv'.format(path_prefix, char), figure_data, delimiter=",",
               header=head, comments='')
if __name__ == '__main__':
    app.run(main)
