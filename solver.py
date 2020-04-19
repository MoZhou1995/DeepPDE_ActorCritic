import logging
import time
import numpy as np
import tensorflow as tf

DELTA_CLIP = 50.0


class ActorCriticSolver(object):
    """The fully connected neural network model."""
    def __init__(self, config, bsde):
        print("define AC solver")
        self.eqn_config = config.eqn_config
        self.net_config = config.net_config
        self.bsde = bsde
        self.model_critic = CriticModel(config, bsde)
        self.model_actor = ActorModel(config, bsde)
        #self.y_init = self.model.y_init
        lr_schedule_critic = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            self.net_config.lr_boundaries_critic, self.net_config.lr_values_critic)
        lr_schedule_actor = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            self.net_config.lr_boundaries_actor, self.net_config.lr_values_actor)
        self.optimizer_critic = tf.keras.optimizers.Adam(learning_rate=lr_schedule_critic, epsilon=1e-8)
        self.optimizer_actor = tf.keras.optimizers.Adam(learning_rate=lr_schedule_actor, epsilon=1e-8)
        self.x = None
        print("finish AC solver initialization")
        #self.control = None
        
    def train(self):
        print("start training")
        start_time = time.time()
        training_history = []
        valid_data = self.bsde.sample(self.net_config.valid_size, control_fcn=self.control_fcn)

        # begin sgd iteration
        for step in range(self.net_config.num_iterations+1):
            print("train step", step)
            if step % self.net_config.logging_frequency == 0:
                loss_critic = self.loss_critic(valid_data, training=False).numpy()
                loss_actor = self.loss_actor(valid_data, training=False).numpy()
                err_value = self.err_value(valid_data).numpy()
                err_control = self.err_control(valid_data).numpy()
                elapsed_time = time.time() - start_time
                training_history.append([step, loss_critic, loss_actor, err_value, err_control, elapsed_time])
                if self.net_config.verbose:
                    logging.info("step: %5u, loss_critic: %.4e, loss_actor: %.4e, err_value: %.4e, err_control: %.4e,  elapsed time: %3u" % (
                        step, loss_critic, loss_actor, err_value, err_control, elapsed_time))
            self.train_step_critic(self.bsde.sample(self.net_config.batch_size, control_fcn=self.control_fcn))
            self.train_step_actor(self.bsde.sample(self.net_config.batch_size, control_fcn=self.control_fcn))
        return np.array(training_history)

    def loss_critic(self, inputs, training):
        print("call loss critic")
        dw, x, x_tau, tau, Exit = inputs
        delta = self.model_critic(inputs, self.model_actor, training)
        # use linear approximation outside the clipped range
        loss = tf.reduce_mean(tf.where(tf.abs(delta) < DELTA_CLIP, tf.square(delta),
                                       2 * DELTA_CLIP * tf.abs(delta) - DELTA_CLIP ** 2))

        return loss
    
    def loss_actor(self, inputs, training):
        print("call loss actor")
        dw, x, x_tau, tau, Exit = inputs
        y = self.model_actor(inputs, self.model_critic, training)
        # use linear approximation outside the clipped range
        loss = tf.reduce_mean(y)
        return loss

    def grad_critic(self, inputs, training):
        print("call grad critic")
        with tf.GradientTape(persistent=True) as tape:
            loss = self.loss_critic(inputs, training)
        grad = tape.gradient(loss, self.model_critic.NN_value.variables)
        del tape
        return grad
    
    def grad_actor(self, inputs, training):
        print("call grad actor")
        with tf.GradientTape(persistent=True) as tape:
            loss = self.loss_actor(inputs, training)
        grad = tape.gradient(loss, self.model_actor.NN_control.variables)
        del tape
        return grad

    @tf.function
    def train_step_critic(self, train_data):
        print("call train step critic")
        grad = self.grad_critic(train_data, training=True)
        self.optimizer_critic.apply_gradients(zip(grad, self.model_critic.trainable_variables))
        
    @tf.function
    def train_step_actor(self, train_data):
        print("call train step actor")
        grad = self.grad_actor(train_data, training=True)
        self.optimizer_actor.apply_gradients(zip(grad, self.model_actor.trainable_variables))
        
    #@tf.function
    def control_fcn(self, x):
        print("call control function")
        return self.model_actor.NN_control(x, training=True)
        #return self.sess.run(self.control, feed_dict = {self.x: x})
        
    def err_value(self, inputs):
        print("call value error")
        dw, x, x_tau, tau, Exit = inputs
        x0 = x[:,:,0]
        return tf.reduce_mean(tf.square(self.bsde.V_true(x0) - self.model_critic.NN_value(x0, training=False)))
         
    def err_control(self, inputs):
        print("call control error")
        dw, x, x_tau, tau, Exit = inputs
        x0 = x[:,:,0]
        return tf.reduce_mean(tf.square(self.bsde.u_true(x0) - self.model_actor.NN_control(x0, training=False)))
        
class CriticModel(tf.keras.Model):
    def __init__(self, config, bsde):
        super(CriticModel, self).__init__()
        print("build critic model")
        self.eqn_config = config.eqn_config
        self.net_config = config.net_config
        self.bsde = bsde
        self.NN_value = DeepNN(config, "critic")

    def call(self, inputs, model_actor, training):
        dw, x, x_tau, tau, ExitIndex = inputs
        #all_zero_vec = tf.zeros(shape=tf.stack([tf.shape(dw)[0], 1]), dtype=self.net_config.dtype)
        #y = all_zero_vec
        num_sample = np.shape(dw)[0]
        y = np.zeros([num_sample])
        print("call critic model")
        for j in range(num_sample):
            print("critic index",j)
            for t in range(self.bsde.num_time_interval):
                print("critic step",t)
            #for j in range(num_sample):
                if t < ExitIndex[j]:
                    # maybe reshape is needed to call NN_control
                    print("critic in")
                    x_jt = tf.reshape(x[j,:,t], [1,self.bsde.dim])
                    y[j] = y[j] + self.bsde.w_tf(x_jt, model_actor.NN_control(x_jt, training)) * self.bsde.delta_t
                elif t == ExitIndex[j]:
                    print("critic out")
                    x_jt = tf.reshape(x[j,:,t], [1,self.bsde.dim])
                    y[j] = y[j] + self.bsde.w_tf(x_jt, model_actor.NN_control(x_jt, training)) * (tau[j] - t*self.bsde.delta_t)
            # currently use boundary condition instead of NN if X_tau reaches boundary
            if ExitIndex[j] < self.bsde.num_time_interval:
                print("critic boundary")
                # the size of Z_tf is currently num_sample, not num_sample * 1
                y[j] = y[j] + self.bsde.Z_tf(tf.reshape(x_tau[j,:], [1, self.eqn_config.dim]))
            else:
                print("critic inside")
                y[j] = y[j] + self.NN_value(tf.reshape(x_tau[j,:], [1, self.eqn_config.dim]), training)
            #y[j] = y[j] + self.bsde.Z_tf(tf.reshape(x_tau[j,:], [1, self.eqn_config.dim]))
        delta = y - self.NN_value(x[:,:,0], training)
        print("critic model computed")
        return delta


class ActorModel(tf.keras.Model):
    def __init__(self, config, bsde):
        super(ActorModel, self).__init__()
        print("build actor model")
        self.eqn_config = config.eqn_config
        self.net_config = config.net_config
        self.bsde = bsde
        self.NN_control = DeepNN(config, "actor")

    def call(self, inputs, model_critic, training):
        print("call actor model")
        dw, x, x_tau, tau, ExitIndex= inputs
        num_sample = np.shape(dw)[0]
        #all_zero_vec = tf.zeros(shape=tf.stack([tf.shape(dw)[0], 1]), dtype=self.net_config.dtype)
        #y = all_zero_vec * self.y_init
        y = np.zeros([num_sample])
        for j in range(num_sample):
            print("actor index",j)
            for t in range(self.bsde.num_time_interval):
            #for j in range(num_sample):
                print("actor step",t)
                if t < ExitIndex[j]:
                    # maybe reshape is needed to call NN_control
                    print("actor in")
                    x_jt = tf.reshape(x[j,:,t], [1,self.bsde.dim])
                    y[j] = y[j] + self.bsde.w_tf(x_jt, self.NN_control(x_jt, training)) * self.bsde.delta_t
                elif t == ExitIndex[j]:
                    print("actor out")
                    x_jt = tf.reshape(x[j,:,t], [1,self.bsde.dim])
                    y[j] = y[j] + self.bsde.w_tf(x_jt, self.NN_control(x_jt, training)) * (tau[j] - t*self.bsde.delta_t)
            # currently use boundary condition instead of NN if X_tau reaches boundary
            if ExitIndex[j] == self.bsde.num_time_interval:
                # the size of Z_tf is currently num_sample, not num_sample * 1
                print("actor boundary")
                y[j] = y[j] + self.bsde.Z_tf(tf.reshape(x_tau[j,:], [1, self.eqn_config.dim]))
            else:
                print("actor inside")
                y[j] = y[j] + model_critic.NN_value(tf.reshape(x_tau[j,:], [1, self.eqn_config.dim]), training)
        print("actor model computed")
        return y


class DeepNN(tf.keras.Model):
    def __init__(self, config, AC):
        super(DeepNN, self).__init__()
        print("define NN")
        #dim = config.eqn_config.dim
        if AC == "critic":
            num_hiddens = config.net_config.num_hiddens_critic
        elif AC == "actor":
            num_hiddens = config.net_config.num_hiddens_actor
        self.bn_layers = [
            tf.keras.layers.BatchNormalization(
                momentum=0.99,
                epsilon=1e-6,
                beta_initializer=tf.random_normal_initializer(0.0, stddev=0.1),
                gamma_initializer=tf.random_uniform_initializer(0.1, 0.5)
            )
            for _ in range(len(num_hiddens) + 2)]
        self.dense_layers = [tf.keras.layers.Dense(num_hiddens[i],
                                                   use_bias=False,
                                                   activation=None)
                             for i in range(len(num_hiddens))]
        # final output should be gradient of size dim
        self.dense_layers.append(tf.keras.layers.Dense(1, activation=None))

    def call(self, x, training):
        """structure: bn -> (dense -> bn -> relu) * len(num_hiddens) -> dense -> bn"""
        print("call NN")
        x = self.bn_layers[0](x, training)
        for i in range(len(self.dense_layers) - 1):
            x = self.dense_layers[i](x)
            x = self.bn_layers[i+1](x, training)
            x = tf.nn.relu(x)
        x = self.dense_layers[-1](x)
        x = self.bn_layers[-1](x, training)
        return x
