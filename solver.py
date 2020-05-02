import logging
import time
import numpy as np
import tensorflow as tf

DELTA_CLIP = 50.0


class ActorCriticSolver(object):
    """The fully connected neural network model."""
    def __init__(self, config, bsde):
        self.eqn_config = config.eqn_config
        self.net_config = config.net_config
        self.bsde = bsde
        self.model_critic = CriticModel(config, bsde)
        self.model_actor = ActorModel(config, bsde)
        lr_schedule_critic = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            self.net_config.lr_boundaries_critic, self.net_config.lr_values_critic)
        lr_schedule_actor = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            self.net_config.lr_boundaries_actor, self.net_config.lr_values_actor)
        self.optimizer_critic = tf.keras.optimizers.Adam(learning_rate=lr_schedule_critic, epsilon=1e-8)
        self.optimizer_actor = tf.keras.optimizers.Adam(learning_rate=lr_schedule_actor, epsilon=1e-8)
        self.x = None
        
    def train(self):
        start_time = time.time()
        training_history = []
        valid_data = self.bsde.sample(self.net_config.valid_size, control_fcn=self.control_fcn)
        # begin sgd iteration
        for step in range(self.net_config.num_iterations+1):
            if step == self.net_config.num_iterations:
                dw, x, coef, x_bdry = valid_data
                y = self.model_critic.NN_value(x[:,:,0], training=False, need_grad=False)
                z = self.model_actor.NN_control(x[:,:,0], training=False, need_grad=False)
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
            #self.train_step_critic(self.bsde.sample(self.net_config.batch_size, control_fcn=self.control_fcn))
            self.train_step_actor(self.bsde.sample(self.net_config.batch_size, control_fcn=self.control_fcn))
        return np.array(training_history), x[:,:,0], y.numpy(), z.numpy()

    def loss_critic(self, inputs, training):
        # this delta is already squared
        delta, delta_bdry = self.model_critic(inputs, self.model_actor, training)
        # use linear approximation outside the clipped range
        loss1 = tf.reduce_mean(tf.where(tf.abs(delta) < DELTA_CLIP, tf.square(delta),
                                       2 * DELTA_CLIP * tf.abs(delta) - DELTA_CLIP ** 2))
        loss2 = tf.reduce_mean(tf.where(tf.abs(delta_bdry) < DELTA_CLIP, tf.square(delta_bdry),
                                       2 * DELTA_CLIP * tf.abs(delta_bdry) - DELTA_CLIP ** 2))
        return loss1 + loss2
    
    def loss_actor(self, inputs, training):
        y = self.model_actor(inputs, self.model_critic, training)
        loss = tf.reduce_mean(y)
        return loss

    def grad_critic(self, inputs, training):
        with tf.GradientTape(persistent=True) as tape:
            loss_critic = self.loss_critic(inputs, training)
        grad = tape.gradient(loss_critic, self.model_critic.trainable_variables)
        del tape
        return grad
    
    def grad_actor(self, inputs, training):
        with tf.GradientTape(persistent=True) as tape:
            loss_actor = self.loss_actor(inputs, training)
        grad = tape.gradient(loss_actor, self.model_actor.trainable_variables)
        del tape
        return grad

    @tf.function
    def train_step_critic(self, train_data):
        grad = self.grad_critic(train_data, training=True)
        #print("train critic", self.model_critic.trainable_variables)
        self.optimizer_critic.apply_gradients(zip(grad, self.model_critic.trainable_variables))
        
    @tf.function
    def train_step_actor(self, train_data):
        grad = self.grad_actor(train_data, training=True)
        #print("train actor", self.model_actor.trainable_variables)
        self.optimizer_actor.apply_gradients(zip(grad, self.model_actor.trainable_variables))
        
    def control_fcn(self, x):
        #return self.bsde.u_true(x)
        return self.bsde.u_true(x) * self.model_actor.lmbd.numpy()
        #return self.model_actor.NN_control(x, training=True, need_grad=False).numpy()
        
    def err_value(self, inputs):
        dw, x, coef, x_bdry= inputs
        x0 = x[:,:,0]
        error_value = tf.reduce_sum(tf.square(self.bsde.V_true(x0) - self.model_critic.NN_value(x0, training=False, need_grad=False)))
        norm = tf.reduce_sum(tf.square(self.bsde.V_true(x0)))
        return tf.sqrt(error_value / norm)
    
    def err_control(self, inputs):
        #dw, x, coef, x_bdry = inputs
        #x0 = x[:,:,0]
        #error_control = tf.reduce_sum(tf.square(self.bsde.u_true(x0) - self.model_actor.NN_control(x0, training=False, need_grad=False)))
        #norm = tf.reduce_sum(tf.square(self.bsde.u_true(x0)))
        #return tf.sqrt(error_control / norm)
        return 1 - self.model_actor.lmbd
        
class CriticModel(tf.keras.Model):
    def __init__(self, config, bsde):
        super(CriticModel, self).__init__()
        self.eqn_config = config.eqn_config
        self.net_config = config.net_config
        self.bsde = bsde
        self.NN_value = DeepNN(config, "critic")
        self.NN_value_grad = DeepNN(config, "actor")
        
    def call(self, inputs, model_actor, training):
        dw, x, coef, x_bdry = inputs
        num_sample = np.shape(dw)[0]
        y = 0
        for t in range(self.bsde.num_time_interval):
            #y = y + tf.reshape(coef[:,t], [num_sample,1]) * self.bsde.w_tf(x[:,:,t], self.bsde.u_true(x[:,:,t])) * self.bsde.delta_t
            y = y + tf.reshape(coef[:,t], [num_sample,1]) * (self.bsde.w_tf(x[:,:,t], self.bsde.u_true(x[:,:,t])) * self.bsde.delta_t
                 - self.bsde.sigma * 2 * self.bsde.sqrtpqoverbeta * tf.reduce_sum(x[:,:,t] * dw[:,:,t],1,keepdims=True))
            #_, grad = self.NN_value(x[:,:,t], training, need_grad=True)
            #y = y + tf.reshape(coef[:,t], [num_sample,1]) * (self.bsde.w_tf(x[:,:,t], self.bsde.u_true(x[:,:,t])) * self.bsde.delta_t
            #    - self.bsde.sigma * tf.reduce_sum(grad * dw[:,:,t], 1, keepdims=True))
            #y = y + tf.reshape(coef[:,t], [num_sample,1]) * (self.bsde.w_tf(x[:,:,t], self.bsde.u_true(x[:,:,t])) * self.bsde.delta_t
            #    - self.bsde.sigma * tf.reduce_sum(self.NN_value_grad(x[:,:,t], training, need_grad=False) * dw[:,:,t], 1, keepdims=True))
            y = y + tf.reshape(coef[:,t], [num_sample,1]) * (self.bsde.w_tf(x[:,:,t], model_actor.NN_control(x[:,:,t], training, need_grad=False)) * self.bsde.delta_t
                - self.bsde.sigma * 2 * self.bsde.sqrtpqoverbeta * tf.reduce_sum(x[:,:,t] * dw[:,:,t],1,keepdims=True))
            #print("model critic", y)
        delta = self.NN_value(x[:,:,0], training, need_grad=False) - y - self.NN_value(x[:,:,-1], training, need_grad=False)
        #delta = self.bsde.V_true(x[:,:,0]) - y - self.bsde.V_true(x[:,:,-1])
        #print("critic delta", delta)
        delta_bdry = self.NN_value(x_bdry, training, need_grad=False) - self.bsde.Z_tf(x_bdry)
        #print("critic bdry delta", delta_bdry)
        return delta, delta_bdry

class ActorModel(tf.keras.Model):
    def __init__(self, config, bsde):
        super(ActorModel, self).__init__()
        self.eqn_config = config.eqn_config
        self.net_config = config.net_config
        self.bsde = bsde
        self.NN_control = DeepNN(config, "actor")
        self.lmbd = tf.Variable(initial_value=0.5, trainable=True, dtype='float64')

    def call(self, inputs, model_critic, training):
        dw, x, coef, x_bdry = inputs
        num_sample = np.shape(dw)[0]
        y = 0
        for t in range(self.bsde.num_time_interval):
            #y = y + tf.reshape(coef[:,t], [num_sample,1]) * self.bsde.w_tf(x[:,:,t], self.bsde.u_true(x[:,:,t])) * self.bsde.delta_t
            y = y + tf.reshape(coef[:,t], [num_sample,1]) * self.bsde.w_tf(x[:,:,t], self.lmbd * self.bsde.u_true(x[:,:,t])) * self.bsde.delta_t
            #y = y + tf.reshape(coef[:,t], [num_sample,1]) * self.bsde.w_tf(x[:,:,t], self.NN_control(x[:,:,t], training, need_grad=False)) * self.bsde.delta_t
            #print("model actor", y)
        #y = y + model_critic.NN_value(x[:,:,-1], training, need_grad=False)
        y = y + self.bsde.V_true(x[:,:,-1])
        #print("plus value",y)
        return y


class DeepNN(tf.keras.Model):
    def __init__(self, config, AC):
        super(DeepNN, self).__init__()
        self.AC = AC
        dim = config.eqn_config.dim
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
        if AC == "critic":
            self.dense_layers.append(tf.keras.layers.Dense(1, activation=None))
        elif AC == "actor":
            self.dense_layers.append(tf.keras.layers.Dense(dim, activation=None))

    def call(self, x, training, need_grad):
        """structure: bn -> (dense -> bn -> relu) * len(num_hiddens) -> dense -> bn"""
        with tf.GradientTape() as g:
            if self.AC == "actor" and need_grad:
                g.watch(x)
            y = self.bn_layers[0](x, training)
            for i in range(len(self.dense_layers) - 1):
                y = self.dense_layers[i](y)
                y = self.bn_layers[i+1](y, training)
                y = tf.nn.relu(y)
            y = self.dense_layers[-1](y)
            y = self.bn_layers[-1](y, training)
        if self.AC == "actor" and need_grad:
            return y, g.gradient(y, x)
        else:
            return y