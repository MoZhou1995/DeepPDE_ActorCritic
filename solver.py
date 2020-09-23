import logging
import time
import numpy as np
import tensorflow as tf
DELTA_CLIP = 50.0


class ActorCriticSolver(object):
    """The fully connected neural network model."""
    #def __init__(self, config, bsde):
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
        self.gamma = self.eqn_config.discount
        
    def train(self):
        start_time = time.time()
        training_history = []
        valid_data_critic = self.bsde.sample2_tf(self.net_config.valid_size, self.eqn_config.total_time_critic, self.eqn_config.num_time_interval_critic)
        valid_data_actor = self.bsde.sample2_tf(self.net_config.valid_size, self.eqn_config.total_time_actor, self.eqn_config.num_time_interval_actor)
        valid_data_cost = self.bsde.sample0_tf(self.net_config.valid_size, self.eqn_config.total_time_actor, self.eqn_config.num_time_interval_actor)
        true_loss_actor = self.loss_actor(valid_data_actor, training=False, cheat_value=True, cheat_control=True).numpy()
        # begin sgd iteration
        for step in range(self.net_config.num_iterations+1):
            if step == self.net_config.num_iterations:
                x0, dw_sample, x_bdry = valid_data_critic
                y = self.model_critic.NN_value(x0, training=False, need_grad=False)
                true_y = self.bsde.V_true(x0)
                z = self.model_actor.NN_control(x0, training=False, need_grad=False)
                true_z = self.bsde.u_true(x0)
            if step % self.net_config.logging_frequency == 0:
                loss_critic = self.loss_critic(valid_data_critic, training=False).numpy()
                loss_actor = self.loss_actor(valid_data_actor, training=False, cheat_value=False, cheat_control=False).numpy()
                err_value = self.err_value(valid_data_critic).numpy()
                err_control = self.err_control(valid_data_actor).numpy()
                error_cost = self.error_cost(valid_data_cost).numpy()
                error_cost2 = self.error_cost2(valid_data_cost).numpy()
                elapsed_time = time.time() - start_time
                training_history.append([step, loss_critic, loss_actor, true_loss_actor, err_value, err_control, error_cost, error_cost2, elapsed_time])
                if self.net_config.verbose:
                    logging.info("step: %5u, loss_critic: %.4e, loss_actor: %.4e, true_loss_actor: %.4e, err_value: %.4e, err_control: %.4e, err_cost: %.4e, err_cost2: %.4e, elapsed time: %3u" % (
                        step, loss_critic, loss_actor, true_loss_actor, err_value, err_control, error_cost, error_cost2, elapsed_time))
            self.train_step_critic(self.bsde.sample2_tf(self.net_config.batch_size, self.eqn_config.total_time_critic, self.eqn_config.num_time_interval_critic))
            self.train_step_actor(self.bsde.sample2_tf(self.net_config.batch_size, self.eqn_config.total_time_actor, self.eqn_config.num_time_interval_actor))
        return np.array(training_history), x0, y, true_y, z, true_z

    def loss_critic(self, inputs, training):
        # this delta is already squared
        delta, delta_bdry = self.model_critic(inputs, self.model_actor, training)
        # use linear approximation outside the clipped range
        loss1 = tf.reduce_mean(tf.where(tf.abs(delta) < DELTA_CLIP, tf.square(delta), 2 * DELTA_CLIP * tf.abs(delta) - DELTA_CLIP ** 2))
        loss2 = tf.reduce_mean(tf.where(tf.abs(delta_bdry) < DELTA_CLIP, tf.square(delta_bdry), 2 * DELTA_CLIP * tf.abs(delta_bdry) - DELTA_CLIP ** 2))
        return (loss1 + loss2) * 100
    
    def loss_actor(self, inputs, training, cheat_value, cheat_control):
        y = self.model_actor(inputs, self.model_critic, training, cheat_value, cheat_control)
        loss = tf.reduce_mean(y)
        return loss

    def grad_critic(self, inputs, training):
        with tf.GradientTape(persistent=True) as tape:
            loss_critic = self.loss_critic(inputs, training)
        grad = tape.gradient(loss_critic, self.model_critic.trainable_variables)
        del tape
        return grad
    
    def grad_actor(self, inputs, training, cheat_value, cheat_control):
        with tf.GradientTape(persistent=True) as tape:
            loss_actor = self.loss_actor(inputs, training, cheat_value, cheat_control)
        grad = tape.gradient(loss_actor, self.model_actor.trainable_variables)
        del tape
        return grad

    @tf.function
    def train_step_critic(self, train_data):
        grad = self.grad_critic(train_data, training=False)
        self.optimizer_critic.apply_gradients(zip(grad, self.model_critic.trainable_variables))
        
    @tf.function
    def train_step_actor(self, train_data):
        grad = self.grad_actor(train_data, training=False, cheat_value=False, cheat_control=False)
        self.optimizer_actor.apply_gradients(zip(grad, self.model_actor.trainable_variables))
        
    def err_value(self, inputs):
        x0, _, _ = inputs
        error_value = tf.reduce_sum(tf.square(self.bsde.V_true(x0) - self.model_critic.NN_value(x0, training=False, need_grad=False)))
        norm = tf.reduce_sum(tf.square(self.bsde.V_true(x0)))
        return tf.sqrt(error_value / norm)
    
    def err_control(self, inputs):
        x0, _, _ = inputs
        error_control = tf.reduce_sum(tf.square(self.bsde.u_true(x0) - self.model_actor.NN_control(x0, training=False, need_grad=False)))
        norm = tf.reduce_sum(tf.square(self.bsde.u_true(x0)))
        return tf.sqrt(error_control / norm)
        #return self.model_actor.lmbd
        
    def error_cost(self, inputs):
        x0, _, _ = inputs
        y = self.model_actor(inputs, self.model_critic, training=False, cheat_value=True, cheat_control=False)
        y_true = self.bsde.V_true(x0)
        return tf.reduce_mean(y-y_true)
    
    def error_cost2(self, inputs):
        x0, _, _ = inputs
        y = self.model_actor(inputs, self.model_critic, training=False, cheat_value=False, cheat_control=False)
        y0 = self.model_critic.NN_value(x0, training=False, need_grad=False)
        return tf.reduce_mean(y-y0)
        
class CriticModel(tf.keras.Model):
    def __init__(self, config, bsde):
        super(CriticModel, self).__init__()
        self.eqn_config = config.eqn_config
        self.net_config = config.net_config
        self.bsde = bsde
        self.NN_value = DeepNN(config, "critic")
        self.NN_value_grad = DeepNN(config, "critic_grad")
        self.gamma = config.eqn_config.discount
        
    def call(self, inputs, model_actor, training):
        x0, dw, x_bdry = inputs
        num_sample = np.shape(dw)[0]
        delta_t = self.eqn_config.total_time_critic / self.eqn_config.num_time_interval_critic
        sqrt_delta_t = np.sqrt(delta_t)
        y = 0
        #别忘了关注propagate12用的是true control还是NN_control
        # x, coef = self.bsde.propagate2_tf(num_sample, x0, dw, model_actor.NN_control, training, self.eqn_config.total_time_critic, self.eqn_config.num_time_interval_critic)
        x, dt, coef = self.bsde.propagate2_tf(num_sample, x0, dw, model_actor.NN_control, training, self.eqn_config.total_time_critic, self.eqn_config.num_time_interval_critic, cheat=False)
        for t in range(self.eqn_config.num_time_interval_critic):
            # old sample with true gradient of V
            # y = y + coef[:,t:t+1] * ((self.bsde.w_tf(x[:,:,t], self.bsde.u_true(x[:,:,t])) - self.gamma * self.NN_value(x[:,:,t], training, need_grad=False)) * delta_t -self.bsde.sigma * 2 * tf.reduce_sum(np.array([-1,1]) * x[:,:,t] * dw[:,:,t],1,keepdims=True) / self.bsde.lmbd / ((x[:,0:1,t])**2 - (x[:,1:2,t])**2 + self.bsde.R**2 +1))
            # old sample with another NN
            # y = y + coef[:,t:t+1] * ((self.bsde.w_tf(x[:,:,t], self.bsde.u_true(x[:,:,t])) - self.gamma * self.NN_value(x[:,:,t], training, need_grad=False)) * delta_t
            #     - self.bsde.sigma * tf.reduce_sum(self.NN_value_grad(x[:,:,t], training, need_grad=False) * dw[:,:,t], 1, keepdims=True))
            # old sample with gradient of NN_value
            # _, grad = self.NN_value(x[:,:,t], training, need_grad=True)
            # y = y + coef[:,t:t+1] * ((self.bsde.w_tf(x[:,:,t], self.bsde.u_true(x[:,:,t])) - self.gamma * self.NN_value(x[:,:,t], training, need_grad=False)) * delta_t
            #     - self.bsde.sigma * tf.reduce_sum(grad * dw[:,:,t], 1, keepdims=True))
            # new sample use no gradient, does not work at all
            # y = y + coef[:,t:t+1] * (self.bsde.w_tf(x[:,:,t], self.bsde.u_true(x[:,:,t])) - self.gamma * self.NN_value(x[:,:,t], training, need_grad=False)) * dt[:,t:t+1]
            # new sample use the gradient of NN_value as gradient V
            # _, grad = self.NN_value(x[:,:,t], training, need_grad=True)
            # y = y + coef[:,t:t+1] * ((self.bsde.w_tf(x[:,:,t], self.bsde.u_true(x[:,:,t])) - self.gamma * self.NN_value(x[:,:,t], training, need_grad=False)) * dt[:,t:t+1] - self.bsde.sigma * tf.reduce_sum(grad * dw[:,:,t], 1, keepdims=True) * tf.sqrt(dt[:,t:t+1]) / sqrt_delta_t )
            # new sample use another NN to represent the gradient of value function, with true cuntrol
            # y = y + coef[:,t:t+1] * ((self.bsde.w_tf(x[:,:,t], self.bsde.u_true(x[:,:,t])) - self.gamma * self.NN_value(x[:,:,t], training, need_grad=False)) * dt[:,t:t+1] - self.bsde.sigma * tf.reduce_sum(self.NN_value_grad(x[:,:,t], training, need_grad=False) * dw[:,:,t], 1, keepdims=True)* tf.sqrt(dt[:,t:t+1]) / sqrt_delta_t)
            # new sample given the true gradient of value function and true control
            #y = y + coef[:,t:t+1] * ((self.bsde.w_tf(x[:,:,t], self.bsde.u_true(x[:,:,t])) - self.gamma * self.NN_value(x[:,:,t], training, need_grad=False)) * dt[:,t:t+1] - self.bsde.sigma * 2 * self.bsde.sqrtpqoverbeta * tf.reduce_sum(x[:,:,t] * dw[:,:,t],1,keepdims=True) * tf.sqrt(dt[:,t:t+1]) / sqrt_delta_t )
            # y = y + coef[:,t:t+1] * ((self.bsde.w_tf(x[:,:,t], self.bsde.u_true(x[:,:,t])) - self.gamma * self.NN_value(x[:,:,t], training, need_grad=False)) * dt[:,t:t+1] - self.bsde.sigma * 2 * tf.reduce_sum(np.array([-1,1]) * x[:,:,t] * dw[:,:,t],1,keepdims=True) / self.bsde.lmbd / ((x[:,0:1,t])**2 - (x[:,1:2,t])**2 + self.bsde.R**2 +1) * tf.sqrt(dt[:,t:t+1]) / sqrt_delta_t)
            # ADMM given the true gradient of value function
            #y = y + coef[:,t:t+1] * ((self.bsde.w_tf(x[:,:,t], model_actor.NN_control(x[:,:,t], training, need_grad=False)) - self.gamma * self.NN_value(x[:,:,t], training, need_grad=False)) * dt[:,t:t+1] - self.bsde.sigma * 2 * self.bsde.sqrtpqoverbeta * tf.reduce_sum(x[:,:,t] * dw[:,:,t],1,keepdims=True) * tf.sqrt(dt[:,t:t+1]) / sqrt_delta_t )
            # ADMM use another NN to represent the gradient of value function
            y = y + coef[:,t:t+1] * ((self.bsde.w_tf(x[:,:,t], model_actor.NN_control(x[:,:,t], training, need_grad=False)) - self.gamma * self.NN_value(x[:,:,t], training, need_grad=False)) * dt[:,t:t+1] - self.bsde.sigma * tf.reduce_sum(self.NN_value_grad(x[:,:,t], training, need_grad=False) * dw[:,:,t], 1, keepdims=True)* tf.sqrt(dt[:,t:t+1]) / sqrt_delta_t)
            # ADMM use the gradient of NN_value as gradient V
            # _, grad = self.NN_value(x[:,:,t], training, need_grad=True)
            # y = y + coef[:,t:t+1] * ((self.bsde.w_tf(x[:,:,t], model_actor.NN_control(x[:,:,t], training, need_grad=False)) - self.gamma * self.NN_value(x[:,:,t], training, need_grad=False)) * dt[:,t:t+1] - self.bsde.sigma * tf.reduce_sum(grad * dw[:,:,t], 1, keepdims=True) * tf.sqrt(dt[:,t:t+1]) / sqrt_delta_t )
        delta = self.NN_value(x[:,:,0], training, need_grad=False) - y - self.NN_value(x[:,:,-1], training, need_grad=False)
        #delta = self.bsde.V_true(x[:,:,0]) - y - self.bsde.V_true(x[:,:,-1])
        delta_bdry = self.NN_value(x_bdry, training, need_grad=False) - self.bsde.Z_tf(x_bdry)
        return delta, delta_bdry

class ActorModel(tf.keras.Model):
    #def __init__(self, config, bsde):
    def __init__(self, config, bsde):
        super(ActorModel, self).__init__()
        self.eqn_config = config.eqn_config
        self.net_config = config.net_config
        self.bsde = bsde
        self.NN_control = DeepNN(config, "actor")
        self.gamma = config.eqn_config.discount
        
    def call(self, inputs, model_critic, training, cheat_value, cheat_control):
        x0, dw, x_bdry = inputs
        num_sample = np.shape(dw)[0]
        # delta_t = self.eqn_config.total_time_actor / self.eqn_config.num_time_interval_actor
        y = 0
        # x, coef = self.bsde.propagate_tf(num_sample, x0, dw, self.NN_control, training, self.eqn_config.total_time_actor, self.eqn_config.num_time_interval_actor)
        x, dt, coef = self.bsde.propagate2_tf(num_sample, x0, dw, self.NN_control, training, self.eqn_config.total_time_actor, self.eqn_config.num_time_interval_actor, cheat_control)
        discount = 1 #broadcast to num_sample x 1
        if cheat_control == False:
            for t in range(self.eqn_config.num_time_interval_actor):
                #y = y + coef[:,t:t+1] * self.bsde.w_tf(x[:,:,t], self.bsde.u_true(x[:,:,t])) * delta_t * discount
                #y = y + coef[:,t:t+1] * self.bsde.w_tf(x[:,:,t], self.lmbd * self.bsde.u_true(x[:,:,t])) * delta_t * discount
                # y = y + coef[:,t:t+1] * self.bsde.w_tf(x[:,:,t], self.NN_control(x[:,:,t], training, need_grad=False)) * delta_t * discount
                y = y + coef[:,t:t+1] * self.bsde.w_tf(x[:,:,t], self.NN_control(x[:,:,t], training, need_grad=False)) * dt[:,t:t+1] * discount
                discount *= tf.math.exp(-self.gamma * dt[:,t:t+1] * coef[:,t:t+1])
        else:
            for t in range(self.eqn_config.num_time_interval_actor):
                y = y + coef[:,t:t+1] * self.bsde.w_tf(x[:,:,t], self.bsde.u_true(x[:,:,t])) * dt[:,t:t+1] * discount
                discount *= tf.math.exp(-self.gamma * dt[:,t:t+1] * coef[:,t:t+1])
        if cheat_value == False:
            y = y + model_critic.NN_value(x[:,:,-1], training, need_grad=False) * discount
        else:
            y = y + self.bsde.V_true(x[:,:,-1]) * discount
        return y


class DeepNN(tf.keras.Model):
    def __init__(self, config, AC):
        super(DeepNN, self).__init__()
        self.AC = AC
        dim = config.eqn_config.dim
        if AC == "actor":
            num_hiddens = config.net_config.num_hiddens_actor
        else: #AC == "critic" or "critic_grad"
            num_hiddens = config.net_config.num_hiddens_critic
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
        elif AC == "critic_grad":
            self.dense_layers.append(tf.keras.layers.Dense(dim, activation=None))
        elif AC == "actor":
            d = config.eqn_config.control_dim
            self.dense_layers.append(tf.keras.layers.Dense(d, activation=None))

    def call(self, x, training, need_grad):
        """structure: bn -> (dense -> bn -> relu) * len(num_hiddens) -> dense -> bn"""
        with tf.GradientTape() as g:
            if self.AC == "critic" and need_grad:
                g.watch(x)
            y = self.bn_layers[0](x, training)
            for i in range(len(self.dense_layers) - 1):
                y = self.dense_layers[i](y)
                y = self.bn_layers[i+1](y, training)
                y = tf.nn.relu(y)
            y = self.dense_layers[-1](y)
            y = self.bn_layers[-1](y, training)
        if self.AC == "critic" and need_grad:
            return y, g.gradient(y, x)
        else:
            return y