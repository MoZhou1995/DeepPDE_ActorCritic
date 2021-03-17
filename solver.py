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
        self.train_config = config.train_config
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
        if self.train_config.sample_type == "normal":
            self.sample = self.bsde.sample_normal
        if self.train_config.sample_type == "bounded":
            self.sample = self.bsde.sample_bounded
        if self.train_config.train == "actor-critic":
            self.cheat_value_in_actor = False
            self.cheat_control_in_critic = False
        elif self.train_config.train == "critic":
            self.cheat_control_in_critic = True
        elif self.train_config.train == "actor":
            self.cheat_value_in_actor = True
         
    def train(self):
        start_time = time.time()
        training_history = []
        valid_data_critic = self.sample(self.net_config.valid_size, self.eqn_config.num_time_interval_critic)
        valid_data_actor = self.sample(self.net_config.valid_size, self.eqn_config.num_time_interval_actor)
        valid_data_cost = self.bsde.sample0(self.net_config.valid_size, self.eqn_config.num_time_interval_actor)
        true_loss_actor = self.loss_actor(valid_data_actor, training=False, cheat_value=True, cheat_control=True).numpy()
        # begin sgd iteration
        for step in range(self.net_config.num_iterations+1):
            if step % self.net_config.logging_frequency == 0:
                loss_critic = self.loss_critic(valid_data_critic, training=False, cheat_control=False).numpy()
                loss_actor = self.loss_actor(valid_data_actor, training=False, cheat_value=False, cheat_control=False).numpy()
                err_value = self.err_value(valid_data_critic).numpy()
                err_control = self.err_control(valid_data_actor).numpy()
                err_value_grad = self.err_value_grad(valid_data_critic).numpy()
                err_value_infty = self.err_value_infty(valid_data_critic).numpy()
                err_cost = self.err_cost(valid_data_cost).numpy()
                elapsed_time = time.time() - start_time
                training_history.append(
                    [
                        step, loss_critic, loss_actor, err_value, err_value_infty,
                        err_control, err_value_grad, err_cost, elapsed_time
                    ]
                )
                if self.net_config.verbose:
                    logging.info(
                        "step: %5u, loss_critic: %.4e, loss_actor: %.4e, " % (step, loss_critic, loss_actor) + \
                        "err_value: %.4e, err_value_infty: %.4e, err_control: %.4e, " % (err_value, err_value_infty, err_control) + \
                        "err_value_grad: %.4e, err_cost: %.4e, elapsed time: %3u" % (err_value_grad, err_cost, elapsed_time)
                    )
            if step == self.net_config.num_iterations:
                x0, dw_sample, x_bdry = valid_data_critic
                y = self.model_critic.NN_value(x0, training=False, need_grad=False)
                true_y = self.bsde.V_true(x0)
                grad_y = self.model_critic.NN_value_grad(x0, training=False, need_grad=False)
                z = self.model_actor.NN_control(x0, training=False, need_grad=False)
                true_z = self.bsde.u_true(x0)
                print("true loss actor: ", true_loss_actor)
                training_history.append([0, 0.0, true_loss_actor, 0.0, 0.0, 0.0, 0.0, 0.0, elapsed_time])
            if self.train_config.train == "actor-critic" or self.train_config.train == "critic":
                self.train_step_critic(
                    self.sample(self.net_config.batch_size, self.eqn_config.num_time_interval_critic)
                )
            if self.train_config.train == "actor-critic" or self.train_config.train == "actor":
                self.train_step_actor(
                    self.sample(self.net_config.batch_size, self.eqn_config.num_time_interval_actor)
                )
        return np.array(training_history), x0, y, true_y, z, true_z, grad_y

    def loss_critic(self, inputs, training, cheat_control):
        delta, delta_bdry = self.model_critic(inputs, self.model_actor, training, cheat_control)
        # use linear approximation outside the clipped range
        loss = tf.reduce_mean(
            tf.where(
                tf.abs(delta) < DELTA_CLIP,
                tf.square(delta),
                2 * DELTA_CLIP * tf.abs(delta) - DELTA_CLIP ** 2
            )
        )
        loss_bdry = tf.reduce_mean(
            tf.where(
                tf.abs(delta_bdry) < DELTA_CLIP,
                tf.square(delta_bdry),
                2 * DELTA_CLIP * tf.abs(delta_bdry) - DELTA_CLIP ** 2
            )
        )
        return (loss + loss_bdry) * 100
        
    def loss_actor(self, inputs, training, cheat_value, cheat_control):
        y = self.model_actor(inputs, self.model_critic, training, cheat_value, cheat_control)
        loss = tf.reduce_mean(y)
        return loss

    def grad_critic(self, inputs, training, cheat_control):
        with tf.GradientTape(persistent=True) as tape:
            loss_critic = self.loss_critic(inputs, training, cheat_control)
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
        grad = self.grad_critic(train_data, training=False, cheat_control=self.cheat_control_in_critic)
        self.optimizer_critic.apply_gradients(zip(grad, self.model_critic.trainable_variables))
        
    @tf.function
    def train_step_actor(self, train_data):
        grad = self.grad_actor(train_data, training=False, cheat_value=self.cheat_value_in_actor, cheat_control=False)
        self.optimizer_actor.apply_gradients(zip(grad, self.model_actor.trainable_variables))
        
    def err_value(self, inputs):
        x0, _, _ = inputs
        err_value = tf.reduce_sum(
            tf.square(self.bsde.V_true(x0) - self.model_critic.NN_value(x0, training=False, need_grad=False))
        )
        norm = tf.reduce_sum(tf.square(self.bsde.V_true(x0)))
        return tf.sqrt(err_value / norm)
    
    def err_control(self, inputs):
        x0, _, _ = inputs
        err_control = tf.reduce_sum(
            tf.square(self.bsde.u_true(x0) - self.model_actor.NN_control(x0, training=False, need_grad=False))
        )
        norm = tf.reduce_sum(tf.square(self.bsde.u_true(x0)))
        return tf.sqrt(err_control / norm)
        
    def err_value_grad(self, inputs):
        x0, _, _ = inputs
        err_value_grad = tf.reduce_sum(
            tf.square(self.bsde.V_grad_true(x0) - self.model_critic.NN_value_grad(x0, training=False, need_grad=False))
        )
        norm = tf.reduce_sum(tf.square(self.bsde.V_grad_true(x0)))
        return tf.sqrt(err_value_grad / norm)
    
    def err_value_infty(self, inputs):
        x0, _, _ = inputs
        err_value = self.bsde.V_true(x0) - self.model_critic.NN_value(x0, training=False, need_grad=False)
        return tf.reduce_max(tf.abs(err_value))
    
    def err_cost(self, inputs):
        x0, _, _ = inputs
        y = self.model_actor(inputs, self.model_critic, training=False, cheat_value=False, cheat_control=False)
        y0 = self.model_critic.NN_value(x0, training=False, need_grad=False)
        return tf.reduce_mean(y-y0)
        
class CriticModel(tf.keras.Model):
    def __init__(self, config, bsde):
        super(CriticModel, self).__init__()
        self.eqn_config = config.eqn_config
        self.net_config = config.net_config
        self.train_config = config.train_config
        self.bsde = bsde
        self.NN_value = DeepNN(config, "critic")
        self.NN_value_grad = DeepNN(config, "critic_grad")
        self.gamma = config.eqn_config.discount
        if self.train_config.scheme == "naive":
            self.propagate = self.bsde.propagate_naive
        elif self.train_config.scheme == "adaptive":
            self.propagate = self.bsde.propagate_adaptive
        
    def control(self, x, cheat_control, model_actor):
        if cheat_control == False:
            return model_actor.NN_control(x, training=False, need_grad=False)
        else:
            return self.bsde.u_true(x)
    
    def call(self, inputs, model_actor, training, cheat_control):
        x0, dw, x_bdry = inputs
        num_sample = np.shape(dw)[0]
        delta_t = self.eqn_config.total_time_critic / self.eqn_config.num_time_interval_critic
        y = 0
        discount = 1 #broadcast to num_sample x 1
        x, dt, coef = self.propagate(
            num_sample, x0, dw, model_actor.NN_control, training,
            self.eqn_config.total_time_critic,
            self.eqn_config.num_time_interval_critic, cheat_control
        )
        for t in range(self.eqn_config.num_time_interval_critic):
            u = self.control(x[:,:,t], cheat_control, model_actor) #this is the control we use
            w = self.bsde.w_tf(x[:,:,t],u) # running cost
            #integrand for drift
            delta_y_drift = w * discount
            #coef for drift
            delta_y_drift_coef = coef[:,t:t+1] * dt[:,t:t+1]
            # update the drift
            y += delta_y_drift * delta_y_drift_coef
            
            # For TD1 we need to consider diffusion
            if self.train_config.TD_type == "TD1":
                #integrand for diffusion
                delta_y_diffusion = self.bsde.sigma * tf.reduce_sum(
                    self.NN_value_grad(x[:,:,t], training, need_grad=False) * dw[:,:,t], 1, keepdims=True
                )
                delta_y_diffusion *= discount 
                #coef for diffusion
                delta_y_diffusion_coef = coef[:,t:t+1] * tf.sqrt(dt[:,t:t+1])
                y -= delta_y_diffusion * delta_y_diffusion_coef
            
            # we need to update the discount
            discount *= tf.math.exp(-self.gamma * dt[:,t:t+1] * coef[:,t:t+1])
            
        delta = self.NN_value(x[:,:,0], training, need_grad=False) - y \
             - self.NN_value(x[:,:,-1], training, need_grad=False) * discount
        delta_bdry = self.NN_value(x_bdry, training, need_grad=False) - self.bsde.Z_tf(x_bdry)
        return delta, delta_bdry

class ActorModel(tf.keras.Model):
    def __init__(self, config, bsde):
        super(ActorModel, self).__init__()
        self.eqn_config = config.eqn_config
        self.net_config = config.net_config
        self.train_config = config.train_config
        self.bsde = bsde
        self.NN_control = DeepNN(config, "actor")
        self.gamma = config.eqn_config.discount
        if self.train_config.scheme == "naive":
            self.propagate = self.bsde.propagate_naive
        elif self.train_config.scheme == "adaptive":
            self.propagate = self.bsde.propagate_adaptive
        
    def call(self, inputs, model_critic, training, cheat_value, cheat_control):
        x0, dw, x_bdry = inputs
        num_sample = np.shape(dw)[0]
        y = 0
        x, dt, coef = self.propagate(
            num_sample, x0, dw, self.NN_control, training,
            self.eqn_config.total_time_actor,
            self.eqn_config.num_time_interval_actor, cheat_control
        )
        discount = 1 #broadcast to num_sample x 1
        for t in range(self.eqn_config.num_time_interval_actor):
            if cheat_control == False:
                w = self.bsde.w_tf(x[:,:,t], self.NN_control(x[:,:,t], training, need_grad=False))
            else:
                w = self.bsde.w_tf(x[:,:,t], self.bsde.u_true(x[:,:,t]))
            y = y + coef[:,t:t+1] * w * dt[:,t:t+1] * discount
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
        self.eqn_config = config.eqn_config
        self.d = config.eqn_config.control_dim
        self.eqn = config.eqn_config.eqn_name
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
        elif AC == "actor" and self.eqn == "ekn":
            self.dense_layers.append(tf.keras.layers.Dense(self.d+1, activation=None))
        else:
            self.dense_layers.append(tf.keras.layers.Dense(self.d, activation=None))

    def call(self, x, training, need_grad):
        """structure: bn -> (dense -> bn -> relu) * len(num_hiddens) -> dense -> bn"""
        with tf.GradientTape() as g:
            if self.AC == "critic" and need_grad:
                g.watch(x)
            y = self.bn_layers[0](x, training)
            for i in range(len(self.dense_layers) - 1):
                y = self.dense_layers[i](y)
                y = self.bn_layers[i+1](y, training)
                y = y + tf.nn.relu(y)
            y = self.dense_layers[-1](y)
            y = self.bn_layers[-1](y, training)
            if self.AC == "actor" and self.eqn == "ekn":
                norm_y = tf.reduce_sum(y[:,0:self.d]**2, axis=1, keepdims=True)**0.5
                y = y[:,0:self.d] / (0.000000000000001 + tf.nn.relu(y[:,self.d:self.d+1]) + norm_y)
        if self.AC == "critic" and need_grad:
            return y, g.gradient(y, x)
        else:
            return y
