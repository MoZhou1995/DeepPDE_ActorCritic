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
        #dw, x, coef= inputs
        delta = self.model_critic(inputs, self.model_actor, training)
        # use linear approximation outside the clipped range
        loss = tf.reduce_mean(tf.where(tf.abs(delta) < DELTA_CLIP, tf.square(delta),
                                       2 * DELTA_CLIP * tf.abs(delta) - DELTA_CLIP ** 2))
        print("loss critic computed")
        return loss
    
    def loss_actor(self, inputs, training):
        print("call loss actor")
        #dw, x, coef= inputs
        y = self.model_actor(inputs, self.model_critic, training)
        # use linear approximation outside the clipped range
        loss = tf.reduce_mean(y)
        print("loss actor computed")
        return loss

    def grad_critic(self, inputs, training):
        print("call grad critic")
        with tf.GradientTape(persistent=True) as tape:
            loss_critic = self.loss_critic(inputs, training)
        #print("b_n",self.model_critic.NN_value.bn_layers)
        #print("dense ",self.model_critic.NN_value.dense_layers)
        #variable = [self.model_critic.NN_value.bn_layers,self.model_critic.NN_value.dense_layers]
        #print("variable",variable)
        #grad = tape.gradient(loss_critic, [self.model_critic.NN_value.bn_layers,self.model_critic.NN_value.dense_layers])
        grad = tape.gradient(loss_critic, self.model_critic.trainable_variables)
        del tape
        print("grad critic computed")
        return grad
    
    def grad_actor(self, inputs, training):
        print("call grad actor")
        with tf.GradientTape(persistent=True) as tape:
            loss_actor = self.loss_actor(inputs, training)
        #grad = tape.gradient(loss_actor, [self.model_actor.NN_control.bn_layers,self.model_actor.NN_control.dense_layers])
        grad = tape.gradient(loss_actor, self.model_actor.trainable_variables)
        del tape
        print("grad actor computed")
        return grad

    #@tf.function
    def train_step_critic(self, train_data):
        print("call train step critic")
        grad = self.grad_critic(train_data, training=True)
        self.optimizer_critic.apply_gradients(zip(grad, self.model_critic.trainable_variables))
        print("train step critic done")
        
    #@tf.function
    def train_step_actor(self, train_data):
        print("call train step actor")
        grad = self.grad_actor(train_data, training=True)
        self.optimizer_actor.apply_gradients(zip(grad, self.model_actor.trainable_variables))
        print("train step actor done")
        
    def control_fcn(self, x):
        #print("call control function")
        return self.model_actor.NN_control(x, training=True).numpy()
        #return self.sess.run(self.control, feed_dict = {self.x: x})
        
    def err_value(self, inputs):
        print("call value error")
        dw, x, coef= inputs
        x0 = x[:,:,0]
        return tf.reduce_mean(tf.square(self.bsde.V_true(x0) - self.model_critic.NN_value(x0, training=False)))
         
    def err_control(self, inputs):
        print("call control error")
        dw, x, coef= inputs
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
        self.y_init = tf.Variable(np.random.uniform(low=0,high=1,size=[1]))

    def call(self, inputs, model_actor, training):
        print("call critic model")
        dw, x, coef= inputs
        num_sample = np.shape(dw)[0]
        y = tf.Variable(initial_value=np.zeros([num_sample,1]), trainable=False, shape=[num_sample,1],dtype=self.net_config.dtype)
        #y = np.zeros([num_sample,1])
        for t in range(self.bsde.num_time_interval):
            y = y + tf.reshape(coef[:,t], [num_sample,1]) * self.bsde.w_tf(x[:,:,t], model_actor.NN_control(x[:,:,t], training))
        delta = self.NN_value(x[:,:,0]) - y - self.NN_value(x[:,:,-1])
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
        dw, x, coef= inputs
        num_sample = np.shape(dw)[0]
        y = tf.Variable(initial_value=np.zeros([num_sample,1]), trainable=False, shape=[num_sample,1],dtype=self.net_config.dtype)
        #y = np.zeros([num_sample])
        for t in range(self.bsde.num_time_interval):
            y = y + tf.reshape(coef[:,t], [num_sample,1]) * self.bsde.w_tf(x[:,:,t], self.NN_control(x[:,:,t], training))
        y = y + model_critic.NN_value(x[:,:,-1])
        print("actor model computed")
        return y


class DeepNN(tf.keras.Model):
    def __init__(self, config, AC):
        super(DeepNN, self).__init__()
        print("define NN")
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

    def call(self, x, training):
        """structure: bn -> (dense -> bn -> relu) * len(num_hiddens) -> dense -> bn"""
        #print("call NN")
        #print(self.AC, "input size", tf.shape(x))
        x = self.bn_layers[0](x, training)
        for i in range(len(self.dense_layers) - 1):
            x = self.dense_layers[i](x)
            x = self.bn_layers[i+1](x, training)
            x = tf.nn.relu(x)
        x = self.dense_layers[-1](x)
        x = self.bn_layers[-1](x, training)
        #print("output size", tf.shape(x))
        return x
