import tensorflow as tf
import numpy as np
from collections import deque
import random
class DDPGAgent:
    def __init__(self, state_dim, action_dim,sess, action_bound=[], name='brain'):
        self.sess = sess
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_holder = tf.placeholder(tf.float32, [None, 1], name='action')
        self.reward_holder = tf.placeholder(tf.float32, [None, 1], name='reward')
        self.state_holder = tf.placeholder(tf.float32, [None, state_dim], name='state')
        self.next_state_holder = tf.placeholder(tf.float32, [None, state_dim], name='next_state')
        self.done_holder = tf.placeholder(tf.float32, [None, 1], name='done')
        self.action_bound = action_bound
        self.name = name
        self.memory = deque(maxlen=20000)
        self.update_step = 0
        self.num_experience = 0
        self.gamma = 0.9
        self.var = 2.

    def buildCriticNetwork(self, ):
        init_w = tf.random_normal_initializer(0., 1.)
        init_b = tf.constant_initializer(0.01)
        with tf.variable_scope('eva-Critic_network' + self.name):
            # enc
            q_layer1s = tf.layers.Dense(128, activation=tf.nn.elu,
                                            kernel_initializer=init_w, bias_initializer=init_b, name='q_layer1s',
                                            trainable=True)
            q_layer1a = tf.layers.Dense(128, activation=tf.nn.elu,
                                            kernel_initializer=init_w, bias_initializer=init_b, name='q_layer1a',
                                            trainable=True)
            self.qs = q_layer1s(self.state_holder)
            self.qa = q_layer1a(self.action_holder)
            q_estimate = tf.layers.Dense(1, activation=None,
                                     kernel_initializer=tf.random_normal_initializer(0., 0.01), bias_initializer=init_b, name='q_estimate',
                                     trainable=True)
            self.q = q_estimate(self.qs+self.qa)
            self.q_policy_eva = q_estimate(self.qs+q_layer1a(self.pi))
        with tf.variable_scope('tar-Critic_network' + self.name):
            # enc
            q_layer1s_ = tf.layers.Dense(128, activation=tf.nn.relu,
                                            kernel_initializer=init_w, bias_initializer=init_b, name='q_layer1s_',
                                            trainable=False)
            q_layer1a_ = tf.layers.Dense(128, activation=tf.nn.relu,
                                            kernel_initializer=init_w, bias_initializer=init_b, name='q_layer1a_',
                                            trainable=False)
            qs_ = q_layer1s_(self.next_state_holder)
            qa_ = q_layer1a_(self.pi_)
            q_estimate_ = tf.layers.Dense(1, activation=None,
                                     kernel_initializer=tf.random_normal_initializer(0., 0.01), bias_initializer=init_b, name='q_estimate_',
                                     trainable=False)
            self.tq = tf.stop_gradient(q_estimate_(qs_+qa_))
			
        self.q_e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eva-Critic_network' + self.name)
        self.q_t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='tar-Critic_network' + self.name)
			
    def buildActorNetwork(self):
        init_w = tf.random_normal_initializer(0., 1.)
        init_b = tf.constant_initializer(0.01)
        with tf.variable_scope('eva-Actor_network' + self.name):
            a_layer1 = tf.layers.Dense(128, activation=tf.nn.elu,kernel_regularizer=tf.contrib.layers.l2_regularizer(0.01),
                                            kernel_initializer=init_w, bias_initializer=init_b, name='a_layer1',
                                            trainable=True)
            a_layer2 = tf.layers.Dense(128, activation=tf.nn.elu,kernel_regularizer=tf.contrib.layers.l2_regularizer(0.01),
                                            kernel_initializer=init_w, bias_initializer=init_b, name='a_layer2',
                                            trainable=True)

            h1 = a_layer1(self.state_holder)
            h2 = a_layer2(h1)
            action_selector = tf.layers.Dense(self.action_dim, activation=tf.nn.tanh,kernel_regularizer=tf.contrib.layers.l2_regularizer(0.01),
                                              kernel_initializer=tf.random_normal_initializer(0., 0.01), bias_initializer=init_b, name='as',
                                              trainable=True)

            self.pi = action_selector(h2)#tf.clip_by_value(1.5*action_selector(h),-1.5,1.)
            
        with tf.variable_scope('tar-Actor_network' + self.name):
            a_layer1_ = tf.layers.Dense(128, activation=tf.nn.elu,kernel_regularizer=tf.contrib.layers.l2_regularizer(0.01),
                                            kernel_initializer=init_w, bias_initializer=init_b, name='a_layer1_',
                                            trainable=False)
            a_layer2_ = tf.layers.Dense(128, activation=tf.nn.elu,kernel_regularizer=tf.contrib.layers.l2_regularizer(0.01),
                                            kernel_initializer=init_w, bias_initializer=init_b, name='a_layer2_',
                                            trainable=True)

            h1_ = a_layer1_(self.next_state_holder)
            h2_ = a_layer2_(h1_)

            action_selector_ = tf.layers.Dense(self.action_dim, activation=tf.nn.tanh,kernel_regularizer=tf.contrib.layers.l2_regularizer(0.01),  
                                              kernel_initializer=tf.random_normal_initializer(0., 0.01), bias_initializer=init_b, name='as_',
                                              trainable=False)
            self.pi_ = action_selector_(h2_)#tf.clip_by_value(1.5*action_selector_(h_),-1.5,1.)

        self.p_e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eva-Actor_network' + self.name)
        self.p_t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='tar-Actor_network' + self.name)

 

	
    def setLearn(self,LR_C = 0.001, LR_A = 0.00025):
        self.q_tar = self.reward_holder + self.gamma*self.tq*(1-self.done_holder)
        self.q_loss = tf.reduce_mean(tf.squared_difference(self.q, self.q_tar))
        self.reg_loss = tf.losses.get_regularization_loss()
        self.p_loss =  tf.reduce_mean(self.q_policy_eva) 
        
        self.q_trainOp = tf.train.AdamOptimizer(LR_C).minimize(self.q_loss, var_list=self.q_e_params)
        self.p_trainOp = tf.train.AdamOptimizer(LR_A).minimize(-self.p_loss, var_list=self.p_e_params)
        
    def setUpdate(self, TAU = 0.01, update_interval = 1):
        self.updateInterval = update_interval
        self.Critic_network_update = [tf.assign(tar, (1 -  TAU) * tar +  TAU * eva) for tar, eva in zip(self.q_t_params, self.q_e_params)]
        self.Actor_network_update = [tf.assign(tar, (1 -  TAU) * tar +  TAU * eva) for tar, eva in zip(self.p_t_params, self.p_e_params)]
    
    def choose_action(self, state):
        a = np.array(self.sess.run(self.pi, feed_dict={self.state_holder:state})).reshape(-1,)
        a = np.random.normal(a[0], self.var)
        
        return   a # returns action

    def remember(self, state, action, reward, next_state,done):
        self.num_experience+=1
        self.memory.append((state, action, reward, next_state,done))

    def learn(self,batch_size=32):
        states, next_states, actions, rewards,dones = [], [], [], [],[]
        #minibatch = list(self.memory)[:]
        minibatch = random.sample(list(self.memory), batch_size)
        for state, action, reward, next_state, done in minibatch:
            states.append(state)
            next_states.append(next_state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
        actions = np.asarray(actions).reshape(-1,1)
        dones = np.asarray(dones).reshape(-1,1)
        rewards = np.asarray(rewards)
        states = np.array(states).reshape((-1, self.state_dim))
        next_states = np.array(next_states).reshape((-1, self.state_dim))
        
        self.sess.run(self.p_trainOp,feed_dict={self.state_holder: states,self.done_holder:dones})
        self.sess.run(self.q_trainOp,feed_dict={self.state_holder: states,self.action_holder: actions,
                      self.reward_holder: np.array(rewards).reshape(-1, 1),
                      self.next_state_holder: next_states,
                      self.done_holder:dones})
        self.qloss, self.ploss, reg_loss,Q  = self.sess.run(
            [   self.q_loss, self.p_loss, self.reg_loss,self.q],
            feed_dict={self.state_holder: states,
                       self.action_holder: actions,
                       self.reward_holder: np.array(rewards).reshape(-1, 1),self.done_holder:dones,
                       self.next_state_holder: next_states})
        self.update_step += 1
        if self.update_step % self.updateInterval == 0:
            self.sess.run(self.Actor_network_update)
            self.sess.run(self.Critic_network_update)

        return self.ploss, self.qloss, reg_loss,Q
