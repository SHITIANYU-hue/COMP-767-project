import tensorflow as tf
import numpy as np
from collections import deque
import random
# SEED=6666
# random.seed(SEED)
# np.random.seed(SEED)
# tf.set_random_seed(SEED)

class DGN:
    def __init__(self, state_dim, action_dim,sess, action_bound=[], neighbors=2,agent_num=3,name='brain'):
        self.sess = sess
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.reward_holder = tf.placeholder(tf.float32, [None, agent_num], name='reward')
        self.accumulated_reward_holder = tf.placeholder(tf.float32, [None, agent_num], name='reward_discounted_sum')
        self.action_holder = tf.placeholder(tf.float32, [None, agent_num], name='action')
        self.state_holder = tf.placeholder(tf.float32, [None, state_dim*agent_num], name='state')
        self.next_state_holder = tf.placeholder(tf.float32, [None, state_dim*agent_num], name='next_state')
        self.next_state_value_holder =  tf.placeholder(tf.float32, [None, agent_num], name='next_value')
        self.done_holder = tf.placeholder(tf.float32, [None, 1], name='done')
        self.action_bound = action_bound
        self.name = name
        self.memory = deque(maxlen=20000)
        self.update_step = 0
        self.num_experience = 0
        self.gamma = 0.9
        self.var = 2.
        self.neighbors = neighbors
        self.agent_num = agent_num
        self.adj = tf.placeholder(tf.float32, [None,self.agent_num,self.neighbors, self.agent_num], name='self.adj')
        self.vecholder = tf.placeholder(tf.float32,[None, 1, self.neighbors])
        self.epsilon_holder = tf.placeholder(tf.float32, None, name='epsilon')

    def buildCriticNetwork(self, d=128,dv=16,dout=128,nv=8):
        init_w = tf.random_normal_initializer(0., 0.01)
        init_b = tf.constant_initializer(0.01)
        with tf.variable_scope('update_Critic_network' + self.name):
            # enc
            f_dim = 128
            encode_layer1 = tf.keras.layers.Dense(128, activation=tf.nn.elu,
                                            kernel_initializer=init_w, bias_initializer=init_b, name='encoder_l1',
                                            trainable=True) ##change the encoding dim
            encode_layer2 = tf.keras.layers.Dense(f_dim, activation=tf.nn.elu,
                                            kernel_initializer=init_w, bias_initializer=init_b, name='encoder_l2',
                                            trainable=True)

            for i in range(self.agent_num):
                e1 = encode_layer1(self.state_holder[:, i * self.state_dim:(i + 1) * self.state_dim])
                feature = encode_layer2(e1)
                if i == 0:
                    self.feature_c = feature
                else:
                    self.feature_c = tf.concat([self.feature_c, feature], axis=1)

            self.feature_c = tf.reshape(self.feature_c, [-1, f_dim, self.agent_num]) ###gai
           # self.feature_c = tf.transpose(self.feature_c, [0, 2, 1])
            # relation1
            d = d
            dv = dv
            dout = dout
            nv = nv

            r1_l1_v = tf.keras.layers.Dense(dv * nv, activation=tf.nn.relu,
                                      kernel_initializer=init_w, bias_initializer=init_b, name='relation_l1_v',
                                      trainable=True)
            r1_l1_q = tf.keras.layers.Dense(dv * nv, activation=tf.nn.relu,
                                      kernel_initializer=init_w, bias_initializer=init_b, name='relation_l1_q',
                                      trainable=True)
            r1_l1_k = tf.keras.layers.Dense(dv * nv, activation=tf.nn.relu,
                                      kernel_initializer=init_w, bias_initializer=init_b, name='relation_l1_k',
                                      trainable=True)
            r1_out = tf.keras.layers.Dense(dout, activation=tf.nn.relu,
                                     kernel_initializer=init_w, bias_initializer=init_b, name='relation_l1_k',
                                     trainable=True)
            for i in range(self.agent_num):
                v1 = tf.matmul(self.feature_c, tf.transpose(self.adj[:, i, :, :], [0, 2, 1]))
                q1 = tf.matmul(self.feature_c, tf.transpose(self.adj[:, i, :, :], [0, 2, 1]))
                k1 = tf.matmul(self.feature_c, tf.transpose(self.adj[:, i, :, :], [0, 2, 1]))

                v1 = tf.transpose(v1, [0, 2, 1])
                q1 = tf.transpose(q1, [0, 2, 1])
                k1 = tf.transpose(k1, [0, 2, 1])

                v2 = r1_l1_v(v1)
                q2 = r1_l1_q(q1)
                k2 = r1_l1_k(k1)
		
		# Can we replace the graph attention layer with another structure?

                v = tf.reshape(v2, shape=[-1, self.neighbors, nv, dv])
                q = tf.reshape(q2, shape=[-1, self.neighbors, nv, dv])
                k = tf.reshape(k2, shape=[-1, self.neighbors, nv, dv])

                v = tf.transpose(v, [0, 2, 1, 3])
                k = tf.transpose(k, [0, 2, 3, 1])
                q = tf.transpose(q, [0, 2, 1, 3])

                att = tf.matmul(q, k) / np.sqrt(dv)
                att = tf.nn.softmax(att, axis=-1)

                out = tf.matmul(att, v)
                out = tf.transpose(out, [0, 2, 1, 3])

                out = tf.reshape(out, shape=[-1, self.neighbors, dv * nv])

                T = tf.matmul(self.vecholder, out)

                out = r1_out(T)
                out = tf.reshape(out,(-1,dv*nv))
                if i == 0:
                    self.relation_1_c = out
                else:
                    self.relation_1_c = tf.concat([self.relation_1_c, out], axis=1)

            self.relation_1_c = tf.reshape(self.relation_1_c, [-1, dv * nv, self.agent_num])  ##gai
            # relation 2
            r2_l1_v = tf.keras.layers.Dense(dv * nv, activation=tf.nn.relu,
                                      kernel_initializer=init_w, bias_initializer=init_b, name='relation_l2_v',
                                      trainable=True)
            r2_l1_q = tf.keras.layers.Dense(dv * nv, activation=tf.nn.relu,
                                      kernel_initializer=init_w, bias_initializer=init_b, name='relation_l2_q',
                                      trainable=True)
            r2_l1_k = tf.keras.layers.Dense(dv * nv, activation=tf.nn.relu,
                                      kernel_initializer=init_w, bias_initializer=init_b, name='relation_l2_k',
                                      trainable=True)
            r2_out = tf.keras.layers.Dense(dout, activation=tf.nn.relu,
                                     kernel_initializer=init_w, bias_initializer=init_b, name='relation_l2_k',
                                     trainable=True)
            for i in range(self.agent_num):
                v1 = tf.matmul(self.relation_1_c, tf.transpose(self.adj[:, i, :, :], [0, 2, 1]))
                q1 = tf.matmul(self.relation_1_c, tf.transpose(self.adj[:, i, :, :], [0, 2, 1]))
                k1 = tf.matmul(self.relation_1_c, tf.transpose(self.adj[:, i, :, :], [0, 2, 1]))

                v1 = tf.transpose(v1, [0, 2, 1])
                q1 = tf.transpose(q1, [0, 2, 1])
                k1 = tf.transpose(k1, [0, 2, 1])

                v2 = r2_l1_v(v1)
                q2 = r2_l1_q(q1)
                k2 = r2_l1_k(k1)

                v = tf.reshape(v2, shape=[-1, self.neighbors, nv, dv])
                q = tf.reshape(q2, shape=[-1, self.neighbors, nv, dv])
                k = tf.reshape(k2, shape=[-1, self.neighbors, nv, dv])

                v = tf.transpose(v, [0, 2, 1, 3])
                k = tf.transpose(k, [0, 2, 3, 1])
                q = tf.transpose(q, [0, 2, 1, 3])

                att = tf.matmul(q, k) / np.sqrt(dv)
                att = tf.nn.softmax(att, axis=-1)

                out = tf.matmul(att, v)
                out = tf.transpose(out, [0, 2, 1, 3])

                out = tf.reshape(out, shape=[-1, self.neighbors, dv * nv])

                T = tf.matmul(self.vecholder, out)

                out = r2_out(T)
                out = tf.reshape(out,(-1,dv*nv))
                if i == 0:
                    self.relation_2_c = out
                else:
                    self.relation_2_c = tf.concat([self.relation_2_c, out], axis=1)

        v_estimate = tf.keras.layers.Dense(1, activation=tf.nn.relu,
                                     kernel_initializer=init_w, bias_initializer=init_b, name='v_estimate',
                                     trainable=True)
        for i in range(self.agent_num):
            f=self.feature_c[:, :, i]
            f=tf.reshape(f,(-1,f_dim))
            h = tf.concat([f, self.relation_1_c[:,:,i], self.relation_2_c[:, i*dv*nv:(i+1)*dv*nv]], axis=1)
            adv=self.reward_holder[:, i] + self.gamma*self.next_state_value_holder[:, i]*(1-self.done_holder) - v_estimate(h)
            if i == 0:
                self.v = v_estimate(h)
                self.advantage = adv
            else:
                self.v = tf.concat([self.v, v_estimate(h)], axis=1)
                self.advantage = tf.concat([self.advantage, adv], axis=1)
        self.v_e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='update_Critic_network' + self.name)
        self.v_loss = tf.reduce_mean(self.advantage**2 )
        self.v_trainOp = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.v_loss)   ###to change

    def buildActorNetwork(self,d=128,dv=16,dout=128,nv=8):
        init_w = tf.random_normal_initializer(0., 0.01)
        init_b = tf.constant_initializer(0.01)
        with tf.variable_scope('update_Actor_network' + self.name):
            # enc
            f_dim = 128
            encode_layer1 = tf.keras.layers.Dense(128, activation=tf.nn.elu,
                                            kernel_initializer=init_w, bias_initializer=init_b, name='encoder_l1',
                                            trainable=True)
            encode_layer2 = tf.keras.layers.Dense(f_dim, activation=tf.nn.elu,
                                            kernel_initializer=init_w, bias_initializer=init_b, name='encoder_l2',
                                            trainable=True)

            for i in range(self.agent_num):
                e1 = encode_layer1(self.state_holder[:, i * self.state_dim:(i + 1) * self.state_dim])
                feature = encode_layer2(e1)
                if i == 0:
                    self.feature_a = feature
                else:
                    self.feature_a = tf.concat([self.feature_a, feature], axis=1)

            self.feature_a = tf.reshape(self.feature_a, [-1, f_dim, self.agent_num]) ##gai
            # relation1
            d = d
            dv = dv
            dout = dout
            nv = nv

            r1_l1_v = tf.keras.layers.Dense(dv * nv, activation=tf.nn.relu,
                                      kernel_initializer=init_w, bias_initializer=init_b, name='relation_l1_v',
                                      trainable=True)
            r1_l1_q = tf.keras.layers.Dense(dv * nv, activation=tf.nn.relu,
                                      kernel_initializer=init_w, bias_initializer=init_b, name='relation_l1_q',
                                      trainable=True)
            r1_l1_k = tf.keras.layers.Dense(dv * nv, activation=tf.nn.relu,
                                      kernel_initializer=init_w, bias_initializer=init_b, name='relation_l1_k',
                                      trainable=True)
            r1_out = tf.keras.layers.Dense(dout, activation=tf.nn.relu,
                                     kernel_initializer=init_w, bias_initializer=init_b, name='relation_l1_k',
                                     trainable=True)
            for i in range(self.agent_num):
                v1 = tf.matmul(self.feature_a, tf.transpose(self.adj[:, i, :, :], [0, 2, 1]))
                q1 = tf.matmul(self.feature_a, tf.transpose(self.adj[:, i, :, :], [0, 2, 1]))
                k1 = tf.matmul(self.feature_a, tf.transpose(self.adj[:, i, :, :], [0, 2, 1]))

                v1 = tf.transpose(v1, [0, 2, 1])
                q1 = tf.transpose(q1, [0, 2, 1])
                k1 = tf.transpose(k1, [0, 2, 1])

                v2 = r1_l1_v(v1)
                q2 = r1_l1_q(q1)
                k2 = r1_l1_k(k1)

                v = tf.reshape(v2, shape=[-1, self.neighbors, nv, dv])
                q = tf.reshape(q2, shape=[-1, self.neighbors, nv, dv])
                k = tf.reshape(k2, shape=[-1, self.neighbors, nv, dv])

                v = tf.transpose(v, [0, 2, 1, 3])
                k = tf.transpose(k, [0, 2, 3, 1])
                q = tf.transpose(q, [0, 2, 1, 3])

                att = tf.matmul(q, k) / np.sqrt(dv)
                att = tf.nn.softmax(att, axis=-1)

                out = tf.matmul(att, v)
                out = tf.transpose(out, [0, 2, 1, 3])

                out = tf.reshape(out, shape=[-1, self.neighbors, dv * nv])

                T = tf.matmul(self.vecholder, out)

                out = r1_out(T)

                if i == 0:
                    self.relation_1_a = out
                else:
                    self.relation_1_a = tf.concat([self.relation_1_a, out], axis=1)

            self.relation_1_a = tf.reshape(self.relation_1_a, [-1, dv * nv, self.agent_num]) ##gai

            # relation 2
            r2_l1_v = tf.keras.layers.Dense(dv * nv, activation=tf.nn.relu,
                                      kernel_initializer=init_w, bias_initializer=init_b, name='relation_l2_v',
                                      trainable=True)
            r2_l1_q = tf.keras.layers.Dense(dv * nv, activation=tf.nn.relu,
                                      kernel_initializer=init_w, bias_initializer=init_b, name='relation_l2_q',
                                      trainable=True)
            r2_l1_k = tf.keras.layers.Dense(dv * nv, activation=tf.nn.relu,
                                      kernel_initializer=init_w, bias_initializer=init_b, name='relation_l2_k',
                                      trainable=True)
            r2_out = tf.keras.layers.Dense(dout, activation=tf.nn.relu,
                                     kernel_initializer=init_w, bias_initializer=init_b, name='relation_l2_k',
                                     trainable=True)
            for i in range(self.agent_num):
                v1 = tf.matmul(self.relation_1_a, tf.transpose(self.adj[:, i, :, :], [0, 2, 1]))
                q1 = tf.matmul(self.relation_1_a, tf.transpose(self.adj[:, i, :, :], [0, 2, 1]))
                k1 = tf.matmul(self.relation_1_a, tf.transpose(self.adj[:, i, :, :], [0, 2, 1]))

                v1 = tf.transpose(v1, [0, 2, 1])
                q1 = tf.transpose(q1, [0, 2, 1])
                k1 = tf.transpose(k1, [0, 2, 1])

                v2 = r2_l1_v(v1)
                q2 = r2_l1_q(q1)
                k2 = r2_l1_k(k1)

                v = tf.reshape(v2, shape=[-1, self.neighbors, nv, dv])
                q = tf.reshape(q2, shape=[-1, self.neighbors, nv, dv])
                k = tf.reshape(k2, shape=[-1, self.neighbors, nv, dv])

                v = tf.transpose(v, [0, 2, 1, 3])
                k = tf.transpose(k, [0, 2, 3, 1])
                q = tf.transpose(q, [0, 2, 1, 3])

                att = tf.matmul(q, k) / np.sqrt(dv)
                att = tf.nn.softmax(att, axis=-1)

                out = tf.matmul(att, v)
                out = tf.transpose(out, [0, 2, 1, 3])

                out = tf.reshape(out, shape=[-1, self.neighbors, dv * nv])

                T = tf.matmul(self.vecholder, out)

                out = r2_out(T)

                if i == 0:
                    self.relation_2_a = out
                else:
                    self.relation_2_a = tf.concat([self.relation_2_a, out], axis=1)

            self.action_mean = tf.keras.layers.Dense(1, activation=None,
                                              kernel_initializer=init_w, bias_initializer=init_b, name='mean',
                                              trainable=True)
            self.action_sigma = tf.keras.layers.Dense(1, activation=None,
                                              kernel_initializer=init_w, bias_initializer=init_b, name='sigma',
                                              trainable=True)                               
            self.pi = []
            self.action = []
            for i in range(self.agent_num):
                h = tf.concat([self.feature_a[:, :, i], self.relation_1_a[:, i, :],
                               self.relation_2_a[:, i, :]], axis=1)
                dis = tf.distributions.Normal(loc=self.action_mean(h), scale=self.action_sigma(h))
       
                self.pi.append(dis)
                self.action.append(tf.squeeze(dis.sample([1])))


        with tf.variable_scope('target_Actor_network' + self.name):
            # enc
            f_dim = 128
            encode_layer1 = tf.keras.layers.Dense(128, activation=tf.nn.elu,
                                            kernel_initializer=init_w, bias_initializer=init_b, name='encoder_l1',
                                            trainable=True)
            encode_layer2 = tf.keras.layers.Dense(f_dim, activation=tf.nn.elu,
                                            kernel_initializer=init_w, bias_initializer=init_b, name='encoder_l2',
                                            trainable=True)

            for i in range(self.agent_num):
                e1 = encode_layer1(self.state_holder[:, i * self.state_dim:(i + 1) * self.state_dim])
                feature = encode_layer2(e1)
                if i == 0:
                    self.feature_a_old = feature
                else:
                    self.feature_a_old = tf.concat([self.feature_a_old, feature], axis=1)
            self.feature_a_old = tf.reshape(self.feature_a_old, [-1, 128, self.agent_num]) ##gai
            # relation1

            d = 128
            dv = 16
            dout = 128
            nv = 8

            r1_l1_v = tf.keras.layers.Dense(dv * nv, activation=tf.nn.relu,
                                      kernel_initializer=init_w, bias_initializer=init_b, name='relation_l1_v',
                                      trainable=True)
            r1_l1_q = tf.keras.layers.Dense(dv * nv, activation=tf.nn.relu,
                                      kernel_initializer=init_w, bias_initializer=init_b, name='relation_l1_q',
                                      trainable=True)
            r1_l1_k = tf.keras.layers.Dense(dv * nv, activation=tf.nn.relu,
                                      kernel_initializer=init_w, bias_initializer=init_b, name='relation_l1_k',
                                      trainable=True)
            r1_out = tf.keras.layers.Dense(dout, activation=tf.nn.relu,
                                     kernel_initializer=init_w, bias_initializer=init_b, name='relation_l1_k',
                                     trainable=True)
            for i in range(self.agent_num):
                v1 = tf.matmul(self.feature_a_old, tf.transpose(self.adj[:, i, :, :], [0, 2, 1]))
                q1 = tf.matmul(self.feature_a_old, tf.transpose(self.adj[:, i, :, :], [0, 2, 1]))
                k1 = tf.matmul(self.feature_a_old, tf.transpose(self.adj[:, i, :, :], [0, 2, 1]))

                v1 = tf.transpose(v1, [0, 2, 1])
                q1 = tf.transpose(q1, [0, 2, 1])
                k1 = tf.transpose(k1, [0, 2, 1])

                v2 = r1_l1_v(v1)
                q2 = r1_l1_q(q1)
                k2 = r1_l1_k(k1)

                v = tf.reshape(v2, shape=[-1, self.neighbors, nv, dv])
                q = tf.reshape(q2, shape=[-1, self.neighbors, nv, dv])
                k = tf.reshape(k2, shape=[-1, self.neighbors, nv, dv])

                v = tf.transpose(v, [0, 2, 1, 3])
                k = tf.transpose(k, [0, 2, 3, 1])
                q = tf.transpose(q, [0, 2, 1, 3])

                att = tf.matmul(q, k) / np.sqrt(dv)
                att = tf.nn.softmax(att, axis=-1)

                out = tf.matmul(att, v)
                out = tf.transpose(out, [0, 2, 1, 3])

                out = tf.reshape(out, shape=[-1, self.neighbors, dv * nv])

                T = tf.matmul(self.vecholder, out)

                out = r1_out(T)

                if i == 0:
                    self.relation_1_a_old = out
                else:
                    self.relation_1_a_old = tf.concat([self.relation_1_a_old, out], axis=1)
            self.relation_1_a_old = tf.reshape(self.relation_1_a_old, [-1, dv * nv, self.agent_num])  ##
            # relation 2
            r2_l1_v = tf.keras.layers.Dense(dv * nv, activation=tf.nn.relu,
                                      kernel_initializer=init_w, bias_initializer=init_b, name='relation_l2_v',
                                      trainable=True)
            r2_l1_q = tf.keras.layers.Dense(dv * nv, activation=tf.nn.relu,
                                      kernel_initializer=init_w, bias_initializer=init_b, name='relation_l2_q',
                                      trainable=True)
            r2_l1_k = tf.keras.layers.Dense(dv * nv, activation=tf.nn.relu,
                                      kernel_initializer=init_w, bias_initializer=init_b, name='relation_l2_k',
                                      trainable=True)
            r2_out = tf.keras.layers.Dense(dout, activation=tf.nn.relu,
                                     kernel_initializer=init_w, bias_initializer=init_b, name='relation_l2_k',
                                     trainable=True)
            for i in range(self.agent_num):
                v1 = tf.matmul(self.relation_1_a_old, tf.transpose(self.adj[:, i, :, :], [0, 2, 1]))
                q1 = tf.matmul(self.relation_1_a_old, tf.transpose(self.adj[:, i, :, :], [0, 2, 1]))
                k1 = tf.matmul(self.relation_1_a_old, tf.transpose(self.adj[:, i, :, :], [0, 2, 1]))

                v1 = tf.transpose(v1, [0, 2, 1])
                q1 = tf.transpose(q1, [0, 2, 1])
                k1 = tf.transpose(k1, [0, 2, 1])

                v2 = r2_l1_v(v1)
                q2 = r2_l1_q(q1)
                k2 = r2_l1_k(k1)

                v = tf.reshape(v2, shape=[-1, self.neighbors, nv, dv])
                q = tf.reshape(q2, shape=[-1, self.neighbors, nv, dv])
                k = tf.reshape(k2, shape=[-1, self.neighbors, nv, dv])

                v = tf.transpose(v, [0, 2, 1, 3])
                k = tf.transpose(k, [0, 2, 3, 1])
                q = tf.transpose(q, [0, 2, 1, 3])

                att = tf.matmul(q, k) / np.sqrt(dv)
                att = tf.nn.softmax(att, axis=-1)

                out = tf.matmul(att, v)
                out = tf.transpose(out, [0, 2, 1, 3])

                out = tf.reshape(out, shape=[-1, self.neighbors, dv * nv])

                T = tf.matmul(self.vecholder, out)

                out = r2_out(T)

                if i == 0:
                    self.relation_2_a_old = out
                else:
                    self.relation_2_a_old = tf.concat([self.relation_2_a_old, out], axis=1)

            self.action_mean_old = tf.keras.layers.Dense(1, activation=None,
                                              kernel_initializer=init_w, bias_initializer=init_b, name='mean_old',
                                              trainable=False)
            self.action_sigma_old = tf.keras.layers.Dense(1, activation=None,
                                              kernel_initializer=init_w, bias_initializer=init_b, name='sigma_old',
                                              trainable=False)                               
            self.pi_old = []
            self.action_old = []
            for i in range(self.agent_num):
                h = tf.concat([self.feature_a_old[:, :, i], self.relation_1_a_old[:, i, :],
                               self.relation_2_a_old[:, i, :]], axis=1)
                dis = tf.distributions.Normal(loc=self.action_mean_old(h), scale=self.action_sigma_old(h))
        
                self.pi_old.append(dis)
                self.action_old.append(tf.squeeze(dis.sample([1])))

        self.p_e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='update_Actor_network' + self.name)
        self.p_t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_Actor_network' + self.name)

        # train setting
        self.p_trainOp = []
        for i in range(self.agent_num):
            ratio = tf.exp(
                tf.reshape(self.pi[i].log_prob(self.action_holder[:, i]), [-1, 1]) - tf.reshape(
                    tf.clip_by_value(self.pi_old[i].log_prob(self.action_holder[:, i]),
                                     -20, 20), [-1, 1]))
 
            self.surrogate = ratio * self.advantage[:, i]
            self.clip_surrogate = tf.clip_by_value(ratio, 1. - self.epsilon_holder,
                                                   1 + self.epsilon_holder) * self.advantage[:, i]
            self.p_loss = -tf.reduce_mean(tf.minimum(self.surrogate, self.clip_surrogate))

            grads, _ = tf.clip_by_global_norm(tf.gradients(self.p_loss, self.p_e_params), 5.)
            grads_and_vars = list(zip(grads, self.p_e_params))
            self.p_trainOp.append(
                tf.train.AdamOptimizer(learning_rate=0.0001).apply_gradients(grads_and_vars, name="apply_gradients"))
        self.Actor_network_update = [tf.assign(tar, eva) for tar, eva in zip(self.p_t_params, self.p_e_params)]

	
    def setLearn(self,LR_C = 0.001, LR_A = 0.00025):
        self.q_tar = self.reward_holder + self.gamma*self.next_state_value_holder*(1-self.done_holder)
        self.q_loss = tf.reduce_mean(tf.squared_difference(self.v, self.q_tar))
        self.reg_loss = tf.losses.get_regularization_loss()
        self.p_loss =  tf.reduce_mean(self.q_policy_eva) 
        
        
        self.p_trainOp = tf.train.AdamOptimizer(LR_A).minimize(-self.p_loss, var_list=self.p_e_params)
        
    def setUpdate(self, TAU = 0.01, update_interval = 1):
        self.updateInterval = update_interval
        #self.Critic_network_update = [tf.assign(tar, (1 -  TAU) * tar +  TAU * eva) for tar, eva in zip(self.q_t_params, self.q_e_params)]
        self.Actor_network_update = [tf.assign(tar, (1 -  TAU) * tar +  TAU * eva) for tar, eva in zip(self.p_t_params, self.p_e_params)]
    
    def choose_action(self, state):
        a = np.array(self.sess.run(self.pi, feed_dict={self.state_holder:state})).reshape(-1,)
        return  a # returns action

    def remember(self, state, action, reward, next_state,done,adjs):
        self.num_experience+=1
        self.memory.append((state, action, reward, next_state,done,adjs))

    def learn(self,batch_size=32):
        states, next_states, actions, rewards,dones,v_preds,adj_set = [], [], [], [],[],[],[]
        #minibatch = list(self.memory)[:]
        minibatch = random.sample(list(self.memory), batch_size)
        vec = np.zeros([1,self.neighbors])
        vec[0,0]=1.
        for state, action, reward, next_state, done ,adjs in minibatch:
            states.append(state)
            next_states.append(next_state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            adj_set.append(adjs)
            v_pred = self.sess.run(self.v, feed_dict={self.vecholder:[vec],self.adj:[adjs],self.state_holder:np.array(next_state).reshape(1,-1)})
            v_preds.append(v_pred)
        actions = np.asarray(actions).reshape(-1,self.agent_num)
        dones = np.asarray(dones).reshape(-1,1)
        rewards = np.asarray(rewards)
        states = np.array(states).reshape((-1, self.state_dim*self.agent_num))
        next_states = np.array(next_states).reshape((-1, self.state_dim*self.agent_num))
        adj_set = np.array(adj_set).reshape(-1,self.agent_num,self.neighbors,self.agent_num)
        
        _,_,self.qloss, self.ploss = self.sess.run(
            [ self.p_trainOp, self.v_trainOp, self.v_loss, self.p_loss ],
            feed_dict={self.state_holder: states,
                       self.action_holder: actions,
                       self.vecholder:[vec],
                       self.reward_holder: np.array(rewards).reshape(-1, self.agent_num),
                       self.next_state_value_holder:np.array(v_preds).reshape(-1, self.agent_num),
                       self.done_holder:dones,
                       self.adj:adj_set,
                       self.epsilon_holder:0.3,
                       self.next_state_holder: next_states})
        self.update_step += 1
        if self.update_step % self.updateInterval == 0:
            self.sess.run(self.Actor_network_update)
    

        return self.ploss, self.qloss 
