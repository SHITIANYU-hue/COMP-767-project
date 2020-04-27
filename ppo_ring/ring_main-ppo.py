# Import all of the necessary pieces of Flow to run the experiments
from flow.core.params import SumoParams, EnvParams, NetParams, InitialConfig, \
    InFlows, SumoLaneChangeParams, SumoCarFollowingParams
from flow.core.params import VehicleParams
from flow.core.params import TrafficLightParams

from flow.controllers import SimLaneChangeController, ContinuousRouter
from flow.core.experiment import Experiment
from ring_Env import Experiment
import logging
from stable_baselines import PPO2

import datetime
import numpy as np
import time
import os
import tensorflow as tf
from ring_RL import PPO
from flow.core.params import SumoParams
import pandas as pd
import  os
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
    else:
        print(path+'exist')

num_runs=150
env = Experiment().env
rl_actions=None
convert_to_csv=True
model_path="./model/model.ckpt"
env.sim_params.emission_path='./emission/'
sim_params = SumoParams(sim_step=0.1, render=False, emission_path='./emission/')
num_steps = env.env_params.horizon
train_test=1 ##define train(1) or test(2)
# raise an error if convert_to_csv is set to True but no emission
# file will be generated, to avoid getting an error at the end of the
# simulation
if convert_to_csv and env.sim_params.emission_path is None:
	raise ValueError(
		'The experiment was run with convert_to_csv set '
		'to True, but no emission file will be generated. If you wish '
		'to generate an emission file, you should set the parameter '
		'emission_path in the simulation parameters (SumoParams or '
		'AimsunParams) to the path of the folder where emissions '
		'output should be generated. If you do not wish to generate '
		'emissions, set the convert_to_csv parameter to False.')




info_dict = {}
if rl_actions is None:
	def rl_actions(*_):
		return None

sess = tf.Session()
agent = PPO()

# agent.buildActorNetwork()
# agent.buildCriticNetwork()
# agent.setLearn()
# agent.setUpdate()

init_op = tf.group(tf.global_variables_initializer())
sess.run(init_op)
saver = tf.train.Saver()

rets = []
mean_rets = []
ret_lists = []
vels = []
mean_vels = []
std_vels = []
outflows = []
t = time.time()
times = []
vehicle_times = []
ploss=0
qloss=0
reg_loss=0


if train_test==1: ##train the model
    for i in range(num_runs):
        vel = np.zeros(num_steps)
        # logging.info("Iter #" + str(i))
        print('episode is:',i)
        ret = 0
        ret_list = []
        state = env.reset()
        aset = []
        for j in range(num_steps):

            # manager actions

            # one controllers
            a = agent.choose_action(state)
            aset.append(a)
            next_state, reward, done, _ = env.step(a)
            if j%300==0:
                print(state)
                print(a)

            ret += reward
            ret_list.append(reward)

            # if agent.num_experience>20000:
            #     ploss,qloss, reg_loss,Q = agent.learn(batch_size=128)
            #
            # if done:
            #     agent.remember( state, a, reward, next_state,1. )
            #     break
            # agent.remember( state, a, reward, next_state,0. )
            # state = next_state[:]
        # agent.var*=0.9
        rets.append(ret)
        save_path = saver.save(sess, model_path)
        print("Round {0}, return: {1}".format(i, ret))
        print('max acation:%g  min acation:%g  var action:%g meanabs action:%g'%(np.max(aset),np.min(aset),np.std(aset),np.mean(np.abs(aset))))

        print('ploss %g q_loss %g'%(ploss,qloss))

    print("Average, std return:    {}, {}".format(
        np.mean(rets), np.std(rets)))
    print("Average, std speed:     {}, {}".format(
        np.mean(mean_vels), np.std(mean_vels)))
    print("Total time:            ", time.time() - t)
    print("steps/second:          ", np.mean(times))
    print("vehicles.steps/second: ", np.mean(vehicle_times))

if train_test==2:
    saver.restore(sess, model_path)
    vel = np.zeros(num_steps)
    ret = 0
    ret_list = []
    state = env.reset()
    aset = []
    agent.var = 0
    for j in range(num_steps):
        a = agent.choose_action(state)
        aset.append(a)
        next_state, reward, done, _ = env.step(a)

        print(state)
        print(a)

        ret += reward
        ret_list.append(reward)

        if done:
            print(num_steps)
            break
        # agent.remember( state, a, reward, next_state,0. )
        state = next_state[:]
    print('reward %g'%ret)
    print('max acation:%g  min acation:%g  var action:%g meanabs action:%g'%(np.max(aset),np.min(aset),np.std(aset),np.mean(np.abs(aset))))


env.terminate()

#return info_dict