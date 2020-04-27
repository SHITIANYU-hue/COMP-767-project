# Import all of the necessary pieces of Flow to run the experiments
from flow.core.params import SumoParams, EnvParams, NetParams, InitialConfig, \
    InFlows, SumoLaneChangeParams, SumoCarFollowingParams
from flow.core.params import VehicleParams
from flow.core.params import TrafficLightParams

from flow.controllers import SimLaneChangeController, ContinuousRouter
from flow.core.experiment import Experiment
from city_Env import para_produce_rl,Experiment
import logging
import pandas as pd
import datetime
import numpy as np
import time
import os
import tensorflow as tf
from city_RL import DDPGAgent
from flow.core.params import SumoParams
### define some parameters
#import pandas as pd
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
    else:
        print(path+'exist')


exp_tag="ddpg_city"
mkdir('{}_results'.format(exp_tag))
flow_params=para_produce_rl(NUM_AUTOMATED=14)
env = Experiment(flow_params=flow_params).env
num_runs=100
rl_actions=None
convert_to_csv=True
model_path="./model/{0}_model.ckpt".format(exp_tag)
env.sim_params.emission_path='./{}_emission/'.format(exp_tag)
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
agent = DDPGAgent(state_dim = 32, action_dim = 1, sess=sess)

agent.buildActorNetwork()
agent.buildCriticNetwork()
agent.setLearn()
agent.setUpdate()

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
results=[]


if train_test==1: ##train the model
    for i in range(num_runs):
        vel = np.zeros(num_steps)
        # logging.info("Iter #" + str(i))
        print('episode is:',i)
        ret = 0
        ret_list = []
        state = env.reset()
        state=state[0:32]
        aset = []
        for j in range(num_steps):

            # manager actions

            # one controllers
            a = agent.choose_action([state])
            aset.append(a)
            next_state, reward, done, _ = env.step(a)
            next_state=next_state[0:32]
            # if j%300==0:
            #     print(state)
            #     print(a)

            ret += reward
            ret_list.append(reward)

            if agent.num_experience>2000:
                ploss,qloss, reg_loss,Q = agent.learn(batch_size=32)

            if done:
                agent.remember(state, a, reward, next_state,1. )
                break
            agent.remember( state, a, reward, next_state,0. )
            state = next_state[:]
        agent.var*=0.9
        rets.append(ret)
        ret_max=max(rets)
        save_path = saver.save(sess, model_path)
        print("Round {0}, return: {1},max_return:{2}".format(i, ret,ret_max))
        print('max acceleration:%g  min acceleration:%g  var acceleration:%g meanabs acceleration:%g'%(np.max(aset),np.min(aset),np.std(aset),np.mean(np.abs(aset))))
        print('ploss %g q_loss %g'%(ploss,qloss))
        result=[i,ret,ret_max,np.max(aset),np.min(aset),np.std(aset),np.mean(np.abs(aset)),ploss,qloss]
        results.append(result)
        name=['Round','Return','max_Return','max acceleration','min acceleration','var acceleration','meanabs acceleration','ploss','qloss']
        test=pd.DataFrame(columns=name,data=results)
        test.to_csv('{}_results/{}_log_summary.csv'.format(exp_tag,exp_tag),encoding='gbk')
        np.save('{}_results/{}_rewards'.format(exp_tag,exp_tag),ret_list)

if train_test==2:
    saver.restore(sess, model_path)
    vel = np.zeros(num_steps)
    ret = 0
    ret_list = []
    state = env.reset()
    aset = []
    agent.var = 0
    for j in range(num_steps):
        a = agent.choose_action([state])
        aset.append(a)
        next_state, reward, done, _ = env.step(a)

        print(state)
        print(a)

        ret += reward
        ret_list.append(reward)

        if done:
            print(num_steps)
            break
        agent.remember( state, a, reward, next_state,0. )
        state = next_state[:]
    print('reward %g'%ret)
    print('max acation:%g  min acation:%g  var action:%g meanabs action:%g'%(np.max(aset),np.min(aset),np.std(aset),np.mean(np.abs(aset))))


env.terminate()

#return info_dict