# Import all of the necessary pieces of Flow to run the experiments
from flow.core.params import SumoParams, EnvParams, NetParams, InitialConfig, \
    InFlows, SumoLaneChangeParams, SumoCarFollowingParams
from flow.core.params import VehicleParams
from flow.core.params import TrafficLightParams
import pandas as pd
from flow.controllers import SimLaneChangeController, ContinuousRouter
from flow.core.experiment import Experiment
from DGN_Env import para_produce_rl,Experiment
import logging

import datetime
import numpy as np
import time
import os
import tensorflow as tf
from DGN import DGN
from flow.core.params import SumoParams
### define some parameters
import pandas as pd
import  os
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
    else:
        print(path+'exist')

## define some environment parameters
exp_tag="dgn_minicity"
mkdir('{}_results'.format(exp_tag))
agent_num=12
neighbors=3
train_test=1 ##define train(1) or test(2)
num_runs=100

## build up settings
flow_params=para_produce_rl(NUM_AUTOMATED=agent_num)
env = Experiment(flow_params=flow_params).env
rl_actions=None
convert_to_csv=True
model_path="./model/{0}_model.ckpt".format(exp_tag)
env.sim_params.emission_path='./{}_emission/'.format(exp_tag)
sim_params = SumoParams(sim_step=0.1, render=False, emission_path='./{0}_emission/'.format(exp_tag))
num_steps = env.env_params.horizon


sess = tf.Session()
agent = DGN(state_dim = 6, action_dim = 1, neighbors=neighbors,agent_num=agent_num,sess=sess)
agent.buildCriticNetwork()
agent.buildActorNetwork()
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

## save simulation videos
def render(render_mode='sumo_gui'):
    from flow.core.params import SimParams as sim_params
    sim_params.render=True
    save_render=True
    setattr(sim_params, 'num_clients', 1)
    # pick your rendering mode
    if render_mode == 'sumo_web3d':
        sim_params.num_clients = 2
        sim_params.render = False
    elif render_mode == 'drgb':
        sim_params.render = 'drgb'
        sim_params.pxpm = 4
    elif render_mode == 'sumo_gui':
        sim_params.render = False  # will be set to True below
    elif render_mode == 'no_render':
        sim_params.render = False
    if save_render:
        if render_mode != 'sumo_gui':
            sim_params.render = 'drgb'
            sim_params.pxpm = 4
        sim_params.save_render = True

### todo how to define agent's relationship
# def Adjacency( env ,neighbors=2):
#     adj = []
#     vels=np.array([env.k.vehicle.get_speed(veh_id) for veh_id in env.k.vehicle.get_rl_ids() ])
#     orders = np.argsort(vels)
#     for rl_id1 in env.k.vehicle.get_rl_ids():
#         l = np.zeros([neighbors,len(env.k.vehicle.get_rl_ids())])
#         j=0
#         for k in range(neighbors):
#             # modify this condition to define the adjacency matrix
#             l[k,orders[k]]=1

#         adj.append(l)
#     return adj
def Adjacency(env ,neighbors=2):
    adj = []
    
    x_pos = np.array([env.k.vehicle.get_x_by_id(veh_id) for veh_id in env.k.vehicle.get_rl_ids() ])
    headways = np.zeros([len(env.k.vehicle.get_rl_ids()),len(env.k.vehicle.get_rl_ids())])
    for d in range(len(env.k.vehicle.get_rl_ids())):
        headways[d,:] = abs(x_pos-x_pos[d])
    
    orders = np.argsort(headways)
    for rl_id1 in env.k.vehicle.get_rl_ids():
        l = np.zeros([neighbors,len(env.k.vehicle.get_rl_ids())])
        j=0
        for k in range(neighbors):
            # modify this condition to define the adjacency matrix
            l[k,orders[k]]=1

        adj.append(l)
    return adj


 
if train_test==1: ##train the model
    for i in range(num_runs):
        vel = np.zeros(num_steps)
        # logging.info("Iter #" + str(i))
        print('episode is:',i)
        ret = 0
        ret_list = []
        state = env.reset()

        aset = []
        vec = np.zeros((1, neighbors))
        vec[0][0] = 1
        for j in range(num_steps):
            # manager actions
            state_ = np.array(list(state.values())).reshape(1,-1).tolist()

            adj = Adjacency(env ,neighbors=neighbors)
 
            action_dict = {}
            
            a=sess.run(agent.action, feed_dict={agent.adj: [adj], agent.state_holder: state_,
                                                        agent.vecholder: np.asarray([vec])})
         
            k=0
            for key,value in state.items():
                action_dict[key]=a[k]
                k+=1
            aset.append(a)

            next_state, reward, done, _ = env.step(action_dict)
            next_state_ = np.array(list(next_state.values())).reshape(1,-1).tolist()
            rewards = list(reward.values())
          ## calculate individual reward
            # for k in range(len(rewards)):
            #     print('agent',k,'reward is:',rewards[k])

            ret += np.average(rewards) ##here we consider the rewards of each agent
            ret_list.append(rewards)

            agent.remember(state_, a, rewards, next_state_,0.,adj )
            if agent.num_experience>200: ##could change to 200
                ploss,qloss = agent.learn(batch_size=32)
            
            vels= np.array([env.k.vehicle.get_speed(veh_id) for veh_id in env.k.vehicle.get_rl_ids()])

        agent.var*=0.9
        rets.append(ret)
        ret_max=max(rets)
        mean_vel=np.mean(vels)
        mean_vels.append(mean_vel)
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
    print("Average, std return:    {}, {}".format(
        np.mean(rets), np.std(rets)))
    print("Average, std speed:     {}, {}".format(
         np.mean(mean_vels), np.std(mean_vels)))
    print("Total time:            ", time.time() - t)
    # print("steps/second:          ", np.mean(times))
    # print("vehicles.steps/second: ", np.mean(vehicle_times))

if train_test==2:
    saver.restore(sess, model_path)
    vel = np.zeros(num_steps)
    ret = 0
    ret_list = []
    state = env.reset()
    aset = []
    agent.var = 0
    vel = np.zeros(num_steps)
    # logging.info("Iter #" + str(i))
    ret = 0
    ret_list = []
    state = env.reset()

    aset = []
    vec = np.zeros((1, neighbors))
    vec[0][0] = 1
    for j in range(num_steps):
        # manager actions
        state_ = np.array(list(state.values())).reshape(1,-1).tolist()
         

        adj = Adjacency(env ,neighbors=neighbors)

        action_dict = {}
        
        a=sess.run(agent.action, feed_dict={agent.adj: [adj], agent.state_holder: state_,
                                                    agent.vecholder: np.asarray([vec])})
     
        k=0
        for key,value in state.items():
            action_dict[key]=a[k]
            k+=1

        aset.append(a)

        next_state, reward, done, _ = env.step(action_dict)
        next_state_ = np.array(list(next_state.values())).reshape(1,-1).tolist()
        rewards = list(reward.values())
     
        ret += np.sum(rewards)
        ret_list.append(rewards)
   
    print('reward %g'%ret)
    print('max acceleration:%g  min acceleration:%g  var acceleration:%g meanabs acceleration:%g'%(np.max(aset),np.min(aset),np.std(aset),np.mean(np.abs(aset))))


env.terminate()

