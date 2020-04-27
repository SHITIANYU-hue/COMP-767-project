# Import all of the necessary pieces of Flow to run the experiments
from flow.core.params import SumoParams, EnvParams, NetParams, InitialConfig, \
    InFlows, SumoLaneChangeParams, SumoCarFollowingParams
from flow.core.params import VehicleParams
from flow.core.params import TrafficLightParams

 
from flow.controllers import RLController, IDMController, ContinuousRouter
from flow.envs.ring.wave_attenuation import WaveAttenuationEnv, WaveAttenuationPOEnv # Env for RL
from flow.envs.multiagent import MultiAgentWaveAttenuationPOEnv
from flow.core.experiment import Experiment
from flow.networks.ring import RingNetwork
from flow.networks.ring import RingNetwork, ADDITIONAL_NET_PARAMS
import logging

import datetime
import numpy as np
import time
import os

from flow.core.util import emission_to_csv
from flow.utils.registry import make_create_env
    
# define parameters


from flow.controllers.rlcontroller import RLController
from flow.controllers.lane_change_controllers import SimLaneChangeController
from flow.controllers.routing_controllers import ContinuousRouter
from copy import deepcopy
from gym.spaces.box import Box
from flow.core import rewards
from flow.envs.base import Env

#flow/examples/exp_configs/rl/singleagent/

def para_produce_rl(HORIZON = 3000,NUM_AUTOMATED = 4):
	
    # time horizon of a single rollout
    HORIZON = 3000
    # number of rollouts per training iteration
    N_ROLLOUTS = 20
    # number of parallel workers
    N_CPUS = 2
    # number of automated vehicles. Must be less than or equal to 22.
    NUM_AUTOMATED = NUM_AUTOMATED


    # We evenly distribute the automated vehicles in the network.
    num_human = 22 - NUM_AUTOMATED
    humans_remaining = num_human

    vehicles = VehicleParams()
    for i in range(NUM_AUTOMATED):
        # Add one automated vehicle.
        vehicles.add(
            veh_id="rl_{}".format(i),
            acceleration_controller=(RLController, {}),
            routing_controller=(ContinuousRouter, {}),
            num_vehicles=1)

        # Add a fraction of the remaining human vehicles.
        vehicles_to_add = round(humans_remaining / (NUM_AUTOMATED - i))
        humans_remaining -= vehicles_to_add
        vehicles.add(
            veh_id="human_{}".format(i),
            acceleration_controller=(IDMController, {
                "noise": 0.2
            }),
            car_following_params=SumoCarFollowingParams(
                min_gap=0
            ),
            routing_controller=(ContinuousRouter, {}),
            num_vehicles=vehicles_to_add)


        flow_params = dict(
        # name of the experiment
        exp_tag="multiagent_ring",

        # name of the flow environment the experiment is running on
        env_name=MultiAgentWaveAttenuationPOEnv,

        # name of the network class the experiment is running on
        network=RingNetwork,

        # simulator that is used by the experiment
        simulator='traci',

        # sumo-related parameters (see flow.core.params.SumoParams)
        sim=SumoParams(
            sim_step=0.1,
            render=False,
            restart_instance=False
        ),

        # environment related parameters (see flow.core.params.EnvParams)
        env=EnvParams(
            horizon=HORIZON,
            warmup_steps=750,
            clip_actions=False,
            additional_params={
                "max_accel": 1,
                "max_decel": 1,
                "ring_length": [220, 270],
            },
        ),

        # network-related parameters (see flow.core.params.NetParams and the
        # network's documentation or ADDITIONAL_NET_PARAMS component)
        net=NetParams(
            additional_params={
                "length": 260,
                "lanes": 1,
                "speed_limit": 30,
                "resolution": 40,
            }, ),

        # vehicles to be placed in the network at the start of a rollout (see
        # flow.core.params.VehicleParams)
        veh=vehicles,

        # parameters specifying the positioning of vehicles upon initialization/
        # reset (see flow.core.params.InitialConfig)
        initial=InitialConfig())

    flow_params['env'].horizon = HORIZON
    return flow_params

#flow_params = para_produce(flow_rate=1000, scaling=1, disable_tb=True, disable_ramp_meter=True)
flow_params = para_produce_rl()
	
class Experiment:

    def __init__(self, flow_params=flow_params):
        """Instantiate Experiment."""
        # Get the env name and a creator for the environment.
        create_env, _ = make_create_env(flow_params)

        # Create the environment.
        self.env = create_env()

        logging.info(" Starting experiment {} at {}".format(
            self.env.network.name, str(datetime.datetime.utcnow())))

        logging.info("Initializing environment.")	
