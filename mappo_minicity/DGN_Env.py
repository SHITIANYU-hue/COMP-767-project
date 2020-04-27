# Import all of the necessary pieces of Flow to run the experiments
## i modified this file to make it adapt to minicity environment
from flow.core.params import SumoParams, EnvParams, NetParams, InitialConfig, \
    InFlows, SumoLaneChangeParams, SumoCarFollowingParams
from flow.core.params import VehicleParams
from flow.core.params import TrafficLightParams

 
from flow.controllers import RLController, IDMController, ContinuousRouter
from flow.controllers.routing_controllers import MinicityRouter
from flow.envs.ring.wave_attenuation import WaveAttenuationEnv, WaveAttenuationPOEnv # Env for RL
from flow.envs.multiagent import MultiAgentWaveAttenuationPOEnv
from flow.core.experiment import Experiment
import logging
from flow.networks import MiniCityNetwork
from accel import ADDITIONAL_ENV_PARAMS, MultiAgentAccelPOEnv

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

def para_produce_rl(HORIZON = 750,NUM_AUTOMATED = 4):
	
    # time horizon of a single rollout
    HORIZON = 750
    # number of rollouts per training iteration
    N_ROLLOUTS = 20
    # number of parallel workers
    N_CPUS = 4
    # number of automated vehicles. Must be less than or equal to 22.
    NUM_AUTOMATED = NUM_AUTOMATED


    # We evenly distribute the automated vehicles in the network.
    num_human = 40 - NUM_AUTOMATED
    humans_remaining = num_human

    vehicles = VehicleParams()
    for i in range(NUM_AUTOMATED):
        # Add one automated vehicle.
        vehicles.add(
            veh_id="rl_{}".format(i),
            acceleration_controller=(RLController, {}),
            routing_controller=(MinicityRouter, {}),
            initial_speed=0,
            car_following_params=SumoCarFollowingParams(
                speed_mode="obey_safe_speed",
            ),
            num_vehicles=1)
        # Add a fraction of the remaining human vehicles.
        vehicles_to_add = round(humans_remaining / (NUM_AUTOMATED - i))
        humans_remaining -= vehicles_to_add
        vehicles.add(
            veh_id="human_{}".format(i),
            acceleration_controller=(IDMController, {
                "noise": 0.2
            }),
            routing_controller=(MinicityRouter, {}),
            car_following_params=SumoCarFollowingParams(
                speed_mode=1
            ),
            lane_change_params=SumoLaneChangeParams(
                lane_change_mode="no_lat_collide"
            ),
            initial_speed=0,
            num_vehicles=vehicles_to_add)


        flow_params = dict(
        # name of the experiment
        exp_tag="minicity",

        # name of the flow environment the experiment is running on
        env_name=MultiAgentAccelPOEnv,

        # name of the network class the experiment is running on
        network=MiniCityNetwork,

        # simulator that is used by the experiment
        simulator='traci',

        # sumo-related parameters (see flow.core.params.SumoParams)
        sim=SumoParams(
            sim_step=0.25,
            render=False,
            save_render=True,
            sight_radius=5,
            pxpm=3,
            show_radius=True,
            restart_instance=True
        ),

        # environment related parameters (see flow.core.params.EnvParams)
        env=EnvParams(
            horizon=750,
            additional_params=ADDITIONAL_ENV_PARAMS
        ),
        # network-related parameters (see flow.core.params.NetParams and the
        # network's documentation or ADDITIONAL_NET_PARAMS component)
        # net=NetParams(
        #     additional_params={
        #         "length": 260,
        #         "lanes": 1,
        #         "speed_limit": 30,
        #         "resolution": 40,
        #     }, ),
        net=NetParams(),

        # vehicles to be placed in the network at the start of a rollout (see
        # flow.core.params.VehicleParams)
        veh=vehicles,

        # parameters specifying the positioning of vehicles upon initialization/
        # reset (see flow.core.params.InitialConfig)
        initial=InitialConfig(
        spacing="uniform",
        min_gap=20,
        ),)

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
