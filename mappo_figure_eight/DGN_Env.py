# Import all of the necessary pieces of Flow to run the experiments
from flow.core.params import SumoParams, EnvParams, NetParams, InitialConfig, \
    InFlows, SumoLaneChangeParams, SumoCarFollowingParams
from flow.core.params import VehicleParams
from flow.core.params import TrafficLightParams
from flow.networks import FigureEightNetwork
from ray.rllib.agents.ppo.ppo_policy import PPOTFPolicy
from ray.tune.registry import register_env

from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.params import VehicleParams, SumoCarFollowingParams
from flow.controllers import IDMController, ContinuousRouter, RLController
from flow.networks.figure_eight import ADDITIONAL_NET_PARAMS
from flow.envs.multiagent import MultiAgentAccelPOEnv
from flow.networks import FigureEightNetwork
from flow.utils.registry import make_create_env 
import logging
import datetime
import numpy as np
import time
import os
from flow.core.util import emission_to_csv
from flow.utils.registry import make_create_env
from flow.controllers.rlcontroller import RLController
from flow.controllers.lane_change_controllers import SimLaneChangeController
from flow.controllers.routing_controllers import ContinuousRouter
from copy import deepcopy
from gym.spaces.box import Box
from flow.core import rewards
from flow.envs.base import Env 
# define parameters


#flow/examples/exp_configs/rl/singleagent/

def para_produce_rl(HORIZON = 3000,NUM_AUTOMATED = 7):
	
    # time horizon of a single rollout
    HORIZON = 1500
    # number of rollouts per training iteration
    N_ROLLOUTS = 20
    # number of parallel workers
    N_CPUS = 2
    # number of automated vehicles. Must be less than or equal to 22.
    NUM_AUTOMATED = NUM_AUTOMATED
    assert NUM_AUTOMATED in [1, 2, 7, 14], \
    "num_automated must be one of [1, 2, 7 14]"
    # desired velocity for all vehicles in the network, in m/s
    TARGET_VELOCITY = 20
    # maximum acceleration for autonomous vehicles, in m/s^2
    MAX_ACCEL = 3
    # maximum deceleration for autonomous vehicles, in m/s^2
    MAX_DECEL = 3

    # We evenly distribute the automated vehicles in the network.
    num_human = 14 - NUM_AUTOMATED
    human_per_automated = int(num_human / NUM_AUTOMATED)

    vehicles = VehicleParams()
    for i in range(NUM_AUTOMATED):
        # Add one automated vehicle.
        vehicles.add(
            veh_id="rl_{}".format(i),
            acceleration_controller=(RLController, {}),
            routing_controller=(ContinuousRouter, {}),
            car_following_params=SumoCarFollowingParams(
                speed_mode="obey_safe_speed",
                accel=MAX_ACCEL,
                decel=MAX_DECEL,
            ),
            num_vehicles=1)

        # Add a fraction of the human driven vehicles.
        vehicles.add(
            veh_id="human_{}".format(i),
            acceleration_controller=(IDMController, {
                "noise": 0.2
            }),
            car_following_params=SumoCarFollowingParams(
                speed_mode="obey_safe_speed",
                decel=1.5
            ),
            routing_controller=(ContinuousRouter, {}),
            num_vehicles=human_per_automated)


        flow_params = dict(
        # name of the experiment
        exp_tag="multiagent_figure_eight",

        # name of the flow environment the experiment is running on
        env_name=MultiAgentAccelPOEnv,

        # name of the network class the experiment is running on
        network=FigureEightNetwork,

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
            additional_params={
                'target_velocity': TARGET_VELOCITY,
                'max_accel': MAX_ACCEL,
                'max_decel': MAX_DECEL,
                'sort_vehicles':False
            },
        ),

        # network-related parameters (see flow.core.params.NetParams and the
        # network's documentation or ADDITIONAL_NET_PARAMS component)
        net=NetParams(
            additional_params=ADDITIONAL_NET_PARAMS.copy(),
        ),

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
