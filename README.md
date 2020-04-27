
## Code structure:

Our main implementation of this model is in dgn_xxx (xxx represents for different scenario, e.g. ring network, figure eight network, or minicity network)

In each folder, you can find e.g. xxx_main-DGN.py this is the file to run; In DGN.py, we define the main network structure and training process, in xxx_Env.py we define the simulation environment. 

Also, we implemented DDPG, multi-agent version of PPO in three different scenarios.

In explore folder, we saved some models that we tested, include different model structures, number of agents, etc.

