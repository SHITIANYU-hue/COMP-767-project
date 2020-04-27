This is the source code of COMP 767 group project of Tianyu Shi & Jiawei Wang.


## Code structure:

Our main implementation of this model is in dgn_xxx (xxx represents for different scenario, e.g. ring network, figure eight network, or minicity network)

In each folder, you can find e.g. xxx_main-DGN.py this is the file to run; In DGN.py, we define the main network structure and training process, in xxx_Env.py we define the simulation environment. 

Also, we implemented DDPG, multi-agent version of PPO in three different scenarios.

In explore folder, we saved some models that we tested, include different model structures, number of agents, etc.

## How to reproduce:

Some parameters in the experiment settings:


| Models config    | units of encoder layer    | activation function | clip ratio   | discount factor | Optimizer | softupdate parameter | 
| ---------------- | --------- | ------------- | ----- | --------- | ------------- | ------------ | 
|   DGN  a | 0.776 | 0.47          | 0.804 | 0.5       | 0.688         | 0.7058     |
| DGN  b  | 0.883 | 0.702         | 0.877 | 0.83      | 0.8367        | 0.8757       |
| DGN  c  | 0.883 | 0.702         | 0.877 | 0.83      | 0.8367        | 0.8757    |
| DGN  d  | 0.883 | 0.702         | 0.877 | 0.83      | 0.8367        | 0.8757     |
| DGN  e  | 0.883 | 0.702         | 0.877 | 0.83      | 0.8367        | 0.8757     |
| DGN  f  | 0.883 | 0.702         | 0.877 | 0.83      | 0.8367        | 0.8757    |
| DGN  g  | 0.883 | 0.702         | 0.877 | 0.83      | 0.8367        | 0.8757    |






## Contact:
Tianyu Shi(tianyu.shi3@mail.mcgill.ca) 

