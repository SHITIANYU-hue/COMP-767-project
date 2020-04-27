This is the source code of COMP 767 group project of Tianyu Shi & Jiawei Wang.


## Code structure:

Our main implementation of this model is in dgn_xxx (xxx represents for different scenario, e.g. ring network, figure eight network, or minicity network)

In each folder, you can find e.g. xxx_main-DGN.py this is the file to run; In DGN.py, we define the main network structure and training process, in xxx_Env.py we define the simulation environment. 

Also, we implemented DDPG, multi-agent version of PPO in three different scenarios.

In explore folder, we saved some models that we tested, include different model structures, number of agents, etc.

## How to reproduce:

Some parameters in the experiment settings:


| Models config    | units of encoder layer    | activation function | clip ratio   | discount factor | Optimizer | softupdate parameter | learning rate|max returns|
| :----------------: | :---------: | :-------------: | :-----: | :---------: | :-------------: | :------------: | :-----:|:---:|
|   DGN  a | (128 , 128) | ReLU         | 0.3 | 0.9      | adam         | 0.01    |（1e-4 , 1e-4)||
| DGN  b  | (128 , 128) | ReLU        | 0.3 | 0.9    | adam      | 0.01       |（1e-4 , 1e-4)||
| DGN  c  |(128 , 128) | ReLU        | 0.3| 0.9     | adam       | 0.01   |（1e-4 , 1e-4)（1e-4 , 1e-4)||
| DGN  d  | (128 , 128) | ReLU        | 0.3 | 0.9    | adam      |0.01    | （1e-4 , 1e-4)||
| DGN  e  | (128 , 128) | ReLU         | 0.3 | 0.9      | adam       | 0.01    | （1e-4 , 1e-4)||
| DGN  f  | (128 , 128)| ReLU         | 0.3 | 0.9      | adam       | 0.01    |（1e-4 , 1e-4)||
| DGN  g  | (128 , 128) | ReLU         | 0.3 | 0.9     | adam        | 0.01    |（1e-4 , 1e-4)||






## Contact:
Tianyu Shi(tianyu.shi3@mail.mcgill.ca) 

