This is the source code of COMP 767 group project of Tianyu Shi & Jiawei Wang.


## Code structure:

Our main implementation of this model is in dgn_xxx (xxx represents for different scenario, e.g. ring network, figure eight network, or minicity network)

In each folder, you can find e.g. xxx_main-DGN.py this is the file to run; In DGN.py, we define the main network structure and training process, in xxx_Env.py we define the simulation environment. 

Also, we implemented DDPG, multi-agent version of PPO in three different scenarios.

In explore folder, we saved some models that we tested, include different model structures, number of agents, etc.

## How to reproduce:

Some parameters in the experiment settings:


| Models config    | LR    | Decision Tree | SVM   | Ada Boost | Random forest | MLP(10, 256) | MLP(30, 1024) | XG Boost | LSTM   |
| ------------- | ----- | ------------- | ----- | --------- | ------------- | ------------ | ------------- | -------- | ------ |
| DGN config a | 0.776 | 0.47          | 0.804 | 0.5       | 0.688         | 0.7058       | 0.7408        | 0.6164   | 0.6761 |
| DGN config b  | 0.883 | 0.702         | 0.877 | 0.83      | 0.8367        | 0.8757       | 0.8801        | 0.7398   | 0.8916 |
| DGN config c  | 0.883 | 0.702         | 0.877 | 0.83      | 0.8367        | 0.8757       | 0.8801        | 0.7398   | 0.8916 |
| DGN config d  | 0.883 | 0.702         | 0.877 | 0.83      | 0.8367        | 0.8757       | 0.8801        | 0.7398   | 0.8916 |
| DGN config e  | 0.883 | 0.702         | 0.877 | 0.83      | 0.8367        | 0.8757       | 0.8801        | 0.7398   | 0.8916 |
| DGN config f  | 0.883 | 0.702         | 0.877 | 0.83      | 0.8367        | 0.8757       | 0.8801        | 0.7398   | 0.8916 |
| DGN config g  | 0.883 | 0.702         | 0.877 | 0.83      | 0.8367        | 0.8757       | 0.8801        | 0.7398   | 0.8916 |



## Contact:
Tianyu Shi(tianyu.shi3@mail.mcgill.ca) 

