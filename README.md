This is the source code of COMP 767 group project of Tianyu Shi & Jiawei Wang.


## Code structure:

Our main implementation of this model is in dgn_xxx (xxx represents for different scenario, e.g. ring network, figure eight network, or minicity network)

In each folder, you can find e.g. xxx_main-DGN.py this is the file to run; In DGN.py, we define the main network structure and training process, in xxx_Env.py we define the simulation environment. 

Also, we implemented DDPG, multi-agent version of PPO in three different scenarios.

In explore folder, we saved some models that we tested, include different model structures, number of agents, etc.

## How to reproduce:

Some parameters in the experiment settings:

\begin{table}
\centering
\caption{Hyper-parameter tuning for DGN-PPO model}
\begin{minipage}{1\textwidth}
\begin{tabular}{c|cccccc}
\hline
  Model config. & units of encoder layer\footnote{It represents different units in each layers, e.g. (128,64,128) stands for 128, 64, 128 units in layer 1, 2,3 respectively.}  & activation function & $\epsilon$ & learning rate\footnote{The units represent the learning rates for actor and the critic network respectively.} & return  \\
\hline
 a & (128, 128)   & ReLU &0.3& (1e-4,1e-4) & \textbf{2982.97} \\
\hline
b & (512, 128)  & ReLU &0.3& (1e-4,1e-4) & 2956.09 \\
c & (128,64,128)  & ReLU &0.3 & (1e-4,1e-4) & 2900.04 \\
% d & (128, 128) & ReLU &0.3 & (1e-4,1e-4) & 2972.35 \\
\hline
d & (128, 128)   &ReLU&0.15 & (1e-4,1e-4)& 2972.85 \\
e & (128, 128)  & eLU&0.3 & (1e-4,1e-4) & 2900.04 \\
f & (128, 128)  & ReLU &0.3 & (2.5e-4,1e-3) & 2898.77 \\
\hline
\end{tabular} 
\end{minipage}
\label{architectures}
\end{table}


## Contact:
Tianyu Shi(tianyu.shi3@mail.mcgill.ca) 

