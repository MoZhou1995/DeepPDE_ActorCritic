# DeepPDE_ActorCritic
This code is the numerical example for https://arxiv.org/abs/2102.11379, using actor-critic method to solve the HJB equations.
The code is based on Tensorflow 2.0.

Run the main.py file for the code. Before running, choose the proper config you want to run by modifying line 20. All the configs are in the folder names "configs".

Choices of training configs include:
sample: normal, bounded
scheme: naive, adapted
TD: TD1, TD2
train: actor-critic, actor, critic
You can change them and the parameters by modifying the configs.
