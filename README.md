Accompanying code for [Actor-Critic Method for High Dimensional Static Hamilton--Jacobi--Bellman Partial Differential Equations based on Neural Networks](https://arxiv.org/abs/2102.11379), using actor-critic method to solve the HJB equations. The code is written on Tensorflow 2.0.

Run the following command to solve the HJB equation directly:
```
python main.py --config_path=configs/lqr_d5.json
```
**Names of config files**:
"Lqr" denotes the linear quadratic regulator;
"Vdp" denotes the stochastic Van Der Pol oscilator;
"Ekn" denotes the diffusive Eikonal equation.

| Experiments in the paper                                     | Config names                                                 |
|--------------------------------------------------------------|--------------------------------------------------------------|
| Linear quadratic regulator (Figure 2)                        | lqr_d5.json, lqr_d10.json, lqr_d20.json                      |
| Stochastic Van Der Pol oscilator (Figure 3)                  | vdp_d5.json, vdp_d10.json, vdp_d20.json                      |
| Diffusive Eikonal equation (Figure 4)                        | ekn_d5.json, ekn_d10.json, ekn_d20.json                      |

Choices of training configs include:
sample: normal, bounded;
scheme: naive, adapted;
TD: TD1, TD2;
train: actor-critic, actor, critic.
You can change them and the parameters by modifying the configs.
