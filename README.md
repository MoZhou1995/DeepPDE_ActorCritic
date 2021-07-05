Accompanying code for [Actor-Critic Method for High Dimensional Static Hamilton--Jacobi--Bellman Partial Differential Equations based on Neural Networks](https://arxiv.org/abs/2102.11379), using actor-critic method to solve the HJB equations. The code is written on Tensorflow 2.0.

Run the following command to solve the HJB equation directly:
```
python main.py --config_path=configs/lqr_d5.json
```
**Names of config files**:
"lqr" denotes the linear quadratic regulator;
"vdp" denotes the stochastic Van Der Pol oscilator;
"ekn" denotes the diffusive Eikonal equation;
"lqr_var" denotes the linear quadratic regulator with a non-constant diffusion coefficient.

| Experiments in the paper                                     | Config names                                                 |
|--------------------------------------------------------------|--------------------------------------------------------------|
| Linear quadratic regulator (Figure 2)                        | lqr_d5.json, lqr_d10.json, lqr_d20.json                      |
| Stochastic Van Der Pol oscilator (Figure 3)                  | vdp_d4.json, vdp_d10.json, vdp_d20.json                      |
| Diffusive Eikonal equation (Figure 4)                        | ekn_d5.json, ekn_d10.json, ekn_d20.json                      |
| Linear quadratic regulator with non-constant diffusion       | lqr_var_d5.json, lqr_var_d10.json, lqr_var_d20.json          |

**Fileds in config files**
"sample": "normal" means sampling Brownian increments with normal distribution and "bounded" means bounded sample.
"scheme": "naive" means using the naive scheme in the paper and "adaptive" means using the stepsize adaptive scheme.
"TD": "TD1" means using the variance-reduced least square temporal difference (VR-LSTD) and "TD2" means using least square temporal difference (LSTD).
"train": "actor-critic" means training both the value function and the control, "actor" means only training the control (given the correct value function), and "critic" means training only the value function (given the correct control).
