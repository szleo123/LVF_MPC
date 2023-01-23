# LVF_MPC

Introduction
-----------------------------------------------------
This project proposed a method that builds a Linear Variational State Space Filter that, given the picture of a linear system, learns a latent space that mimics the state space of the dynamical system. It then applies MPC to achieve control objectives in the learned latent space and then maps it back through the decoder to generate images showing the states of the dynamical system. We show the effectiveness of our pipeline on a pendulum system and shows directions for future improvements like combining it with more advance MPC techniques like lumped System Level Synthesis Robust MPC, etc.

How to run the code? 
-----------------------------------------------------

- Go to ./variational_state_space_models 
- Use ./docker/build.sh to build docker container
- Use ./docker/run.sh to enter the container 
- Install cvxpy, pytope, polytope packages insider the container using pip3 
- Run lsvae/train_lsvae.py to start training 
- Go to ./variational_state_space_models/lsvae
- Use python3 MPC_robust.py (path to model checkpoint) to run the MPC

Resulting images of MPC will be saved in ./MPC folder 
