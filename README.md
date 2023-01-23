# LVF_MPC

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
