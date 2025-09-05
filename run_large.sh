export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL="osmesa"

python tests/main.py --run_group Large --env ant_maze_large --max_path_length 300 --seed 0 --traj_batch_size 16 --n_parallel 4 --n_epochs_per_eval 500  --dim_option 4  --algo SZPC --exp_name TheBest --phi_type Projection --explore_type SZN  --trans_optimization_epochs 150  --is_wandb 1  --SZN_w2 10 --SZN_w3 3 --SZN_window_size 20 --SZN_repeat_time 3  --n_epochs 20000 --Repr_max_step 300 --dual_slack 1e-3 
