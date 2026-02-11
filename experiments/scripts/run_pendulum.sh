#!/bin/bash

# FM
## BB-FM PENDULUM
python src/runner/run_task.py --config experiments/scripts/configs/exp/pendulum.yaml \
    dataset.data_path_tr="./experiments/dataset/pendulum/one_to_many/friction/training/data_bd57ff1300c896bedae4815e639bb34a" \
    algorithm=dyn-fm \
    sampler.sampler_mode=seq-pairs \
    sampler.max_length=25 \
    vf_model.phys_model=false \
    vf_model.second_order=false \
    vf_model.interpolation=linear \
    optimization.beta_p=1.0 \
    training.early_stopping=false \
    training.max_time_tr=60 \
    optimization.n_epochs=6000 \
    seed=3

python src/runner/run_task.py --config experiments/scripts/configs/exp/pendulum.yaml \
    dataset.data_path_tr="./experiments/dataset/pendulum/one_to_many/friction/training/data_bd57ff1300c896bedae4815e639bb34a" \
    algorithm=dyn-fm \
    sampler.sampler_mode=seq-pairs \
    sampler.max_length=25 \
    vf_model.phys_model=false \
    vf_model.second_order=false \
    vf_model.interpolation=linear \
    optimization.beta_p=1.0 \
    training.early_stopping=false \
    training.max_time_tr=60 \
    optimization.n_epochs=6000 \
    seed=4

python src/runner/run_task.py --config experiments/scripts/configs/exp/pendulum.yaml \
    dataset.data_path_tr="./experiments/dataset/pendulum/one_to_many/friction/training/data_bd57ff1300c896bedae4815e639bb34a" \
    algorithm=dyn-fm \
    sampler.sampler_mode=seq-pairs \
    sampler.max_length=25 \
    vf_model.phys_model=false \
    vf_model.second_order=false \
    vf_model.interpolation=linear \
    optimization.beta_p=1.0 \
    training.early_stopping=false \
    training.max_time_tr=60 \
    optimization.n_epochs=6000 \
    seed=5

# NODE
# PhysVAE
python src/runner/run_task.py --config experiments/scripts/configs/exp/pendulum.yaml \
    dataset.data_path_tr="./experiments/dataset/pendulum/one_to_many/friction/training/data_bd57ff1300c896bedae4815e639bb34a" \
    algorithm=node \
    sampler.sampler_mode=trajectory \
    sampler.max_length=-1 \
    enc_model.len_episode=25 \
    vf_model.phys_model=true \
    vf_model.dim_state=2 \
    vf_model.ode_method=euler \
    optimization.enc_lr=5e-4 \
    optimization.vf_lr=5e-4 \
    optimization.beta_p=1.0 \
    optimization.alpha=0.01 \
    optimization.beta=0.001 \
    optimization.gamma=0.1 \
    training.early_stopping=false \
    training.max_time_tr=60 \
    optimization.n_epochs=6000 \
    seed=3

python src/runner/run_task.py --config experiments/scripts/configs/exp/pendulum.yaml \
    dataset.data_path_tr="./experiments/dataset/pendulum/one_to_many/friction/training/data_bd57ff1300c896bedae4815e639bb34a" \
    algorithm=node \
    sampler.sampler_mode=trajectory \
    sampler.max_length=-1 \
    enc_model.len_episode=25 \
    vf_model.phys_model=true \
    vf_model.dim_state=2 \
    vf_model.ode_method=euler \
    optimization.enc_lr=5e-4 \
    optimization.vf_lr=5e-4 \
    optimization.beta_p=1.0 \
    optimization.alpha=0.01 \
    optimization.beta=0.001 \
    optimization.gamma=0.1 \
    training.early_stopping=false \
    training.max_time_tr=60 \
    optimization.n_epochs=6000 \
    seed=4
    
python src/runner/run_task.py --config experiments/scripts/configs/exp/pendulum.yaml \
    dataset.data_path_tr="./experiments/dataset/pendulum/one_to_many/friction/training/data_bd57ff1300c896bedae4815e639bb34a" \
    algorithm=node \
    sampler.sampler_mode=trajectory \
    sampler.max_length=-1 \
    enc_model.len_episode=25 \
    vf_model.phys_model=true \
    vf_model.dim_state=2 \
    vf_model.ode_method=euler \
    optimization.enc_lr=5e-4 \
    optimization.vf_lr=5e-4 \
    optimization.beta_p=1.0 \
    optimization.alpha=0.01 \
    optimization.beta=0.001 \
    optimization.gamma=0.1 \
    training.early_stopping=false \
    training.max_time_tr=60 \
    optimization.n_epochs=6000 \
    seed=5


## BB-NODE
python src/runner/run_task.py --config experiments/scripts/configs/exp/pendulum.yaml \
    dataset.data_path_tr="./experiments/dataset/pendulum/one_to_many/friction/training/data_bd57ff1300c896bedae4815e639bb34a" \
    algorithm=node \
    sampler.sampler_mode=trajectory \
    sampler.max_length=-1 \
    enc_model.len_episode=25 \
    vf_model.phys_model=false \
    vf_model.ode_method=euler \
    optimization.enc_lr=5e-4 \
    optimization.vf_lr=5e-4 \
    optimization.beta_p=1.0 \
    training.early_stopping=false \
    training.max_time_tr=60 \
    optimization.n_epochs=6000 \
    seed=3

python src/runner/run_task.py --config experiments/scripts/configs/exp/pendulum.yaml \
    dataset.data_path_tr="./experiments/dataset/pendulum/one_to_many/friction/training/data_bd57ff1300c896bedae4815e639bb34a" \
    algorithm=node \
    sampler.sampler_mode=trajectory \
    sampler.max_length=-1 \
    enc_model.len_episode=25 \
    vf_model.phys_model=false \
    vf_model.ode_method=euler \
    optimization.enc_lr=5e-4 \
    optimization.vf_lr=5e-4 \
    optimization.beta_p=1.0 \
    training.early_stopping=false \
    training.max_time_tr=60 \
    optimization.n_epochs=6000 \
    seed=4

python src/runner/run_task.py --config experiments/scripts/configs/exp/pendulum.yaml \
    dataset.data_path_tr="./experiments/dataset/pendulum/one_to_many/friction/training/data_bd57ff1300c896bedae4815e639bb34a" \
    algorithm=node \
    sampler.sampler_mode=trajectory \
    sampler.max_length=-1 \
    enc_model.len_episode=25 \
    vf_model.phys_model=false \
    vf_model.ode_method=euler \
    optimization.enc_lr=5e-4 \
    optimization.vf_lr=5e-4 \
    optimization.beta_p=1.0 \
    training.early_stopping=false \
    training.max_time_tr=60 \
    optimization.n_epochs=6000 \
    seed=5


