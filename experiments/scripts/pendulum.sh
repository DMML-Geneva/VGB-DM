#!/bin/bash

# Pendulum

## VGB-DM
python src/runner/run_hydra.py --multirun \
    exp=pendulum \
    exp.dataset.data_path_tr="./experiments/dataset/pendulum/one_to_many/friction/training/data_57ee8408ecdef126709e4b05ce621796" \
    exp.dataset.data_path_va="./experiments/dataset/pendulum/one_to_many/friction/training/data_3b2adad285b6fe8dde511d7605c6735e" \
    exp.algorithm=dyn-fm \
    exp.sampler.sampler_mode=pairs-history \
    exp.sampler.max_length=-1 \
    exp.sampler.enc_len_episode=25 \
    exp.enc_model.len_episode=25 \
    exp.enc_model.p_dim=1 \
    exp.enc_model.z_dim=1 \
    exp.vf_model.phys_model=true \
    exp.vf_model.second_order=true \
    exp.vf_model.interpolation=lagrange \
    exp.vf_model.history_size=2 \
    exp.optimization.beta_p=0.01 \
    exp.optimization.gamma_acc=0.5 \
    exp.training.early_stopping=false \
    exp.training.max_time_tr=60 \
    exp.optimization.n_epochs=6000 \
    exp.seed=3,4,5


# TFM with no Physics Model - exp.vf_model.phys_model=false
python src/runner/run_hydra.py --multirun \
    exp=pendulum \
    exp.dataset.data_path_tr="./experiments/dataset/pendulum/one_to_many/friction/training/data_57ee8408ecdef126709e4b05ce621796" \
    exp.dataset.data_path_va="./experiments/dataset/pendulum/one_to_many/friction/training/data_3b2adad285b6fe8dde511d7605c6735e" \
    exp.algorithm=dyn-fm \
    exp.sampler.sampler_mode=pairs-history \
    exp.sampler.max_length=-1 \
    exp.enc_model.p_dim=0 \
    exp.enc_model.z_dim=0 \
    exp.vf_model.phys_model=false \
    exp.vf_model.second_order=false \
    exp.vf_model.interpolation=linear \
    exp.vf_model.history_size=1 \
    exp.vf_model.activation=SELU \
    exp.training.early_stopping=false \
    exp.optimization.vf_lr=1e-3 \
    exp.optimization.n_epochs=6000 \
    exp.seed=3,4,5

#### TO TEST below ########
## Phys-VAE
python src/runner/run_hydra.py --multirun \
    exp=pendulum \
    exp.dataset.data_path_tr="./experiments/dataset/pendulum/one_to_many/friction/training/data_57ee8408ecdef126709e4b05ce621796" \
    exp.dataset.data_path_va="./experiments/dataset/pendulum/one_to_many/friction/training/data_3b2adad285b6fe8dde511d7605c6735e" \
    exp.algorithm=node \
    exp.sampler.sampler_mode=trajectory \
    exp.sampler.max_length=25 \
    exp.sampler.enc_len_episode=25 \
    exp.enc_model.len_episode=25 \
    exp.enc_model.p_dim=1 \
    exp.enc_model.z_dim=1 \
    exp.vf_model.phys_model=true \
    exp.vf_model.dim_state=2 \
    exp.vf_model.history_size=0 \
    exp.vf_model.second_order=false \
    exp.vf_model.ode_method=euler \
    exp.optimization.enc_lr=5e-4 \
    exp.optimization.vf_lr=5e-4 \
    exp.optimization.beta_p=1.0 \
    exp.optimization.alpha=0.01 \
    exp.optimization.beta=0.001 \
    exp.optimization.gamma=0.1 \
    exp.training.early_stopping=false \
    exp.training.max_time_tr=60 \
    exp.optimization.n_epochs=6000 \
    exp.seed=3,4,5

## BB-NODE
python src/runner/run_hydra.py --multirun \
    exp=pendulum \
    exp.dataset.data_path_tr="./experiments/dataset/pendulum/one_to_many/friction/training/data_57ee8408ecdef126709e4b05ce621796" \
    exp.dataset.data_path_va="./experiments/dataset/pendulum/one_to_many/friction/training/data_3b2adad285b6fe8dde511d7605c6735e" \
    exp.algorithm=node \
    exp.sampler.sampler_mode=trajectory \
    exp.sampler.max_length=25 \
    exp.sampler.enc_len_episode=25 \
    exp.enc_model.len_episode=25 \
    exp.enc_model.p_dim=0 \
    exp.enc_model.z_dim=2 \
    exp.vf_model.phys_model=false \
    exp.vf_model.dim_state=1 \
    exp.vf_model.history_size=0 \
    exp.vf_model.second_order=false \
    exp.vf_model.ode_method=euler \
    exp.optimization.enc_lr=5e-4 \
    exp.optimization.vf_lr=5e-4 \
    exp.training.early_stopping=false \
    exp.training.max_time_tr=60 \
    exp.optimization.n_epochs=6000 \
    exp.seed=3,4,5