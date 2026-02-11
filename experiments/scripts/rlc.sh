#!/bin/bash

# RLC 

## VGB-DM
python src/runner/run_hydra.py --multirun \
    exp=rlc \
    exp.algorithm=dyn-fm \
    exp.dataset.data_path_tr=./experiments/dataset/RLC/train_size_1000.pkl \
    exp.dataset.data_path_va=./experiments/dataset/RLC/valid_size_100.pkl \
    exp.sampler.sampler_mode=pairs-history \
    exp.sampler.max_length=-1 \
    exp.sampler.enc_len_episode=25 \
    exp.enc_model.len_episode=25 \
    exp.enc_model.p_dim=2 \
    exp.enc_model.z_dim=4 \
    exp.vf_model.phys_model=true \
    exp.vf_model.activation=SELU \
    exp.vf_model.ode_method=dopri5 \
    exp.optimization.n_epochs=2000 \
    exp.optimization.vf_lr=1e-3 \
    exp.optimization.enc_lr=1e-4 \
    exp.training.early_stopping=false \
    exp.training.log_val_step=20 \
    exp.seed=3,4,5

## TFM with no Physics Model - exp.vf_model.phys_model=false
python src/runner/run_hydra.py --multirun \
    exp=rlc \
    exp.algorithm=dyn-fm \
    exp.dataset.data_path_tr=./experiments/dataset/RLC/train_size_1000.pkl \
    exp.dataset.data_path_va=./experiments/dataset/RLC/valid_size_100.pkl \
    exp.sampler.sampler_mode=pairs-history \
    exp.sampler.max_length=-1 \
    exp.sampler.enc_len_episode=25 \
    exp.enc_model.len_episode=0 \
    exp.enc_model.p_dim=0 \
    exp.enc_model.z_dim=0 \
    exp.vf_model.phys_model=false \
    exp.vf_model.history_size=1 \
    exp.vf_model.activation=ReLU \
    exp.vf_model.ode_method=dopri5 \
    exp.optimization.n_epochs=2000 \
    exp.optimization.vf_lr=1e-4 \
    exp.optimization.enc_lr=1e-4 \
    exp.training.early_stopping=false \
    exp.training.log_val_step=20 \
    exp.seed=3,4,5


