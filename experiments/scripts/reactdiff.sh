#!/bin/bash

# ReactDiff

## VGB-DM
python src/runner/run_hydra.py --multirun \
    exp=reactdiff \
    exp.algorithm=dyn-fm \
    exp.sampler.sampler_mode=pairs-history \
    exp.sampler.max_length=-1 \
    exp.sampler.enc_len_episode=5 \
    exp.dataset.data_path_tr=experiments/dataset/reactdiff/training/new/2b65b9822b077a8ff7a0655d74696823/dataset_2b65b9822b077a8ff7a0655d74696823.pt \
    exp.dataset.data_path_va=experiments/dataset/reactdiff/val/879e48f34e8175caa7af8bb0a5cf224e/dataset_879e48f34e8175caa7af8bb0a5cf224e.pt \
    exp.enc_model.len_episode=5 \
    exp.enc_model.p_dim=2 \
    exp.enc_model.z_dim=10 \
    exp.vf_model.phys_model=true \
    exp.vf_model.history_size=1 \
    exp.vf_model.t_dependent=false \
    exp.vf_model.activation=SELU \
    exp.optimization.n_epochs=2000 \
    exp.optimization.vf_lr=1e-3 \
    exp.optimization.enc_lr=1e-3 \
    +exp.optimization.use_kl_annealing=True \
    +exp.optimization.kl_anneal_steps=500 \
    +exp.optimization.phys_weight_scheduler=True \
    +exp.optimization.phys_warming_steps=500 \
    exp.training.early_stopping=false \
    exp.training.log_val_step=20 \
    exp.seed=3,4,5

## TFM with no Physics Model - exp.vf_model.phys_model=false
python src/runner/run_hydra.py --multirun \
    exp=reactdiff \
    exp.algorithm=dyn-fm \
    exp.dataset.data_path_tr=./experiments/dataset/reactdiff/training/new/2b65b9822b077a8ff7a0655d74696823/dataset_2b65b9822b077a8ff7a0655d74696823.pt \
    exp.dataset.data_path_va=./experiments/dataset/reactdiff/val/879e48f34e8175caa7af8bb0a5cf224e/dataset_879e48f34e8175caa7af8bb0a5cf224e.pt \
    exp.sampler.sampler_mode=pairs-history \
    exp.sampler.max_length=-1 \
    exp.sampler.enc_len_episode=5 \
    exp.enc_model.len_episode=0 \
    exp.enc_model.p_dim=0 \
    exp.enc_model.z_dim=0 \
    exp.vf_model.phys_model=false \
    exp.vf_model.history_size=1 \
    exp.vf_model.activation=SELU \
    exp.vf_model.ode_method=dopri5 \
    exp.optimization.n_epochs=2000 \
    exp.optimization.vf_lr=1e-3 \
    exp.training.early_stopping=false \
    exp.seed=3,4,5