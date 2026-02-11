
#!/bin/bash

# Lorenz Attractor

## VGB-DM
python src/runner/run_hydra.py --multirun \
    exp=lorenz \
    exp.dataset.data_path_tr="./experiments/dataset/lorenz_attractor/train/tr_lorenz_data_seed_1_f99f234a781aeda45d3ec612da565d6b.pt"\
    exp.dataset.data_path_va="./experiments/dataset/lorenz_attractor/val/val_lorenz_data_seed_2_1ec97522a1b023702f6b3e4ba4eba3b5.pt"\
    exp.algorithm=dyn-fm \
    exp.sampler.sampler_mode=pairs-history \
    exp.sampler.max_length=-1 \
    exp.sampler.enc_len_episode=30 \
    exp.enc_model.len_episode=30 \
    exp.enc_model.p_dim=2 \
    exp.enc_model.z_dim=8 \
    exp.vf_model.phys_model=true \
    exp.vf_model.vf_phys=true \
    exp.optimization.phys_weight_scheduler=True \
    exp.optimization.phys_warming_steps=500 \
    exp.vf_model.interpolation=linear \
    exp.vf_model.history_size=4 \
    exp.training.early_stopping=false \
    exp.optimization.vf_lr=1e-3 \
    exp.optimization.enc_lr=1e-4 \
    exp.optimization.beta_p=0.1 \
    exp.optimization.beta_z=0.1 \
    exp.optimization.use_kl_annealing=True \
    exp.optimization.kl_anneal_steps=1000 \
    exp.optimization.n_epochs=3500 \
    exp.training.max_time_tr=120 \
    exp.seed=3,4,5

# BB TFM
python src/runner/run_hydra.py --multirun \
    exp=lorenz \
    exp.dataset.data_path_tr="./experiments/dataset/lorenz_attractor/train/tr_lorenz_data_seed_1_f99f234a781aeda45d3ec612da565d6b.pt"\
    exp.dataset.data_path_va="./experiments/dataset/lorenz_attractor/val/val_lorenz_data_seed_2_1ec97522a1b023702f6b3e4ba4eba3b5.pt"\
    exp.algorithm=dyn-fm \
    exp.sampler.sampler_mode=pairs-history \
    exp.sampler.enc_len_episode=30 \
    exp.sampler.max_length=-1 \
    exp.enc_model.p_dim=0 \
    exp.enc_model.z_dim=0 \
    exp.vf_model.phys_model=false \
    exp.vf_model.vf_phys=false \
    exp.vf_model.interpolation=linear \
    exp.vf_model.history_size=4 \
    exp.vf_model.activation=SELU \
    exp.training.early_stopping=false \
    exp.optimization.vf_lr=1e-3 \
    exp.optimization.n_epochs=3500 \
    exp.training.max_time_tr=120 \
    exp.seed=3,4,5


# Phys VAE
python src/runner/run_hydra.py --multirun \
    exp=lorenz \
    exp.dataset.data_path_tr="./experiments/dataset/lorenz_attractor/train/tr_lorenz_data_seed_1_f99f234a781aeda45d3ec612da565d6b.pt"\
    exp.dataset.data_path_va="./experiments/dataset/lorenz_attractor/val/val_lorenz_data_seed_2_1ec97522a1b023702f6b3e4ba4eba3b5.pt"\
    exp.algorithm=node \
    exp.sampler.sampler_mode=trajectory \
    exp.sampler.enc_len_episode=30 \
    exp.sampler.max_length=30 \
    exp.enc_model.p_dim=2 \
    exp.enc_model.z_dim=8 \
    exp.vf_model.phys_model=true \
    exp.vf_model.vf_phys=false \
    exp.vf_model.ode_method=euler \
    exp.vf_model.val_ode_method=dopri5 \
    exp.vf_model.history_size=0 \
    exp.training.early_stopping=false \
    exp.optimization.vf_lr=1e-3 \
    exp.optimization.enc_lr=1e-4 \
    exp.optimization.n_epochs=3500 \
    exp.optimization.alpha=0.1,0.0 \
    exp.optimization.beta=0.0 \
    exp.optimization.gamma=0.0 \
    exp.seed=3,4,5

# BB VAE
python src/runner/run_hydra.py --multirun \
    exp=lorenz \
    exp.dataset.data_path_tr="./experiments/dataset/lorenz_attractor/train/tr_lorenz_data_seed_1_f99f234a781aeda45d3ec612da565d6b.pt"\
    exp.dataset.data_path_va="./experiments/dataset/lorenz_attractor/val/val_lorenz_data_seed_2_1ec97522a1b023702f6b3e4ba4eba3b5.pt"\
    exp.algorithm=node \
    exp.sampler.sampler_mode=trajectory \
    exp.sampler.enc_len_episode=30 \
    exp.sampler.max_length=30 \
    exp.enc_model.p_dim=2 \
    exp.enc_model.z_dim=8 \
    exp.vf_model.phys_model=false \
    exp.vf_model.vf_phys=false \
    exp.vf_model.ode_method=euler \
    exp.vf_model.val_ode_method=dopri5 \
    exp.vf_model.history_size=0 \
    exp.training.early_stopping=false \
    exp.optimization.vf_lr=1e-3 \
    exp.optimization.enc_lr=1e-4 \
    exp.optimization.n_epochs=3500 \
    exp.optimization.alpha=0.0 \
    exp.optimization.beta=0.0 \
    exp.optimization.gamma=0.0 \
    exp.seed=3,4,5




