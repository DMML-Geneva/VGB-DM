# Variational Grey-Box Dynamic Matching (VGB-DM)

This repository contains the official implementation of the paper **Variational Grey-Box Dynamic Matching**, accepted at AISTATS 2026.

## TL;DR

VGB-DM is a novel approach for learning the dynamics of physical systems that combines incomplete physics models with data-driven deep generative methods. Unlike traditional simulation-based approaches that rely on expensive ODE solvers during training, VGB-DM adopts a simulation-free framework inspired by Flow Matching, enabling scalable and stable learning of high-dimensional dynamics. The method employs structured variational inference with separate latent variables to model stochasticity in the dynamics and infer unknown physical parameters, allowing it to capture complex multi-modal behaviours while maintaining interpretability. We applied VGB-DM various ODE/PDE systems and weather forecasting tasks compared to state-of-the-art baselines.

## Installation

1.  **Create and activate the Conda environment:**

    ```bash
    conda create -n vgb-dm python=3.12
    conda activate vgb-dm
    ```

2.  **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Install the project in editable mode:**

    ```bash
    pip install -e .
    ```

## Dataset Generation

To generate the datasets for each task, follow the instructions below.

### RLC Circuit

```bash
python src/data/generate_dataset/rlc/generate_rlc_dataset.py
```

This command saves the training and validation/test datasets in `experiments/dataset/RLC`.

### Damped Pendulum

To generate the training, validation, and test sets, run the following command with different seed values:

```bash
python src/data/generate_dataset/pendulum/simulate_from_conf.py --seed=<seed_value> --n-samples=1000
```

### Reaction-Diffusion

```bash
python src/data/generate_dataset/reactdiff/make_dataset.py
```

To change the number of samples and the seed, edit the `n_samples` and `seed` properties in the `src/data/generate_dataset/reactdiff/params.yaml` file.

### Lorenz System

```bash
python src/data/generate_dataset/lorenz_attractor/generate_dataset.py --seed=<seed_value> --n_trajectories=1000
```

## Model Training and Evaluation

To run a model training experiment, refer to the bash script examples in `experiments/scripts/` for training different models.

### Training

An example command to train VGB-DM on the Lorenz dataset:

```bash
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
    exp.vf_model.interpolation=linear \
    exp.vf_model.history_size=4 \
    exp.training.early_stopping=false \
    exp.optimization.vf_lr=1e-3 \
    exp.optimization.enc_lr=1e-4 \
    exp.optimization.beta_p=0.1 \
    exp.optimization.beta_z=0.1 \
    exp.optimization.n_epochs=3500 \
    exp.training.max_time_tr=120 \
    exp.seed=3
```

The best checkpoint for each training run is saved in the `outputs/` directory. Model directory names are generated from a hashed string of the training configuration. The best checkpoint is stored as `best_chkpt.pth` in the corresponding model directory.

### Evaluation

To evaluate trained models, use the `evaluate.py` script, which automatically searches for all model checkpoints in the specified directory:

```bash
python src/evaluate/evaluate.py --root_dir=./outputs/lorenz/1cd9c8e5b1a0c7e4f2b1a3d9c8e5b1a0c7e4f2b1a3d9c8e5b1a0c7e4f2b1a3 --dataset_path=./experiments/dataset/lorenz_attractor/test/test_lorenz_data_seed_3_9c8e5b1a0c7e4f2b1a3d9c8e5b1a0c7e4f2b1a3
```

Arguments:
- `--root_dir`: Path to the directory containing trained model checkpoints (searches recursively for `best_chkpt.pth` files)
- `--dataset_path`: Path to the test dataset file

# Weather and Climate Modelling

## Dataset Preparation

For the climate modelling experiments, we use the ERA5 reanalysis dataset for medium-range weather forecasting (similar to ClimODE). The dataset includes five key meteorological variables: (i) geopotential, (ii) ground temperature, (iii) atmospheric temperature, (iv) and (v) the two ground-level wind components (u10 and v10).

To prepare the dataset:

1. Download ERA5 data with 5.625° resolution from [WeatherBench](https://dataserv.ub.tum.de/index.php/s/m1524895)
2. Save the data in `experiments/dataset/era5_data`

The required directory structure:

```
era5_data/
├── 10m_u_component_of_wind/
├── 10m_v_component_of_wind/
├── 2m_temperature/
├── constants/
├── geopotential_500/
└── temperature_850/
```




## Training

After preparing the dataset, you can train the models using the following commands.

### GB-DM Model

For hourly data:

```bash
python src/grey_box_clim/gb_dm_train_hourly.py --force_reload
```

For monthly data:

```bash
python src/grey_box_clim/gb_dm_train_monthly.py --force_reload
```

### ClimODE Model

For hourly data:

```bash
python src/grey_box_clim/climode_train_hourly.py --force_reload
```
The GB-DM can run also on a small-mid GPU model (e.g 12GB VRAM) with a batch size of 16, while the ClimODE model requires a larger GPU (e.g. 25GB VRAM or slightly more) and a smaller batch size of 8 for training on hourly data. However, we suggest to keep the 32 GB of batch size for the GB-DM model to achieve better performance.

For monthly data:

```bash
python src/grey_box_clim/climode_train_monthly.py --force_reload
```

The `--force_reload` flag forces regeneration of cached datasets. Use it when running the model for the first time or after making changes to the dataset generation code.

## Evaluation

To evaluate trained models, use the following commands:

### Hourly Data Evaluation

```bash
python src/grey_box_clim/evaluation_hourly.py --model=src/grey_box_clim/Models/bests/gb_dm_683e4c13.pt
```

### Monthly Data Evaluation

```bash
python src/grey_box_clim/evaluation_monthly.py --model_path=src/grey_box_clim/Models/bests/gb_dm_monthly_7a303583.pt
```

The `--model` (or `--model_path`) argument should point to the checkpoint of the trained model. Pre-trained checkpoints are available in `src/grey_box_clim/Models/bests/`, while newly trained models are saved in the `outputs/` directory.

To evaluate the naive baseline (using the last observed state for all future predictions), set `--model=data`.

**Note:** Before evaluation, ensure the dataset is prepared and run the training script at least once (for either hourly or monthly data) to generate the cached velocity data. The evaluation script will automatically load cached velocity data if available, or compute and cache it for future use.


