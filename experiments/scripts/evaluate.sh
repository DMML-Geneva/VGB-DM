# RLC
python src/evaluate/evaluate.py --root_dir=./outputs/rlc/690e4ca73015a5782ba9a0b414eee1c4 \
    --dataset_path=./experiments/dataset/RLC/test_size_100.pkl

# pendulum
python src/evaluate/evaluate.py --root_dir=./outputs/pendulum/6870aee67349ef6f60dc1eb74d8dc778/ \
    --dataset_path=./experiments/dataset/pendulum/one_to_many/friction/test/data_863c006a9cdeb07ab35bfd4c6f92a9d5 

# reactdiff
python src/evaluate/evaluate.py --root_dir=./outputs/reactdiff/ffa3924e0c0b8ff13f61dd78cea08cf2/dyn-fm/black_box/ \
    --dataset_path=./experiments/dataset/reactdiff/test/3e7cf06d9323caaea6c4f566b7d328f2/dataset_3e7cf06d9323caaea6c4f566b7d328f2.pt 

# lorenz
python src/evaluate/evaluate.py --root_dir=./outputs/lorenz/1cd9c8e5b1a0c7e4f2b1a3d9c8e5b1a0c7e4f2b1a3d9c8e5b1a0c7e4f2b1a3/ \
    --dataset_path=./experiments/dataset/lorenz_attractor/test/test_lorenz_data_seed_3_9c8e5b1a0c7e4f2b1a3d9c8e5b1a0c7e4f2b1a3.pt
    
