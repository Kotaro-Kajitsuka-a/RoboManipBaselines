python ./bin/Train.py DiffusionPolicy \
  --dataset_dir ./dataset/45epis_highspeed_board_removed \
  --camera_names top \
  --scheduler ddim --num_epochs 1000 \
  --train_ratio 1.0 --val_ratio 0.01 ;

python ./bin/Train.py DiffusionPolicy \
  --dataset_dir ./dataset/45epis_highspeed_board_removed \
  --camera_names top \
  --scheduler ddim --num_epochs 1000 \
  --train_ratio 1.0 --val_ratio 0.01 \
  --seed 52;

python ./bin/Train.py DiffusionPolicy \
  --dataset_dir ./dataset/45epis_highspeed_board_removed \
  --camera_names top \
  --scheduler ddim --num_epochs 1000 \
  --train_ratio 1.0 --val_ratio 0.01 \
  --seed 62






python ./bin/Train.py DiffusionPolicy \
  --dataset_dir ./dataset/45epis_highspeed \
  --camera_names top \
  --scheduler ddim --num_epochs 1000 \
  --train_ratio 1.0 --val_ratio 0.01 \
  --seed 52;

python ./bin/Train.py DiffusionPolicy \
  --dataset_dir ./dataset/45epis_highspeed \
  --camera_names top \
  --scheduler ddim --num_epochs 1000 \
  --train_ratio 1.0 --val_ratio 0.01 \
  --seed 62;

python ./bin/Train.py DiffusionPolicy \
  --dataset_dir ./dataset/45epis_highspeed_board_removed \
  --camera_names top \
  --scheduler ddim --num_epochs 1000 \
  --train_ratio 1.0 --val_ratio 0.01 \
  --seed 72;

python ./bin/Train.py DiffusionPolicy \
  --dataset_dir ./dataset/45epis_highspeed_board_removed \
  --camera_names top \
  --scheduler ddim --num_epochs 1000 \
  --train_ratio 1.0 --val_ratio 0.01 \
  --seed 82;

python ./bin/Train.py DiffusionPolicy \
  --dataset_dir ./dataset/45epis_birdseye_human \
  --camera_names top \
  --scheduler ddim --num_epochs 1000 \
  --train_ratio 1.0 --val_ratio 0.01 \
  --seed 72;

python ./bin/Train.py DiffusionPolicy \
  --dataset_dir ./dataset/45epis_birdseye_human \
  --camera_names top \
  --scheduler ddim --num_epochs 1000 \
  --train_ratio 1.0 --val_ratio 0.01 \
  --seed 82;

  