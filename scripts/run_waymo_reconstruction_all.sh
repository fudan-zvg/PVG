CUDA_VISIBLE_DEVICES=1 python train.py \
--config configs/waymo_reconstruction.yaml \
source_path=data/waymo_scenes/0145050 \
model_path=eval_output/waymo_reconstruction/0145050

CUDA_VISIBLE_DEVICES=1 python train.py \
--config configs/waymo_reconstruction.yaml \
source_path=data/waymo_scenes/0147030 \
model_path=eval_output/waymo_reconstruction/0147030

CUDA_VISIBLE_DEVICES=1 python train.py \
--config configs/waymo_reconstruction.yaml \
source_path=data/waymo_scenes/0158150 \
model_path=eval_output/waymo_reconstruction/0158150

CUDA_VISIBLE_DEVICES=1 python train.py \
--config configs/waymo_reconstruction.yaml \
source_path=data/waymo_scenes/0017085 \
model_path=eval_output/waymo_reconstruction/0017085
