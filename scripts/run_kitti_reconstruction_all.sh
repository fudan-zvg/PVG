CUDA_VISIBLE_DEVICES=3 python train.py \
--config configs/kitti_reconstruction.yaml \
source_path=data/kitti_mot/training/image_02/0001 \
model_path=eval_output/kitti_reconstruction/0001 \
start_frame=380 end_frame=431 

CUDA_VISIBLE_DEVICES=3 python train.py \
--config configs/kitti_reconstruction.yaml \
source_path=data/kitti_mot/training/image_02/0002 \
model_path=eval_output/kitti_reconstruction/0002 \
start_frame=140 end_frame=224 

CUDA_VISIBLE_DEVICES=3 python train.py \
--config configs/kitti_reconstruction.yaml \
source_path=data/kitti_mot/training/image_02/0006 \
model_path=eval_output/kitti_reconstruction/0006 \
start_frame=65 end_frame=120 
