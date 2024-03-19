import os
from os.path import join
from tqdm import tqdm

data_root = '/HDD_DISK/datasets/waymo/kitti_format/training'

tags = ['image_0','image_1','image_2','image_3','image_4','calib','velodyne','pose']
posts = ['.png','.png','.png','.png','.png','.txt', '.bin','.txt']

out_dir = 'data/waymo_scenes_streetsurf'

scene_ids = [3, 19, 36, 69, 81, 126, 139, 140, 146, 
             148, 157, 181, 200, 204, 226, 232, 237, 
             241, 245, 246, 271, 297, 302, 312, 314, 
             362, 482, 495, 524, 527]
scene_nums = [
    [0, 163],
    [0, 198],
    [0, 198],
    [0, 198],
    [0, 198],
    [0, 198],
    [0, 198],
    [17, 198],
    [0, 198],
    [0, 198],
    [0, 140],
    [24, 198],
    [0, 198],
    [0, 198],
    [0, 198],
    [0, 198],
    [0, 198],
    [30, 198],
    [80, 198],
    [0, 170],
    [70, 198],
    [0, 198],
    [0, 198],
    [0, 120],
    [0, 198],
    [0, 198],
    [0, 198],
    [0, 198],
    [0, 198],
    [0, 90],
]
os.makedirs(out_dir, exist_ok=True)

for scene_idx, scene_id in enumerate(scene_ids):
    scene_dir = join(out_dir, f'{scene_id:04d}001')
    os.makedirs(scene_dir, exist_ok=True)

    for tag in tags:
        os.makedirs(join(scene_dir, tag), exist_ok=True)
    for post, tag in zip(posts,tags):
        for i in tqdm(range(scene_nums[scene_idx][0], scene_nums[scene_idx][1])):
            cmd = "cp {} {}".format(join(data_root,tag,f'{scene_id:04d}{i:03d}'+post), 
                                    join(scene_dir, tag, f'{scene_id:04d}{i:03d}'+post))
            os.system(cmd)



