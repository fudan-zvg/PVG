import os
import cv2

import imageio
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from scene.scene_utils import CameraInfo, SceneInfo, getNerfppNorm, fetchPly, storePly
from pathlib import Path
camera_ls = [2, 3]

"""
Most function brought from MARS
https://github.com/OPEN-AIR-SUN/mars/blob/69b9bf9d992e6b9f4027dfdc2a741c2a33eef174/mars/data/mars_kitti_dataparser.py
"""

def pad_poses(p):
    """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
    bottom = np.broadcast_to([0, 0, 0, 1.], p[..., :1, :4].shape)
    return np.concatenate([p[..., :3, :4], bottom], axis=-2)


def unpad_poses(p):
    """Remove the homogeneous bottom row from [..., 4, 4] pose matrices."""
    return p[..., :3, :4]


def transform_poses_pca(poses, fix_radius=0):
    """Transforms poses so principal components lie on XYZ axes.

  Args:
    poses: a (N, 3, 4) array containing the cameras' camera to world transforms.

  Returns:
    A tuple (poses, transform), with the transformed poses and the applied
    camera_to_world transforms.
    
    From https://github.com/SuLvXiangXin/zipnerf-pytorch/blob/af86ea6340b9be6b90ea40f66c0c02484dfc7302/internal/camera_utils.py#L161
  """
    t = poses[:, :3, 3]
    t_mean = t.mean(axis=0)
    t = t - t_mean

    eigval, eigvec = np.linalg.eig(t.T @ t)
    # Sort eigenvectors in order of largest to smallest eigenvalue.
    inds = np.argsort(eigval)[::-1]
    eigvec = eigvec[:, inds]
    rot = eigvec.T
    if np.linalg.det(rot) < 0:
        rot = np.diag(np.array([1, 1, -1])) @ rot

    transform = np.concatenate([rot, rot @ -t_mean[:, None]], -1)
    poses_recentered = unpad_poses(transform @ pad_poses(poses))
    transform = np.concatenate([transform, np.eye(4)[3:]], axis=0)

    # Flip coordinate system if z component of y-axis is negative
    if poses_recentered.mean(axis=0)[2, 1] < 0:
        poses_recentered = np.diag(np.array([1, -1, -1])) @ poses_recentered
        transform = np.diag(np.array([1, -1, -1, 1])) @ transform

    # Just make sure it's it in the [-1, 1]^3 cube
    if fix_radius>0:
        scale_factor = 1./fix_radius
    else:
        scale_factor = 1. / (np.max(np.abs(poses_recentered[:, :3, 3])) + 1e-5)
        scale_factor = min(1 / 10, scale_factor)

    poses_recentered[:, :3, 3] *= scale_factor
    transform = np.diag(np.array([scale_factor] * 3 + [1])) @ transform

    return poses_recentered, transform, scale_factor

def kitti_string_to_float(str):
    return float(str.split("e")[0]) * 10 ** int(str.split("e")[1])


def get_rotation(roll, pitch, heading):
    s_heading = np.sin(heading)
    c_heading = np.cos(heading)
    rot_z = np.array([[c_heading, -s_heading, 0], [s_heading, c_heading, 0], [0, 0, 1]])

    s_pitch = np.sin(pitch)
    c_pitch = np.cos(pitch)
    rot_y = np.array([[c_pitch, 0, s_pitch], [0, 1, 0], [-s_pitch, 0, c_pitch]])

    s_roll = np.sin(roll)
    c_roll = np.cos(roll)
    rot_x = np.array([[1, 0, 0], [0, c_roll, -s_roll], [0, s_roll, c_roll]])

    rot = np.matmul(rot_z, np.matmul(rot_y, rot_x))

    return rot


def tracking_calib_from_txt(calibration_path):
    """
    Extract tracking calibration information from a KITTI tracking calibration file.

    This function reads a KITTI tracking calibration file and extracts the relevant
    calibration information, including projection matrices and transformation matrices
    for camera, LiDAR, and IMU coordinate systems.

    Args:
        calibration_path (str): Path to the KITTI tracking calibration file.

    Returns:
        dict: A dictionary containing the following calibration information:
            P0, P1, P2, P3 (np.array): 3x4 projection matrices for the cameras.
            Tr_cam2camrect (np.array): 4x4 transformation matrix from camera to rectified camera coordinates.
            Tr_velo2cam (np.array): 4x4 transformation matrix from LiDAR to camera coordinates.
            Tr_imu2velo (np.array): 4x4 transformation matrix from IMU to LiDAR coordinates.
    """
    # Read the calibration file
    f = open(calibration_path)
    calib_str = f.read().splitlines()

    # Process the calibration data
    calibs = []
    for calibration in calib_str:
        calibs.append(np.array([kitti_string_to_float(val) for val in calibration.split()[1:]]))

    # Extract the projection matrices
    P0 = np.reshape(calibs[0], [3, 4])
    P1 = np.reshape(calibs[1], [3, 4])
    P2 = np.reshape(calibs[2], [3, 4])
    P3 = np.reshape(calibs[3], [3, 4])

    # Extract the transformation matrix for camera to rectified camera coordinates
    Tr_cam2camrect = np.eye(4)
    R_rect = np.reshape(calibs[4], [3, 3])
    Tr_cam2camrect[:3, :3] = R_rect

    # Extract the transformation matrices for LiDAR to camera and IMU to LiDAR coordinates
    Tr_velo2cam = np.concatenate([np.reshape(calibs[5], [3, 4]), np.array([[0.0, 0.0, 0.0, 1.0]])], axis=0)
    Tr_imu2velo = np.concatenate([np.reshape(calibs[6], [3, 4]), np.array([[0.0, 0.0, 0.0, 1.0]])], axis=0)

    return {
        "P0": P0,
        "P1": P1,
        "P2": P2,
        "P3": P3,
        "Tr_cam2camrect": Tr_cam2camrect,
        "Tr_velo2cam": Tr_velo2cam,
        "Tr_imu2velo": Tr_imu2velo,
    }


def calib_from_txt(calibration_path):
    """
    Read the calibration files and extract the required transformation matrices and focal length.

    Args:
        calibration_path (str): The path to the directory containing the calibration files.

    Returns:
        tuple: A tuple containing the following elements:
            traimu2v (np.array): 4x4 transformation matrix from IMU to Velodyne coordinates.
            v2c (np.array): 4x4 transformation matrix from Velodyne to left camera coordinates.
            c2leftRGB (np.array): 4x4 transformation matrix from left camera to rectified left camera coordinates.
            c2rightRGB (np.array): 4x4 transformation matrix from right camera to rectified right camera coordinates.
            focal (float): Focal length of the left camera.
    """
    c2c = []

    # Read and parse the camera-to-camera calibration file
    f = open(os.path.join(calibration_path, "calib_cam_to_cam.txt"), "r")
    cam_to_cam_str = f.read()
    [left_cam, right_cam] = cam_to_cam_str.split("S_02: ")[1].split("S_03: ")
    cam_to_cam_ls = [left_cam, right_cam]

    # Extract the transformation matrices for left and right cameras
    for i, cam_str in enumerate(cam_to_cam_ls):
        r_str, t_str = cam_str.split("R_0" + str(i + 2) + ": ")[1].split("\nT_0" + str(i + 2) + ": ")
        t_str = t_str.split("\n")[0]
        R = np.array([kitti_string_to_float(r) for r in r_str.split(" ")])
        R = np.reshape(R, [3, 3])
        t = np.array([kitti_string_to_float(t) for t in t_str.split(" ")])
        Tr = np.concatenate([np.concatenate([R, t[:, None]], axis=1), np.array([0.0, 0.0, 0.0, 1.0])[None, :]])

        t_str_rect, s_rect_part = cam_str.split("\nT_0" + str(i + 2) + ": ")[1].split("\nS_rect_0" + str(i + 2) + ": ")
        s_rect_str, r_rect_part = s_rect_part.split("\nR_rect_0" + str(i + 2) + ": ")
        r_rect_str = r_rect_part.split("\nP_rect_0" + str(i + 2) + ": ")[0]
        R_rect = np.array([kitti_string_to_float(r) for r in r_rect_str.split(" ")])
        R_rect = np.reshape(R_rect, [3, 3])
        t_rect = np.array([kitti_string_to_float(t) for t in t_str_rect.split(" ")])
        Tr_rect = np.concatenate(
            [np.concatenate([R_rect, t_rect[:, None]], axis=1), np.array([0.0, 0.0, 0.0, 1.0])[None, :]]
        )

        c2c.append(Tr_rect)

    c2leftRGB = c2c[0]
    c2rightRGB = c2c[1]

    # Read and parse the Velodyne-to-camera calibration file
    f = open(os.path.join(calibration_path, "calib_velo_to_cam.txt"), "r")
    velo_to_cam_str = f.read()
    r_str, t_str = velo_to_cam_str.split("R: ")[1].split("\nT: ")
    t_str = t_str.split("\n")[0]
    R = np.array([kitti_string_to_float(r) for r in r_str.split(" ")])
    R = np.reshape(R, [3, 3])
    t = np.array([kitti_string_to_float(r) for r in t_str.split(" ")])
    v2c = np.concatenate([np.concatenate([R, t[:, None]], axis=1), np.array([0.0, 0.0, 0.0, 1.0])[None, :]])

    # Read and parse the IMU-to-Velodyne calibration file
    f = open(os.path.join(calibration_path, "calib_imu_to_velo.txt"), "r")
    imu_to_velo_str = f.read()
    r_str, t_str = imu_to_velo_str.split("R: ")[1].split("\nT: ")
    R = np.array([kitti_string_to_float(r) for r in r_str.split(" ")])
    R = np.reshape(R, [3, 3])
    t = np.array([kitti_string_to_float(r) for r in t_str.split(" ")])
    imu2v = np.concatenate([np.concatenate([R, t[:, None]], axis=1), np.array([0.0, 0.0, 0.0, 1.0])[None, :]])

    # Extract the focal length of the left camera
    focal = kitti_string_to_float(left_cam.split("P_rect_02: ")[1].split()[0])

    return imu2v, v2c, c2leftRGB, c2rightRGB, focal


def get_poses_calibration(basedir, oxts_path_tracking=None, selected_frames=None):
    """
    Extract poses and calibration information from the KITTI dataset.

    This function processes the OXTS data (GPS/IMU) and extracts the
    pose information (translation and rotation) for each frame. It also
    retrieves the calibration information (transformation matrices and focal length)
    required for further processing.

    Args:
        basedir (str): The base directory containing the KITTI dataset.
        oxts_path_tracking (str, optional): Path to the OXTS data file for tracking sequences.
            If not provided, the function will look for OXTS data in the basedir.
        selected_frames (list, optional): A list of frame indices to process.
            If not provided, all frames in the dataset will be processed.

    Returns:
        tuple: A tuple containing the following elements:
            poses (np.array): An array of 4x4 pose matrices representing the vehicle's
                position and orientation for each frame (IMU pose).
            calibrations (dict): A dictionary containing the transformation matrices
                and focal length obtained from the calibration files.
            focal (float): The focal length of the left camera.
    """

    def oxts_to_pose(oxts):
        """
        OXTS (Oxford Technical Solutions) data typically refers to the data generated by an Inertial and GPS Navigation System (INS/GPS) that is used to provide accurate position, orientation, and velocity information for a moving platform, such as a vehicle. In the context of the KITTI dataset, OXTS data is used to provide the ground truth for the vehicle's trajectory and 6 degrees of freedom (6-DoF) motion, which is essential for evaluating and benchmarking various computer vision and robotics algorithms, such as visual odometry, SLAM, and object detection.

        The OXTS data contains several important measurements:

        1. Latitude, longitude, and altitude: These are the global coordinates of the moving platform.
        2. Roll, pitch, and yaw (heading): These are the orientation angles of the platform, usually given in Euler angles.
        3. Velocity (north, east, and down): These are the linear velocities of the platform in the local navigation frame.
        4. Accelerations (ax, ay, az): These are the linear accelerations in the platform's body frame.
        5. Angular rates (wx, wy, wz): These are the angular rates (also known as angular velocities) of the platform in its body frame.

        In the KITTI dataset, the OXTS data is stored as plain text files with each line corresponding to a timestamp. Each line in the file contains the aforementioned measurements, which are used to compute the ground truth trajectory and 6-DoF motion of the vehicle. This information can be further used for calibration, data synchronization, and performance evaluation of various algorithms.
        """
        poses = []

        def latlon_to_mercator(lat, lon, s):
            """
            Converts latitude and longitude coordinates to Mercator coordinates (x, y) using the given scale factor.

            The Mercator projection is a widely used cylindrical map projection that represents the Earth's surface
            as a flat, rectangular grid, distorting the size of geographical features in higher latitudes.
            This function uses the scale factor 's' to control the amount of distortion in the projection.

            Args:
                lat (float): Latitude in degrees, range: -90 to 90.
                lon (float): Longitude in degrees, range: -180 to 180.
                s (float): Scale factor, typically the cosine of the reference latitude.

            Returns:
                list: A list containing the Mercator coordinates [x, y] in meters.
            """
            r = 6378137.0  # the Earth's equatorial radius in meters
            x = s * r * ((np.pi * lon) / 180)
            y = s * r * np.log(np.tan((np.pi * (90 + lat)) / 360))
            return [x, y]

        # Compute the initial scale and pose based on the selected frames
        if selected_frames is None:
            lat0 = oxts[0][0]
            scale = np.cos(lat0 * np.pi / 180)
            pose_0_inv = None
        else:
            oxts0 = oxts[selected_frames[0][0]]
            lat0 = oxts0[0]
            scale = np.cos(lat0 * np.pi / 180)

            pose_i = np.eye(4)

            [x, y] = latlon_to_mercator(oxts0[0], oxts0[1], scale)
            z = oxts0[2]
            translation = np.array([x, y, z])
            rotation = get_rotation(oxts0[3], oxts0[4], oxts0[5])
            pose_i[:3, :] = np.concatenate([rotation, translation[:, None]], axis=1)
            pose_0_inv = invert_transformation(pose_i[:3, :3], pose_i[:3, 3])

        # Iterate through the OXTS data and compute the corresponding pose matrices
        for oxts_val in oxts:
            pose_i = np.zeros([4, 4])
            pose_i[3, 3] = 1

            [x, y] = latlon_to_mercator(oxts_val[0], oxts_val[1], scale)
            z = oxts_val[2]
            translation = np.array([x, y, z])

            roll = oxts_val[3]
            pitch = oxts_val[4]
            heading = oxts_val[5]
            rotation = get_rotation(roll, pitch, heading)  # (3,3)

            pose_i[:3, :] = np.concatenate([rotation, translation[:, None]], axis=1)  # (4, 4)
            if pose_0_inv is None:
                pose_0_inv = invert_transformation(pose_i[:3, :3], pose_i[:3, 3])

            pose_i = np.matmul(pose_0_inv, pose_i)
            poses.append(pose_i)

        return np.array(poses)

    # If there is no tracking path specified, use the default path
    if oxts_path_tracking is None:
        oxts_path = os.path.join(basedir, "oxts/data")
        oxts = np.array([np.loadtxt(os.path.join(oxts_path, file)) for file in sorted(os.listdir(oxts_path))])
        calibration_path = os.path.dirname(basedir)

        calibrations = calib_from_txt(calibration_path)

        focal = calibrations[4]

        poses = oxts_to_pose(oxts)

    # If a tracking path is specified, use it to load OXTS data and compute the poses
    else:
        oxts_tracking = np.loadtxt(oxts_path_tracking)
        poses = oxts_to_pose(oxts_tracking)  # (n_frames, 4, 4)
        calibrations = None
        focal = None
        # Set velodyne close to z = 0
        # poses[:, 2, 3] -= 0.8

    # Return the poses, calibrations, and focal length
    return poses, calibrations, focal


def invert_transformation(rot, t):
    t = np.matmul(-rot.T, t)
    inv_translation = np.concatenate([rot.T, t[:, None]], axis=1)
    return np.concatenate([inv_translation, np.array([[0.0, 0.0, 0.0, 1.0]])])


def get_camera_poses_tracking(poses_velo_w_tracking, tracking_calibration, selected_frames, scene_no=None):
    exp = False
    camera_poses = []

    opengl2kitti = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    start_frame = selected_frames[0]
    end_frame = selected_frames[1]

    #####################
    # Debug Camera offset
    if scene_no == 2:
        yaw = np.deg2rad(0.7)  ## Affects camera rig roll: High --> counterclockwise
        pitch = np.deg2rad(-0.5)  ## Affects camera rig yaw: High --> Turn Right
        # pitch = np.deg2rad(-0.97)
        roll = np.deg2rad(0.9)  ## Affects camera rig pitch: High -->  up
        # roll = np.deg2rad(1.2)
    elif scene_no == 1:
        if exp:
            yaw = np.deg2rad(0.3)  ## Affects camera rig roll: High --> counterclockwise
            pitch = np.deg2rad(-0.6)  ## Affects camera rig yaw: High --> Turn Right
            # pitch = np.deg2rad(-0.97)
            roll = np.deg2rad(0.75)  ## Affects camera rig pitch: High -->  up
            # roll = np.deg2rad(1.2)
        else:
            yaw = np.deg2rad(0.5)  ## Affects camera rig roll: High --> counterclockwise
            pitch = np.deg2rad(-0.5)  ## Affects camera rig yaw: High --> Turn Right
            roll = np.deg2rad(0.75)  ## Affects camera rig pitch: High -->  up
    else:
        yaw = np.deg2rad(0.05)
        pitch = np.deg2rad(-0.75)
        # pitch = np.deg2rad(-0.97)
        roll = np.deg2rad(1.05)
        # roll = np.deg2rad(1.2)

    cam_debug = np.eye(4)
    cam_debug[:3, :3] = get_rotation(roll, pitch, yaw)

    Tr_cam2camrect = tracking_calibration["Tr_cam2camrect"]
    Tr_cam2camrect = np.matmul(Tr_cam2camrect, cam_debug)
    Tr_camrect2cam = invert_transformation(Tr_cam2camrect[:3, :3], Tr_cam2camrect[:3, 3])
    Tr_velo2cam = tracking_calibration["Tr_velo2cam"]
    Tr_cam2velo = invert_transformation(Tr_velo2cam[:3, :3], Tr_velo2cam[:3, 3])

    camera_poses_imu = []
    for cam in camera_ls:
        Tr_camrect2cam_i = tracking_calibration["Tr_camrect2cam0" + str(cam)]
        Tr_cam_i2camrect = invert_transformation(Tr_camrect2cam_i[:3, :3], Tr_camrect2cam_i[:3, 3])
        # transform camera axis from kitti to opengl for nerf:
        cam_i_camrect = np.matmul(Tr_cam_i2camrect, opengl2kitti)
        cam_i_cam0 = np.matmul(Tr_camrect2cam, cam_i_camrect)
        cam_i_velo = np.matmul(Tr_cam2velo, cam_i_cam0)

        cam_i_w = np.matmul(poses_velo_w_tracking, cam_i_velo)
        camera_poses_imu.append(cam_i_w)

    for i, cam in enumerate(camera_ls):
        for frame_no in range(start_frame, end_frame + 1):
            camera_poses.append(camera_poses_imu[i][frame_no])

    return np.array(camera_poses)


def get_scene_images_tracking(tracking_path, sequence, selected_frames):
    [start_frame, end_frame] = selected_frames
    img_name = []
    sky_name = []

    left_img_path = os.path.join(os.path.join(tracking_path, "image_02"), sequence)
    right_img_path = os.path.join(os.path.join(tracking_path, "image_03"), sequence)

    left_sky_path = os.path.join(os.path.join(tracking_path, "sky_02"), sequence)
    right_sky_path = os.path.join(os.path.join(tracking_path, "sky_03"), sequence)

    for frame_dir in [left_img_path, right_img_path]:
        for frame_no in range(len(os.listdir(left_img_path))):
            if start_frame <= frame_no <= end_frame:
                frame = sorted(os.listdir(frame_dir))[frame_no]
                fname = os.path.join(frame_dir, frame)
                img_name.append(fname)

    for frame_dir in [left_sky_path, right_sky_path]:
        for frame_no in range(len(os.listdir(left_sky_path))):
            if start_frame <= frame_no <= end_frame:
                frame = sorted(os.listdir(frame_dir))[frame_no]
                fname = os.path.join(frame_dir, frame)
                sky_name.append(fname)

    return img_name, sky_name

def rotation_matrix(a, b):
    """Compute the rotation matrix that rotates vector a to vector b.

    Args:
        a: The vector to rotate.
        b: The vector to rotate to.
    Returns:
        The rotation matrix.
    """
    a = a / torch.linalg.norm(a)
    b = b / torch.linalg.norm(b)
    v = torch.cross(a, b)
    c = torch.dot(a, b)
    # If vectors are exactly opposite, we add a little noise to one of them
    if c < -1 + 1e-8:
        eps = (torch.rand(3) - 0.5) * 0.01
        return rotation_matrix(a + eps, b)
    s = torch.linalg.norm(v)
    skew_sym_mat = torch.Tensor(
        [
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ]
    )
    return torch.eye(3) + skew_sym_mat + skew_sym_mat @ skew_sym_mat * ((1 - c) / (s**2 + 1e-8))

def auto_orient_and_center_poses(
    poses,
):
    """
    From nerfstudio
    https://github.com/nerfstudio-project/nerfstudio/blob/8e0c68754b2c440e2d83864fac586cddcac52dc4/nerfstudio/cameras/camera_utils.py#L515
    """
    origins = poses[..., :3, 3]
    mean_origin = torch.mean(origins, dim=0)
    translation = mean_origin
    up = torch.mean(poses[:, :3, 1], dim=0)
    up = up / torch.linalg.norm(up)
    rotation = rotation_matrix(up, torch.Tensor([0, 0, 1]))
    transform = torch.cat([rotation, rotation @ -translation[..., None]], dim=-1)
    oriented_poses = transform @ poses
    return oriented_poses, transform

def readKittiMotInfo(args):
    cam_infos = []
    points = []
    points_time = []
    scale_factor = 1.0

    basedir = args.source_path
    scene_id = basedir[-4:]  # check
    kitti_scene_no = int(scene_id)
    tracking_path = basedir[:-13]  # check
    calibration_path = os.path.join(os.path.join(tracking_path, "calib"), scene_id + ".txt")
    oxts_path_tracking = os.path.join(os.path.join(tracking_path, "oxts"), scene_id + ".txt")

    tracking_calibration = tracking_calib_from_txt(calibration_path)
    focal_X = tracking_calibration["P2"][0, 0]
    focal_Y = tracking_calibration["P2"][1, 1]
    poses_imu_w_tracking, _, _ = get_poses_calibration(basedir, oxts_path_tracking)  # (n_frames, 4, 4) imu pose

    tr_imu2velo = tracking_calibration["Tr_imu2velo"]
    tr_velo2imu = invert_transformation(tr_imu2velo[:3, :3], tr_imu2velo[:3, 3])
    poses_velo_w_tracking = np.matmul(poses_imu_w_tracking, tr_velo2imu)  # (n_frames, 4, 4) velodyne pose

    # Get camera Poses   camare id: 02, 03
    for cam_i in range(2):
        transformation = np.eye(4)
        projection = tracking_calibration["P" + str(cam_i + 2)]  # rectified camera coordinate system -> image
        K_inv = np.linalg.inv(projection[:3, :3])
        R_t = projection[:3, 3]
        t_crect2c = np.matmul(K_inv, R_t)
        transformation[:3, 3] = t_crect2c
        tracking_calibration["Tr_camrect2cam0" + str(cam_i + 2)] = transformation

    first_frame = args.start_frame
    last_frame = args.end_frame

    frame_num = last_frame-first_frame+1
    if args.frame_interval > 0:
        time_duration = [-args.frame_interval*(frame_num-1)/2,args.frame_interval*(frame_num-1)/2]
    else:
        time_duration = args.time_duration

    selected_frames = [first_frame, last_frame]
    sequ_frames = selected_frames

    cam_poses_tracking = get_camera_poses_tracking(
        poses_velo_w_tracking, tracking_calibration, sequ_frames, kitti_scene_no
    )
    poses_velo_w_tracking = poses_velo_w_tracking[first_frame:last_frame + 1]

    # Orients and centers the poses
    oriented = torch.from_numpy(np.array(cam_poses_tracking).astype(np.float32))  # (n_frames, 3, 4)
    oriented, transform_matrix = auto_orient_and_center_poses(
        oriented
    )  # oriented (n_frames, 3, 4), transform_matrix (3, 4)
    row = torch.tensor([0, 0, 0, 1], dtype=torch.float32)
    zeros = torch.zeros(oriented.shape[0], 1, 4)
    oriented = torch.cat([oriented, zeros], dim=1)
    oriented[:, -1] = row  # (n_frames, 4, 4)
    transform_matrix = torch.cat([transform_matrix, row[None, :]], dim=0)  # (4, 4)
    cam_poses_tracking = oriented.numpy()
    transform_matrix = transform_matrix.numpy()

    image_filenames, sky_filenames = get_scene_images_tracking(
        tracking_path, scene_id, sequ_frames)

    # # Align Axis with vkitti axis
    poses = cam_poses_tracking.astype(np.float32)
    poses[:, :, 1:3] *= -1

    test_load_image = imageio.imread(image_filenames[0])
    image_height, image_width = test_load_image.shape[:2]
    cx, cy = image_width / 2.0, image_height / 2.0
    poses[..., :3, 3] *= scale_factor

    c2ws = poses
    for idx in tqdm(range(len(c2ws)), desc="Loading data"):
        c2w = c2ws[idx]
        w2c = np.linalg.inv(c2w)
        image_path = image_filenames[idx]
        image_name = os.path.basename(image_path)[:-4]
        sky_path = sky_filenames[idx]
        im_data = Image.open(image_path)
        W, H = im_data.size
        image = np.array(im_data) / 255.

        sky_mask = cv2.imread(sky_path)

        timestamp = time_duration[0] + (time_duration[1] - time_duration[0]) * (idx % (len(c2ws) // 2)) / (len(c2ws) // 2 - 1)
        R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]

        if idx < len(c2ws) / 2:
            point = np.fromfile(os.path.join(tracking_path, "velodyne", scene_id, image_name + ".bin"), dtype=np.float32).reshape(-1, 4)
            point_xyz = point[:, :3]
            point_xyz_world = (np.pad(point_xyz, ((0, 0), (0, 1)), constant_values=1) @ poses_velo_w_tracking[idx].T)[:, :3]
            points.append(point_xyz_world)
            point_time = np.full_like(point_xyz_world[:, :1], timestamp)
            points_time.append(point_time)
        frame_num = len(c2ws) // 2
        point_xyz = points[idx%frame_num]
        point_camera = (np.pad(point_xyz, ((0, 0), (0, 1)), constant_values=1)@ transform_matrix.T @ w2c.T)[:, :3]*scale_factor

        cam_infos.append(CameraInfo(uid=idx, R=R, T=T,
                                    image=image,
                                    image_path=image_filenames[idx], image_name=image_filenames[idx],
                                    width=W, height=H, timestamp=timestamp,
                                    fx=focal_X, fy=focal_Y, cx=cx, cy=cy, sky_mask=sky_mask,
                                    pointcloud_camera=point_camera))

        if args.debug_cuda and idx > 5:
            break
    pointcloud = np.concatenate(points, axis=0)
    pointcloud = (np.concatenate([pointcloud, np.ones_like(pointcloud[:,:1])], axis=-1) @ transform_matrix.T)[:, :3]

    pointcloud_timestamp = np.concatenate(points_time, axis=0)

    indices = np.random.choice(pointcloud.shape[0], args.num_pts, replace=True)
    pointcloud = pointcloud[indices]
    pointcloud_timestamp = pointcloud_timestamp[indices]

    # normalize poses
    w2cs = np.zeros((len(cam_infos), 4, 4))
    Rs = np.stack([c.R for c in cam_infos], axis=0)
    Ts = np.stack([c.T for c in cam_infos], axis=0)
    w2cs[:, :3, :3] = Rs.transpose((0, 2, 1))
    w2cs[:, :3, 3] = Ts
    w2cs[:, 3, 3] = 1
    c2ws = unpad_poses(np.linalg.inv(w2cs))
    c2ws, transform, scale_factor = transform_poses_pca(c2ws, fix_radius=args.fix_radius)
    c2ws = pad_poses(c2ws)
    for idx, cam_info in enumerate(tqdm(cam_infos, desc="Transform data")):
        c2w = c2ws[idx]
        w2c = np.linalg.inv(c2w)
        cam_info.R[:] = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
        cam_info.T[:] = w2c[:3, 3]
        cam_info.pointcloud_camera[:] *= scale_factor
    pointcloud = (np.pad(pointcloud, ((0, 0), (0, 1)), constant_values=1) @ transform.T)[:, :3]

    if args.eval:
        num_frame = len(cam_infos)//2
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if (idx % num_frame + 1) % args.testhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if (idx % num_frame + 1) % args.testhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    # for kitti have some static ego videos, we dont calculate radius here
    nerf_normalization = getNerfppNorm(train_cam_infos)
    nerf_normalization['radius'] = 1

    ply_path = os.path.join(args.source_path, "points3d.ply")
    if not os.path.exists(ply_path):
        rgbs = np.random.random((pointcloud.shape[0], 3))
        storePly(ply_path, pointcloud, rgbs, pointcloud_timestamp)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    time_interval = (time_duration[1] - time_duration[0]) / (frame_num - 1)


    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           time_interval=time_interval)

    return scene_info