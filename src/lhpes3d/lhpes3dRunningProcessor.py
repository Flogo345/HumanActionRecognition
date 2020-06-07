from argparse import ArgumentParser
import json
import os

import cv2
import numpy as np
import torch

from lhpes3d.modules.input_reader import VideoReader, ImageReader
from lhpes3d.modules.draw import Plotter3d, draw_poses
from lhpes3d.modules.parse_poses import parse_poses


class LHPES3DRunningProcessor():
    def __init__(self, model_path, use_openvino=False, device='GPU'):
        if use_openvino:
            from lhpes3d.modules.inference_engine_openvino import InferenceEngineOpenVINO
            self.model = InferenceEngineOpenVINO(model_path, device)
        else:
            from lhpes3d.modules.inference_engine_pytorch import InferenceEnginePyTorch
            self.model = InferenceEnginePyTorch(model_path, device)


    def __call__(self, scaled_img):
        out = self.model.infer(scaled_img)
        return out




def rotate_poses(poses_3d, R, t):
    R_inv = np.linalg.inv(R)
    for pose_id in range(len(poses_3d)):
        pose_3d = poses_3d[pose_id].reshape((-1, 4)).transpose()
        pose_3d[0:3, :] = np.dot(R_inv, pose_3d[0:3, :] - t)
        poses_3d[pose_id] = pose_3d.transpose().reshape(-1)

    return poses_3d
