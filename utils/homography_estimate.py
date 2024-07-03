from scipy.optimize import least_squares, newton
import numpy as np
from cv2 import warpPerspective
import random
import torch
from math import sin, cos, pi

magnification_min = 0.5
magnification_max = 1.5
translation_range = 25
projective_range = 0    # 平飞，平行线仍然平行

def homography_transformation(image: torch.Tensor):  # 用来替换dataloader中的torchvision.transforms
    shape = image.shape
    magnification = random.uniform(magnification_min, magnification_max)
    angle = random.uniform(-pi, pi)
    translation_x = random.uniform(-translation_range, translation_range)
    translation_y = random.uniform(-translation_range, translation_range)
    H_11 = magnification * cos(angle)
    H_12 = -magnification * sin(angle)
    H_13 = translation_y
    H_21 = magnification * sin(angle)
    H_22 = magnification * cos(angle)
    H_23 = translation_x
    H = torch.Tensor([H_11, H_12, H_13], [H_21, H_22, H_23], [0, 0, 1])
    warp_M = H.numpy()[0:2, :]
    image_np = image.numpy()
    target_image = warpPerspective(image_np, warp_M, shape)
    return target_image, H
    


# class homography_transform(torch.nn.Module):
#     def __init__(self, size):
#         super().__init__()
        

#     def forward(self):
#         self.H = self.generate_homography()


#     def generate_homography(self):
#         self.magnification = random.uniform(magnification_min, magnification_max)
#         self.angle = random.uniform(-pi, pi)
#         self.translation_x = random.uniform(-translation_range, translation_range)
#         self.translation_y = random.uniform(-translation_range, translation_range)
#         H_11 = self.magnification * cos(self.angle)
#         H_12 = -self.magnification * sin(self.angle)
#         H_13 = self.translation_y
#         H_21 = self.magnification * sin(self.angle)
#         H_22 = self.magnification * cos(self.angle)
#         H_23 = self.translation_x
#         H = torch.Tensor([H_11, H_12, H_13], [H_21, H_22, H_23], [0, 0, 1])
#         return H

    
#     def __repr__(self) -> str:
#         s = (
#             f"{self.__class__.__name__}("
#             f", magnification={self.magnification}"
#             f", angle={self.angle}"
#             f", translation_x={self.translation_x}"
#             f", translation_y={self.translation_y}"
#             f", H={self.H}"
#             f")"
#         )
#         return s



# def homography_estimate(feature_map_image: torch.Tensor, feature_map_map: torch.Tensor):
#     H = np.eye(3)
#     fMapI = 
#     image_size = feature_map_image.shape
#     warp_F_image = warpPerspective(feature_map_image, H, )
    
