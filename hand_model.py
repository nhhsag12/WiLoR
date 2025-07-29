import os
import sys
import time

os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["MESA_GL_VERSION_OVERRIDE"] = "4.1"

import gradio as gr
# import spaces
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from pathlib import Path
import argparse
import json
from typing import Dict, Optional

from wilor.models import WiLoR, load_wilor
from wilor.utils import recursive_to
from wilor.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from wilor.utils.renderer import Renderer, cam_crop_to_full


LIGHT_PURPLE = (0.25098039, 0.274117647, 0.65882353)

class HandModel:
    def __init__(self, 
                 checkpoint_path: str = 'pretrained_models/wilor_final.ckpt',
                 cfg_path: str = 'pretrained_models/model_config.yaml'):
        

        # Setup the renderer and the model
        self.model, self.model_cfg = load_wilor(checkpoint_path=checkpoint_path, cfg_path=cfg_path)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.renderer = Renderer(self.model_cfg, faces=self.model.mano.faces)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.detector = YOLO(f'pretrained_models/detector.pt').to(self.device)

    def render_reconstruction(self, image, conf, IoU_threshold=0.3):
        start_time = time.time()
        input_img, num_dets, reconstructions = self.run_wilow_model(image, conf, IoU_threshold=0.5)
        print(f"Processing time: {time.time() - start_time:.2f} seconds")
        if num_dets > 0:
            # Render front view

            misc_args = dict(
                mesh_base_color=LIGHT_PURPLE,
                scene_bg_color=(1, 1, 1),
                focal_length=reconstructions['focal'],
            )

            cam_view = self.renderer.render_rgba_multiple(reconstructions['verts'],
                                                    cam_t=reconstructions['cam_t'],
                                                    render_res=reconstructions['img_size'],
                                                    is_right=reconstructions['right'], **misc_args)

            # Overlay image

            input_img = np.concatenate([input_img, np.ones_like(input_img[:, :, :1])], axis=2)  # Add alpha channel
            input_img_overlay = input_img[:, :, :3] * (1 - cam_view[:, :, 3:]) + cam_view[:, :, :3] * cam_view[:, :, 3:]
            print(f"Total processing time: {time.time() - start_time:.2f} seconds")
            return input_img_overlay, f'{num_dets} hands detected'
        else:
            print(f"Total processing time: {time.time() - start_time:.2f} seconds")
            return input_img, f'{num_dets} hands detected'

    def run_wilow_model(self, image, conf, IoU_threshold=0.5):
        img_cv2 = image[..., ::-1]
        img_vis = image.copy()

        detections = self.detector(img_cv2, conf=conf, verbose=False, iou=IoU_threshold)[0]

        bboxes = []
        is_right = []
        for det in detections:
            Bbox = det.boxes.data.cpu().detach().squeeze().numpy()
            Conf = det.boxes.conf.data.cpu().detach()[0].numpy().reshape(-1).astype(np.float16)
            Side = det.boxes.cls.data.cpu().detach()
            # Bbox[:2] -= np.int32(0.1 * Bbox[:2])
            # Bbox[2:] += np.int32(0.1 * Bbox[ 2:])
            is_right.append(det.boxes.cls.cpu().detach().squeeze().item())
            bboxes.append(Bbox[:4].tolist())

            color = (255 * 0.208, 255 * 0.647, 255 * 0.603) if Side == 0. else (255 * 1, 255 * 0.78039, 255 * 0.2353)
            label = f'L - {Conf[0]:.3f}' if Side == 0 else f'R - {Conf[0]:.3f}'

            cv2.rectangle(img_vis, (int(Bbox[0]), int(Bbox[1])), (int(Bbox[2]), int(Bbox[3])), color, 3)
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(img_vis, (int(Bbox[0]), int(Bbox[1]) - 20), (int(Bbox[0]) + w, int(Bbox[1])), color, -1)
            cv2.putText(img_vis, label, (int(Bbox[0]), int(Bbox[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)


        if len(bboxes) != 0:
            boxes = np.stack(bboxes)
            right = np.stack(is_right)
            dataset = ViTDetDataset(self.model_cfg, img_cv2, boxes, right, rescale_factor=2.0)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

            all_verts = []
            all_cam_t = []
            all_right = []
            all_joints = []

            for batch in dataloader:
                batch = recursive_to(batch, self.device)

                with torch.no_grad():
                    out = self.model(batch)

                multiplier = (2 * batch['right'] - 1)
                pred_cam = out['pred_cam']
                pred_cam[:, 1] = multiplier * pred_cam[:, 1]
                box_center = batch["box_center"].float()
                box_size = batch["box_size"].float()
                img_size = batch["img_size"].float()
                scaled_focal_length = self.model_cfg.EXTRA.FOCAL_LENGTH / self.model_cfg.MODEL.IMAGE_SIZE * img_size.max()
                pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size,
                                                scaled_focal_length).detach().cpu().numpy()

                batch_size = batch['img'].shape[0]
                for n in range(batch_size):
                    verts = out['pred_vertices'][n].detach().cpu().numpy()
                    joints = out['pred_keypoints_3d'][n].detach().cpu().numpy()

                    is_right = batch['right'][n].cpu().numpy()
                    verts[:, 0] = (2 * is_right - 1) * verts[:, 0]
                    joints[:, 0] = (2 * is_right - 1) * joints[:, 0]

                    cam_t = pred_cam_t_full[n]

                    all_verts.append(verts)
                    all_cam_t.append(cam_t)
                    all_right.append(is_right)
                    all_joints.append(joints)

            reconstructions = {'verts': all_verts, 'cam_t': all_cam_t, 'right': all_right, 'img_size': img_size[n],
                            'focal': scaled_focal_length}
            return img_vis.astype(np.float32) / 255.0, len(detections), reconstructions
        else:
            return img_vis.astype(np.float32) / 255.0, len(detections), None