# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import argparse
import os
import time
import cv2
import torch
from torch import nn
from os.path import isfile, join
import imgproc
import model
from utils import load_state_dict
import shutil
import numpy as np
import matplotlib.pyplot as plt
model_names = sorted(
    name for name in model.__dict__ if
    name.islower() and not name.startswith("__") and callable(model.__dict__[name]))






def choice_device(device_type: str) -> torch.device:
    # Select model processing equipment type
    if device_type == "cuda":
        device = torch.device("cuda", 0)
    else:
        device = torch.device("cpu")
    return device


def build_model(model_arch_name: str, device: torch.device) -> nn.Module:
    # Initialize the super-resolution model
    g_model = model.__dict__[model_arch_name](in_channels=3,
                                              out_channels=3,
                                              channels=64,
                                              num_blocks=16)
    g_model = g_model.to(device=device)

    return g_model


def main(args):
    device = choice_device(args["device_type"])

    # Initialize the model
    g_model = build_model(args["model_arch_name"], device)
    print(f"Build {args['model_arch_name']} model successfully.")

    # Load model weights
    g_model = load_state_dict(g_model, args['model_weights_path'])
    print(f"Load {args['model_arch_name']} model weights {os.path.abspath(args['model_weights_path'])} successfully.")

    # Start the verification mode of the model.

    g_model.eval()
    timestart = time.perf_counter()


    lr_tensors = imgproc.preprocess_one_image(args['inputs_path'], device)
    frame_array = []
    for i in range(len(lr_tensors)):
        # Use the model to generate super-resolved images
        with torch.no_grad():
            sr_tensor = g_model(lr_tensors[i])

        # Save image
        sr_image = imgproc.tensor_to_image(sr_tensor, False, False)
        sr_image = cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR)
        
        cv2.imwrite(args['output_path'] + "_" + str (i) + ".png", sr_image)
        #         img = cv2.imread(filename)
        #     height, width, layers = img.shape
        #     size = (width,height)
        #     print(filename)
        #     #inserting the frames into an image array
        
        frame_array.append(sr_image)
    fps = 30;
    size = (60, 120)
    out = cv2.VideoWriter('./figure/videos/',cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()
    print(time.perf_counter() - timestart)
    print(f"SR image save to {args['output_path']}")

resFile = 'res2'
resFolderPath = os.path.join(os.getcwd(), "figure", "video_files", resFile)
if os.path.exists(resFolderPath):
    shutil.rmtree(resFolderPath)
os.makedirs(resFolderPath)

def convert_frames_to_video(pathIn,pathOut,fps):
    frame_array = []
    size = None
    
    
    for i in range(len([entry for entry in os.listdir(pathIn) if os.path.isfile(os.path.join(pathIn, entry))])):
        filename = f"{pathIn}/{resFile}_{i}.png"
        print(i)
        print(filename)
        #reading each files
        img = cv2.imread(filename)
        
        height, width, layers = img.shape
        size = (width,height)
        prototxt = "./colorization_deploy_v2.prototxt"
        caffe_model = "./colorization_release_v2.caffemodel"
        pts_npy = "./pts_in_hull.npy"
        test_image =  filename

        net = cv2.dnn.readNetFromCaffe(prototxt, caffe_model)
        pts = np.load(pts_npy)
        
        layer1 = net.getLayerId("class8_ab")
        #print(layer1)
        layer2 = net.getLayerId("conv8_313_rh")
        #print(layer2)
        pts = pts.transpose().reshape(2, 313, 1, 1)
        net.getLayer(layer1).blobs = [pts.astype("float32")]
        net.getLayer(layer2).blobs = [np.full([1, 313], 2.606, dtype="float32")]

        test_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        test_image = cv2.cvtColor(test_image, cv2.COLOR_GRAY2RGB)

        normalized = test_image.astype("float32") / 255.0
        lab_image = cv2.cvtColor(normalized, cv2.COLOR_RGB2LAB)
        resized = cv2.resize(lab_image, (224, 224))
        L = cv2.split(resized)[0]
        L -= 50

        net.setInput(cv2.dnn.blobFromImage(L))
        ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
        ab = cv2.resize(ab, (test_image.shape[1], test_image.shape[0]))
        L = cv2.split(lab_image)[0]
        LAB_colored = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
        RGB_colored = cv2.cvtColor(LAB_colored,cv2.COLOR_LAB2RGB)
        RGB_colored = np.clip(RGB_colored, 0, 1)
        RGB_colored = (255 * RGB_colored).astype("uint8")
        # if (i % 5 == 0):
        #     plt.imshow(RGB_colored)
        #     plt.title('Colored Image')
        #     plt.show()
        RGB_BGR = cv2.cvtColor(RGB_colored, cv2.COLOR_RGB2BGR)

        #inserting the frames into an image array
        frame_array.append(RGB_BGR)
    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()

    print("Saved video to", pathOut)

def real_main(res_file_name):
    resFile = res_file_name

    args = {
        "device_type": "cpu",
        "model_weights_path": "./results/SRGAN_x4-ImageNet-c71a4860.pth.tar",
        "output_path": "./figure/video_files/" + resFile + '/' + resFile,
        "inputs_path": "./figure/" + resFile + ".txt",
        "model_arch_name": "srresnet_x4"
    }
    #timestart = time.perf_counter()
    main(args)

    pathIn= f'./figure/video_files/{resFile}'
    pathOut = f'./figure/videos/video_{resFile}.mp4'
    fps = 25.0
    convert_frames_to_video(pathIn, pathOut, fps)
    #print(time.perf_counter() - timestart)


if __name__ == "__main__":
    real_main("res2")
