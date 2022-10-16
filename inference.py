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
    device = choice_device(args.device_type)

    # Initialize the model
    g_model = build_model(args.model_arch_name, device)
    print(f"Build {args.model_arch_name} model successfully.")

    # Load model weights
    g_model = load_state_dict(g_model, args.model_weights_path)
    print(f"Load {args.model_arch_name} model weights {os.path.abspath(args.model_weights_path)} successfully.")

    # Start the verification mode of the model.

    g_model.eval()
    timestart = time.perf_counter()


    lr_tensors = imgproc.preprocess_one_image(args.inputs_path, device)
    frame_array = []
    for i in range(len(lr_tensors)):
        # Use the model to generate super-resolved images
        with torch.no_grad():
            sr_tensor = g_model(lr_tensors[i])

        # Save image
        sr_image = imgproc.tensor_to_image(sr_tensor, False, False)
        sr_image = cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR)
        
        cv2.imwrite(args.output_path + "_" + str (i) + ".png", sr_image)
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
    print(f"SR image save to {args.output_path}")

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
        print(filename)
        #reading each files
        img = cv2.imread(filename)
        
        height, width, layers = img.shape
        size = (width,height)
        
        #inserting the frames into an image array
        frame_array.append(img)
    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()

    print("Saved video to", pathOut)

    


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Using the model generator super-resolution images.")
    parser.add_argument("--model_arch_name",
                        type=str,
                        default="srresnet_x4")
    parser.add_argument("--inputs_path",
                        type=str,
                        default="./figure/" + resFile + ".txt",
                        help="Low-resolution image path.")
    parser.add_argument("--output_path",
                        type=str,
                        default="./figure/video_files/" + resFile + '/' + resFile,
                        help="Super-resolution image path.")
    parser.add_argument("--model_weights_path",
                        type=str,
                        default="./results/SRGAN_x4-ImageNet-c71a4860.pth.tar",
                        help="Model weights file path.")
    parser.add_argument("--device_type",
                        type=str,
                        default="cpu",
                        choices=["cpu", "cuda"])
    args = parser.parse_args()
    #timestart = time.perf_counter()
    main(args)

    pathIn= f'./figure/video_files/{resFile}'
    pathOut = f'./figure/videos/video_{resFile}.mp4'
    fps = 25.0
    convert_frames_to_video(pathIn, pathOut, fps)
    #print(time.perf_counter() - timestart)