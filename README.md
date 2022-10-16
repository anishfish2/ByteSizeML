# ByteSizeML

1. Download/Clone repo

2. If you know how to and want to make a venv, make a venv

3. pip install -r requirements.txt
 
4. Go to the [PyTorch website](https://pytorch.org/) and use the "Install PyTorch" prompt to find the appropriate pytorch download command. You'll probably want to use pip as your package manager. Once you have the install command, run it

5. Ensure SRGAN_x4-ImageNet-c71a4860.pth.tar file is inside results folder

6. Change line 101 in "inference.py" from "res2" to the name of your text file

7. Change line 208 and 209 in "imgproc.py" to the height and width of your image

8. Create a folder called "figure" which contains the folders "video_files" and "videos" as well as your text file

9. Run "inference.py" and an .mp4 file should appeaer in the "videos" folder for you to play
