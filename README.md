# ByteSizeML

1. Download/Clone repo

2. pip install requirements.txt

3. Ensure SRGAN_x4-ImageNet-c71a4860.pth.tar file is inside results folder

4. Change line 101 in "inference.py" from "res2" to the name of your text file

5. Change line 208 and 209 in "imgproc.py" to the height and width of your image

6. Create a folder called "figure" which contains the folders "video_files" and "videos" as well as your text file

7. Run "inference.py" and an .mp4 file should appeaer in the "videos" folder for you to play
