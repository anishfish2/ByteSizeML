# ByteSizeML

**Explore our project on Devpost for in-depth details and documentation: [ByteSize on Devpost](https://devpost.com/software/bytesize?ref_content=my-projects-tab&ref_feature=my_projects)**

## Getting Started

Follow these steps to set up and run ByteSizeML on your local machine:

1. **Download/Clone Repository**: Download or clone this repository to your local machine.

2. **Optional: Create a Virtual Environment (venv)**: If you prefer, create a virtual environment to isolate project dependencies.

3. **Install Requirements**: Install the required Python packages by running the following command:

   ```bash
   pip install -r requirements.txt

4. Install PyTorch: Visit the PyTorch website and follow the "Install PyTorch" prompt to find the appropriate PyTorch download command. You can use pip as your package manager. Once you have the installation command, execute it to install PyTorch.

5. Ensure SRGAN_x4-ImageNet-c71a4860.pth.tar File: Make sure the SRGAN_x4-ImageNet-c71a4860.pth.tar file is located inside the results folder.

6. Modify "inference.py": Change line 101 in the inference.py script from "res2" to the name of your text file.

7. Modify "imgproc.py": Adjust line 208 and 209 in the imgproc.py script to specify the height and width of your image.

8. Create "figure" Folder: Create a folder named "figure," and within it, include the subfolders "video_files" and "videos." Place your text file inside the "figure" folder.

9. Run "inference.py": Execute the inference.py script. After running it, an .mp4 video file should appear in the "videos" subfolder, ready for playback.

## Frontend and Backend Repositories
1. Frontend: Explore the frontend of this project on GitHub: [ByteSize Frontend](https://github.com/Abhishek-More/ByteSize)

2. Backend: Access the backend codebase on GitHub: [ByteSize Backend](https://github.com/NitroGuy10/ByteSizeBackend)

## Additional Details
For more comprehensive information, documentation, and project details, please visit our ByteSize project page on Devpost:  [ByteSize on Devpost](https://devpost.com/software/bytesize?ref_content=my-projects-tab&ref_feature=my_projects)**

Enjoy exploring ByteSizeML, and feel free to reach out if you have any questions or feedback!



