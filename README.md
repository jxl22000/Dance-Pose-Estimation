# Dance-Pose-Estimation
Given a reference and a user-inputted video, the files return a side-by-side comparison video using the yolov7 pose estimation model. First, I synched the audio of the two files using librosa's cross similarity function, which lined up the music (assuming the music in the two files is the same). With OpenCV, I ran the yolo pose detection model on each frame and marked the joints as well as the error in each frame.

![](https://github.com/jxl22000/Dance-Pose-Estimation/blob/main/PoseEstimationDemo.gif)

<img src="https://github.com/jxl22000/Dance-Pose-Estimation/blob/main/PoseEstimationDemo.gif" alt="Databay showcase gif" title="Databay showcase gif" width="500"/>


# Setup Notes
Start with start_here.py, download respective files

In CLI: python start_here.py --video1 "video1.mp4" --video2 "video2.mp4", where "video1.mp4" is the user video and "video2.mp4" is the reference video to compare the two poses. 

Video1 should be the "cover" of the song, and the algorithm will try to clip the beginning portion of video1 so the music aligns with video2. 

--threshold: If the joint angle is above this threshold, the joint will highlight as red. If it is above half of this threshold, it will highlight as yellow.
--mirror1: mirror the first video
--mirror2: mirror the second video

The output will be in a file called 'output.mp4' which shows the two videos side-by-side with the pose estimation on each person.

# Future Improvements

Occluded joints cause trouble and the camera's viewpoint may cause inaccuracies in joint angles. The algorithm can also only compress the first video. 

