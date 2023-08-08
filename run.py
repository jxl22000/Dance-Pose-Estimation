import math
import cv2
import numpy as np
import torch
from torchvision import transforms
import audio
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint
from moviepy.editor import VideoFileClip, vfx
import os


# (B, G, R)
green = (50, 255, 50)
yellow = (50, 255, 255)
red = (50, 50, 255)
white = (255, 255, 255)


def load_model():

    weights = torch.load('yolov7-w6-pose.pt', map_location=device)
    model = weights['model']
    _ = model.float().eval()

    if torch.cuda.is_available():
        # half() turns predictions into float16 tensors
        # which significantly lowers inference time
        model.half().to(device)
    return model


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = load_model()

# Given an image, use the yolo human pose estimation model to predict a pose


def run_inference(image):
    # Resize and pad image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB )

    image = letterbox(image, 960, stride=64, auto=True)[0] # shape: (567, 960, 3)
    # Apply transforms
    image = transforms.ToTensor()(image) # torch.Size([3, 567, 960])
    if torch.cuda.is_available():
        image = image.half().to(device)
    # Turn image into batch
    image = image.unsqueeze(0) # torch.Size([1, 3, 567, 960])
    with torch.no_grad():
        output, _ = model(image)
    return output, image


def draw_keypoints(output, image):

    model = load_model()
    output = non_max_suppression_kpt(output,
                                     0.25, # Confidence Threshold
                                     0.65, # IoU Threshold
                                     nc=model.yaml['nc'], # Number of Classes
                                     nkpt=model.yaml['nkpt'], # Number of Keypoints
                                     kpt_label=True)
    with torch.no_grad():
        output = output_to_keypoint(output)
    nimg = image[0].permute(1, 2, 0) * 255
    nimg = nimg.cpu().numpy().astype(np.uint8)
    nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB )
    # for idx in range(output.shape[0]):
    #   plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)

    return nimg, output


def findAngle(image, kpts, tup, color=white):

    x1, y1 = kpts[3*tup[0]], kpts[3*tup[0] + 1]
    x2, y2 = kpts[3*tup[1]], kpts[3*tup[1] + 1]
    x3, y3 = kpts[3*tup[2]], kpts[3*tup[2] + 1]

    angle = math.degrees(math.atan2(y3-y2, x3-x2) - math.atan2(y1-y2, x1-x2))
    if angle < 0: angle += 360

    cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
    cv2.line(image, (int(x3), int(y3)), (int(x2), int(y2)), color, 3)

    cv2.circle(image, (int(x1), int(y1)), 5, white, cv2.FILLED)
    cv2.circle(image, (int(x2), int(y2)), 5, white, cv2.FILLED)
    cv2.circle(image, (int(x3), int(y3)), 5, white, cv2.FILLED)

    return angle


def error(image, kpts, tup, color):

    x1, y1 = kpts[3 * tup[0]], kpts[3 * tup[0] + 1]
    x2, y2 = kpts[3 * tup[1]], kpts[3 * tup[1] + 1]
    x3, y3 = kpts[3 * tup[2]], kpts[3 * tup[2] + 1]

    cv2.line(image, (int(x1/ 2 + x2/ 2) , int(y1 / 2 + y2/ 2)) , (int(x2), int(y2)), color, 3)
    cv2.line(image, (int(x3/ 2 + x2/ 2) , int(y3/ 2 + y2/ 2)) , (int(x2), int(y2)), color, 3)

    cv2.circle(image, (int(x2), int(y2)), 5, color, cv2.FILLED)


def start(video1, video2, threshold, mirror1, mirror2):

    # attempt to match the audios together:

    cap1 = cv2.VideoCapture(video1)
    cap2 = cv2.VideoCapture(video2)
    hi = int(min(cap1.get(4), cap2.get(4)))
    wi = int(min(cap1.get(3), cap2.get(3)))

    clip1, clip2 = audio.audio(video1, video2)
    filename1, filename2, fps = audio.matchClips(clip1, clip2, mirror1, mirror2)

    if filename1 is None:
        print("error")

    cap1 = cv2.VideoCapture(filename1)
    cap2 = cv2.VideoCapture(filename2)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    # hi = int(min(cap1.get(4), cap2.get(4)))
    # wi = int(min(cap1.get(3), cap2.get(3)))

    out = cv2.VideoWriter('output.mp4', fourcc, fps, (wi * 2, hi))
    poseTuples = [(5, 7, 9), (6, 8, 10), (6, 12, 14), (5, 11, 13), (11, 13, 15), (12, 14, 16)]

    while cap1.isOpened():
        ret1, image1 = cap1.read()
        ret2, image2 = cap2.read()

        if ret1 is True and ret2 is True:
            output, image1 = run_inference(image1)
            image1, output = draw_keypoints(output, image1)

            Langles = []
            Rangles = []

            kpts1 = output[0, 7:].T
            for i in range(len(poseTuples)):
                Langles.append(findAngle(image1, kpts1, poseTuples[i], color=green))

            output, image2 = run_inference(image2)
            image2, output = draw_keypoints(output, image2)

            kpts2 = output[0, 7:].T
            for i in range(len(poseTuples)):
                Rangles.append(findAngle(image2, kpts2, poseTuples[i], color=green))

            # compare scores of each joint

            for i in range(len(Langles)):
                deg = abs(Langles[i] - Rangles[i])
                if deg > threshold:
                    color = red
                elif deg > threshold / 2:
                    color = (red[0], red[1] + deg * 20 - 100, red[2])
                else:
                    color = (green[0], green[1], green[2] + deg * 20)
                error(image1, kpts1, poseTuples[i], color)

            # if hi < image1.shape[0]: image1 = image1[int(image1.shape[0]/2 - hi/2): int(image1.shape[0]/2 + hi/2), :]
            # if hi < image2.shape[0]: image2 = image2[int(image2.shape[0]/2 - hi/2): int(image2.shape[0]/2 + hi/2), :]
            # if wi < image1.shape[1]: image1 = image1[:, int(image1.shape[1]/2 - wi/2): int(image1.shape[1]/2 + wi/2)]
            # if wi < image2.shape[1]: image2 = image2[:, int(image2.shape[1]/2 - wi/2): int(image2.shape[1]/2 + wi/2)]

            # if (hi > image1.shape[0] and hi > image2.shape[0]) or (wi > image1.shape[1] and wi > image2.shape[1]):
            #     cv2.resize(image1, (wi, hi))
            #     cv2.resize(image2, (wi ,hi))

            image1 = cv2.resize(image1, (wi, hi))
            image2 = cv2.resize(image2, (wi, hi))

            vis = np.zeros((hi, 2 * wi, 3), np.uint8)
            vis[:hi, :wi, :3] = image1
            vis[:hi, wi:2*wi, :3] = image2

            cv2.imshow('Pose estimation', vis)
            out.write(vis)
        else:
            break

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap1.release()
    cap2.release()
    out.release()
    cv2.destroyAllWindows()