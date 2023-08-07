import argparse
from run import start
# from tftest import start

# python start_here.py --video1 "vbswing1.mov" --video2 "vbswing2.mov"
# python start_here.py --video1 "feels1.mov" --video2 "feels2.mov"
# python start_here.py --video1 "feelsreal1.mp4" --video2 "feelsreal2.mp4"

ap = argparse.ArgumentParser()
ap.add_argument("-v2", "--video2", required=True,
	help="video file 2 (REFERENCE)")
ap.add_argument("-v1", "--video1", required=True,
	help="video file 1 (USER)")
ap.add_argument("-thresh", "--threshold", default=20,
				help="threshold in degrees of where error is reported")

args = vars(ap.parse_args())

start(args["video1"], args["video2"], args["threshold"])
