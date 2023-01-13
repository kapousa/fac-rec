import sys

from CVHelper import CVHelper

video_location = sys.argv[0]
video_name = sys.argv[1]
save_video = sys.argv[2]
cam_source = sys.argv[3]

cvhelper = CVHelper()
recgfaces = cvhelper.recognise_face(video_location, video_name, save_video, cam_source)
