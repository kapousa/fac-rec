import glob
import shutil

import cv2
import os

from simple_facerec import SimpleFacerec


class CVHelper:
    test_value = ''

    def __init__(self):
        self.test_value = '_'

    def recognise_face(self, save_path, save_file_name, save_as_video=False, source=0):
        try:
            # Encode faces from a folder
            sfr = SimpleFacerec()
            sfr.load_encoding_images("images/")

            # Load Camera
            cap = cv2.VideoCapture(source)
            os.mkdir("temp")
            n = 0
            while True:
                ret, frame = cap.read()

                # Detect Faces
                face_locations, face_names = sfr.detect_known_faces(frame)
                height, width, layers = frame.shape
                for face_loc, name in zip(face_locations, face_names):
                    y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
                    cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)
                    cv2.imwrite('{}_{}.{}'.format("temp/", n, "png"), frame)
                    n += 1
                cv2.imshow("Frame", frame)

                key = cv2.waitKey(1)
                if key == 27:
                    break

            # Save Video with detection
            if (save_as_video):
                print("{0}{1}.mp4".format(save_path, save_file_name))
                video = cv2.VideoWriter("{0}{1}.mp4".format(save_path, save_file_name), cv2.VideoWriter_fourcc(*'mp4v'),
                                        5.0, (width, height))
                aa = "{}{}".format(os.getcwd(), "/temp/*.png")
                filenames = glob.glob(aa, recursive=True)
                for filename in filenames:
                    img = cv2.imread(filename)
                    video.write(img)
                video.release()
                shutil.rmtree('temp')

            cv2.destroyAllWindows()

            print("Face Detection Run Successfully")
            return 1
        except Exception as e:
            shutil.rmtree('temp')
            print("Error Happened")

            return -1
