import cv2
from toolbelt import *


glb_cam_width = 640
glb_cam_height = 480

canvas_img = np.zeros((glb_cam_width, glb_cam_height, 3), np.uint8)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
glb_display_img = None
glb_insp_img = None
glb_gray_img = None
glb_hsv_img = None

glb_roi_tl = (0,0)
glb_roi_br = (0,0)
glb_teach_flag = False
glb_h_chan_max = 0
glb_h_chan_min = 0
glb_s_chan_max = 0
glb_s_chan_min = 0
glb_v_chan_max = 0
glb_v_chan_min = 0


def main():
    global glb_display_img, glb_insp_img, glb_gray_img, glb_hsv_img

    vidcap = cv2.VideoCapture(0)
    vidcap.set(cv2.CAP_PROP_FRAME_WIDTH, glb_cam_width)
    vidcap.set(cv2.CAP_PROP_FRAME_HEIGHT, glb_cam_height)
    main_str = "Main View"


    while vidcap.isOpened():
        success, buffer = vidcap.read()
        k = cv2.waitKey(1)
        org_img = buffer

        glb_display_img = org_img.copy()
        glb_insp_img = org_img.copy()
        glb_gray_img = cv2.cvtColor(glb_insp_img, cv2.COLOR_BGR2GRAY)
        glb_hsv_img = cv2.cvtColor(glb_insp_img, cv2.COLOR_BGR2HSV)

        imshow(main_str, glb_display_img)

        if k == 27:
            vidcap.release()
            break


if __name__ == '__main__':
    main()