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


class CRect:
    def __init__(self, tL, bR):  #tL and bR should be passed as tuples
        self.tL = tL
        self.tR = (bR[0], tL[1])
        self.bL = (tL[0], bR[1])
        self.bR = bR
        self.top = tL[1]
        self.bottom = bR[1]
        self.left = tL[0]
        self.right = bR[0]
        self.width = bR[0]- tL[0]
        self.height = bR[1] - tL[1]
        self.center = ( int((self.right+self.left)*0.5),  int((self.top+self.bottom)*0.5) )

    def area(self):
        return (self.width * self.height)

    def perim(self):
        return (2 * self.width) + (2 * self.height)

    def pos(self):
        return self.tL, self.tR, self.bR, self.bL

    def isValid(self):
        if self.bR[0] > self.tL[0] and self.bR[1] > self.tL[1]:
            return True
        else:
            return False

    def validate(self, image):
        h, w = image.shape[:2]
        if self.top < 0:
            self.top = 0
            self.tL = (self.tL[0], self.top)
            self.tR = (self.bR[0], self.top)
        if self.bottom > h:
            self.bottom = h-1
            self.bL = (self.bL[0], self.bottom)
            self.bR = (self.bR[0], self.bottom)
        if self.left < 0:
            self.left = 0
            self.tL = (self.left, self.tL[1])
            self.bL = (self.left, self.bL[1])
        if self.right > w:
            self.right = w-1
            self.tR = (self.right, self.tR[1])
            self.bR = (self.right, self.bR[1])


def get_patch_hsv_range(img, rect_roi):
    hsv_img = img.copy()
    hsv_img = cv2.medianBlur(hsv_img, 5)
    hsv_img = get_crop_img(hsv_img, rect_roi.tL[0], rect_roi.tL[1], rect_roi.height, rect_roi.width)
    h_chan, s_chan, v_chan = cv2.split(hsv_img)
    min_max_list = []
    hsv_channel = [h_chan, s_chan, v_chan]
    for n, chan in enumerate(hsv_channel):
        # c_list = chan[np.nonzero(chan)]
        min_max_list.append(np.max(chan))
        min_max_list.append(np.min(chan))
    return min_max_list


def run_teach():
    global glb_teach_flag, glb_h_chan_max, glb_h_chan_min, glb_s_chan_max, glb_s_chan_min, glb_v_chan_max, glb_v_chan_min

    teach_roi()
    rect_roi = CRect(glb_roi_tl, glb_roi_br)
    glb_h_chan_max, glb_h_chan_min, glb_s_chan_max, glb_s_chan_min, glb_v_chan_max, glb_v_chan_min = get_patch_hsv_range(glb_hsv_img, rect_roi)
    glb_teach_flag = True


def teach_roi():
    global canvas_img, glb_teach_flag, glb_roi_tl, glb_roi_br

    str = "Select ROI"
    canvas_img = glb_insp_img.copy()
    cv2.putText(canvas_img, "Draw a rectangle", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 1, cv2.LINE_AA)
    cv2.putText(canvas_img, "Press 'Enter' once done", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 1, cv2.LINE_AA)
    cv2.namedWindow(str, cv2.WINDOW_NORMAL)
    x,y,w,h = cv2.selectROI(str, canvas_img, False)
    try:
        glb_teach_flag = True
        glb_roi_tl = (x, y)
        glb_roi_br = (x+w, y+h)
        cv2.destroyWindow(str)
    except:
        cv2.destroyWindow(str)


def main():
    global glb_display_img, glb_insp_img, glb_gray_img, glb_hsv_img

    vidcap = cv2.VideoCapture(0)
    vidcap.set(cv2.CAP_PROP_FRAME_WIDTH, glb_cam_width)
    vidcap.set(cv2.CAP_PROP_FRAME_HEIGHT, glb_cam_height)
    main_str = "Main View"

    while vidcap.isOpened():
        success, org_img = vidcap.read()
        k = cv2.waitKey(20)
        glb_display_img = org_img.copy()
        glb_insp_img = org_img.copy()
        glb_gray_img = cv2.cvtColor(glb_insp_img, cv2.COLOR_BGR2GRAY)
        glb_hsv_img = cv2.cvtColor(glb_insp_img, cv2.COLOR_BGR2HSV)

        if k == ord('t'):
            run_teach()

        if glb_teach_flag:
            lower_color = np.array([glb_h_chan_min, glb_s_chan_min, glb_v_chan_min])
            upper_color = np.array([glb_h_chan_max, glb_s_chan_max, glb_v_chan_max])
            mask = cv2.inRange(glb_hsv_img.copy(), lower_color, upper_color)
            output_img = glb_insp_img.copy()
            output_img[np.where(mask == 0)] = 0
            output_img[np.where(mask != 0)] = 255
            imshow("Result View", output_img)

        imshow(main_str, glb_display_img)

        if k == 27:
            vidcap.release()
            break


if __name__ == '__main__':
    main()