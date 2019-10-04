from pdb import set_trace as byebug
import math
import cv2

class Webcam:
    def __init__(self, device_number = 0):
        self.video = cv2.VideoCapture(device_number)
        self.buffer = None

    def set_prop_frame_width(self, value):
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, value)
    def set_prop_frame_height(self, value):
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, value)
    def set_prop_auto_focus(self, value):
        self.video.set(cv2.CAP_PROP_AUTOFOCUS, value)
    def set_prop_focus(self, value):
        self.video.set(cv2.CAP_PROP_FOCUS, value)
    def set_prop_brightness(self, value):
        self.video.set(cv2.CAP_PROP_BRIGHTNESS, value)
    def set_prop_contrast(self, value):
        self.video.set(cv2.CAP_PROP_CONTRAST, value)
    def set_prop_saturation(self, value):
        self.video.set(cv2.CAP_PROP_SATURATION, value)
    def set_prop_auto_exposure(self, value):
        self.video.set(cv2.CAP_PROP_AUTO_EXPOSURE, value)
    def set_prop_gain(self, value):
        self.video.set(cv2.CAP_PROP_GAIN, value)
    def set_prop_sharpness(self, value):
        self.video.set(cv2.CAP_PROP_SHARPNESS, value)

    def grab_frame(self, grayscale = False):
        success, self.buffer = self.video.read()

        if grayscale:
            self.buffer = cv2.cvtColor(self.buffer, cv2.COLOR_BGR2GRAY)

        return success

    def release(self):
        self.video.release()


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

    def translateX(self, offset_x):
        point_tl = (self.tL[0]+offset_x, self.tL[1])
        point_br = (self.bR[0]+offset_x, self.bR[1])
        return ( CRect(point_tl, point_br) )

    def translateY(self, offset_y):
        point_tl = (self.tL[0], self.tL[1]+offset_y)
        point_br = (self.bR[0], self.bR[1]+offset_y)
        return ( CRect(point_tl, point_br) )

    def inflatX(self, offset_x):
        self.tL = (self.tL[0]-offset_x, self.tL[1])
        self.bR = (self.bR[0]+offset_x, self.bR[1])
        self.top = self.tL[1]
        self.bottom = self.bR[1]
        self.left = self.tL[0]
        self.right = self.bR[0]
        self.width = self.bR[0]- self.tL[0]
        self.height = self.bR[1] - self.tL[1]
        self.center = ( int((self.right+self.left)*0.5),  int((self.top+self.bottom)*0.5) )

    def inflatY(self, offset_y):
        self.tL = (self.tL[0], self.tL[1]-offset_y)
        self.bR = (self.bR[0], self.bR[1]+offset_y)
        self.top = self.tL[1]
        self.bottom = self.bR[1]
        self.left = self.tL[0]
        self.right = self.bR[0]
        self.width = self.bR[0]- self.tL[0]
        self.height = self.bR[1] - self.tL[1]
        self.center = ( int((self.right+self.left)*0.5),  int((self.top+self.bottom)*0.5) )

    def inflat(self, offset):
        self.tL = (self.tL[0]-offset, self.tL[1]-offset)
        self.bR = (self.bR[0]+offset, self.bR[1]+offset)
        self.top = self.tL[1]
        self.bottom = self.bR[1]
        self.left = self.tL[0]
        self.right = self.bR[0]
        self.width = self.bR[0]- self.tL[0]
        self.height = self.bR[1] - self.tL[1]
        self.center = ( int((self.right+self.left)*0.5),  int((self.top+self.bottom)*0.5) )

    def rotate(self, degrees):
        rotated = []
        corners = [self.tL, self.tR, self.bR,  self.bL ]
        theta = math.radians(degrees)
        cos_theta, sin_theta = math.cos(theta), math.sin(theta)
        for point in corners:
            ox, oy = self.center[0], self.center[1]
            px, py = point
            qx = ox + cos_theta * (px - ox) - sin_theta * (py - oy)
            qy = oy + sin_theta * (px - ox) + cos_theta * (py - oy)
            rotated.append((int(qx), int(qy)))
        return rotated

    def point_in_rect(self, point):
        bool = False
        x, y = point

        if x >= self.left and x <= self.right and y >= self.top and y <= self.bottom:
            bool = True
        return bool


class CLine:
    def __init__(self, start_point, end_point):
        self.start = start_point
        self.end = end_point
        self.m = self.__get_slope(self.start, self.end)
        self.c = self.__get_c(self.m, self.start)

    def __get_slope(self, point1, point2):
        if point2[1] == point1[1]:
            slope = 0
        elif point2[0] == point1[0]:
            slope = -1
        else:
            slope = abs(point2[1] - point1[1]) / abs(point2[0] - point1[0])
        return slope

    def __get_c(self, slope, point_on_line):
        if slope is 0: #horz line
            c = point_on_line[1]
        elif slope is -1: #vert line
            c = point_on_line[0]
        else:
            c = int( point_on_line[1] - (slope * point_on_line[0]) )
        return c

    def is_horizontal(self):
        return True if self.m is 0 else False

    def is_vertical(self):
        return True if self.m is -1 else False

    def get_y_coord_on_line(self, x):
        if self.m is 0:
            pos_y = self.start[1]
        elif self.m is -1:
            pos_y = None
        else:
            pos_y = (self.m*x)+self.c
        return pos_y

    def get_x_coord_on_line(self, y):
        if self.m is 0:
            pos_x = None
        elif self.m is -1:
            pos_x = self.start[0]
        else:
            pos_x = int( (y-self.c)/self.m )
        return pos_x

    def get_line_intercept_point(self, line_input):
        m_input = self.__get_slope(line_input.start, line_input.end)
        c_input = self.__get_c(line_input.m, line_input.start)
        x = int( (c_input- self.c)/(self.m - m_input) )
        y = int( ((self.m*c_input) - (m_input*self.c)) / (self.m - m_input) )
        return x, y

    def get_perpendicular_distance(self, point):
        current_line = CLine(self.start, self.end)
        distance = 0
        if self.m == 0:
            point_online_x = point[0]
            point_online_y = current_line.get_y_coord_on_line(point_online_x)
            distance = math.hypot(point_online_x - point[0], point_online_y - point[1])
        elif self.m == -1:
            point_online_y = point[1]
            point_online_x = current_line.get_x_coord_on_line(point_online_y)
            distance = math.hypot(point_online_x - point[0], point_online_y - point[1])
        else:
            m2 = -self.m
            c2 = self.__get_c(m2, point)
            point_online_x = int( (c2 - self.c)/(self.m - m2) )
            point_online_y = current_line.get_y_coord_on_line(point_online_x)
            distance = math.hypot(point_online_x - point[0], point_online_y - point[1])
        return distance


class CPoint:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def pos(self):
        return self.x, self.y

    def rotate(self, angle_degree, ref_points):
        theta = math.radians(angle_degree)
        cos_theta, sin_theta = math.cos(theta), math.sin(theta)
        ox, oy = ref_points.pos()
        px, py = self.x, self.y
        qx = ox + cos_theta * (px - ox) - sin_theta * (py - oy)
        qy = oy + sin_theta * (px - ox) + cos_theta * (py - oy)
        return CPoint( int(qx), int(qy) )

    def translateX(self, offset_x):
        px = self.x + offset_x
        py = self.y
        return ( CPoint(px, py) )

    def translateY(self, offset_y):
        px = self.x
        py = self.y + offset_y
        return ( CPoint(px, py) )


class CHistogram:
    def __init__(self, img, mask=None):
        self.data = cv2.calcHist([img], [0], mask, [256], [0, 256])
        self.pixels_list = [self.data[x][0].astype(int) for x in range(256)]

    def get_total_pixel_in_range(self, start, end):
        fnd_list = self.pixels_list[start: end+1]
        return sum(fnd_list)

    def get_mode_bin(self):
        mode_value = max(self.pixels_list)
        mode_index = self.pixels_list.index(mode_value)
        return mode_index

    def get_bin_total_pixels(self, bin_number):
        return self.pixels_list[bin_number]

    def get_local_peak_bin(self , diff_threshold=1):
        local_peak = []
        for n in range(0, 256):
            if n == 0:
                pxl_count_next = self.pixels_list[n+1]
                pxl_count_current = self.pixels_list[n]
                if int(pxl_count_current - pxl_count_next) > diff_threshold:
                    local_peak.append(n)
            elif n == 255:
                pxl_count_previous = self.pixels_list[n - 1]
                pxl_count_current = self.pixels_list[n]
                if int(pxl_count_current - pxl_count_previous) > diff_threshold:
                    local_peak.append(n)
            else:
                pxl_count_previous = self.pixels_list[n - 1]
                pxl_count_current = self.pixels_list[n]
                pxl_count_next = self.pixels_list[n + 1]

                if int(pxl_count_current-pxl_count_previous) > diff_threshold and int(pxl_count_current-pxl_count_next) > diff_threshold:
                    local_peak.append(n)
        return local_peak

    def get_lowest_bin_with_minimum_coverage_of(self, coverage_percent):
        accumulate_pixels = 0
        total_pixels = sum(self.pixels_list)
        target_pixels = int(total_pixels * coverage_percent/100)
        for n in range(256):
            if accumulate_pixels >= target_pixels:
                break
            else:
                accumulate_pixels += self.pixels_list[n]
        return n

    def get_highest_bin_with_minimum_coverage_of(self, coverage_percent):
        accumulate_pixels = 0
        total_pixels = sum(self.pixels_list)
        target_pixels = int(total_pixels * coverage_percent/100)
        for n in reversed(range(256)):
            if accumulate_pixels >= target_pixels:
                break
            else:
                accumulate_pixels += self.pixels_list[n]
        return n












