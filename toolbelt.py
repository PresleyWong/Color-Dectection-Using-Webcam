import cv2
import time
import numpy as np
from scipy import ndimage
from skimage.feature import greycomatrix, greycoprops
from constants import *
from base_classes import *
import traceback
import sys


def skeletonizing(binary_img):
    skeleton = np.zeros(binary_img.shape,np.uint8)
    eroded = np.zeros(binary_img.shape,np.uint8)
    temp = np.zeros(binary_img.shape,np.uint8)

    thresh = binary_img.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))

    iters = 0
    while(True):
        cv2.erode(thresh, kernel, eroded)
        cv2.dilate(eroded, kernel, temp)
        cv2.subtract(thresh, temp, temp)
        cv2.bitwise_or(skeleton, temp, skeleton)
        thresh, eroded = eroded, thresh # Swap instead of copy

        iters += 1
        if cv2.countNonZero(thresh) == 0:
            return (skeleton, iters)

def get_crop_img(src, x, y, h, w, offset=0):
    output = src[max(0, y - offset):min(y + h + (2 * offset), src.shape[0] - 1),
             max(0, x - offset):min(x + w + (2 * offset), src.shape[1] - 1)]
    return output


def get_high_or_low_pass_threshold(binary_crop_img, gray_crop_img, target_percentage, highpass=True):
    # Find highpass pixel value that occupy at least 10% of total pixels
    array_number_pixels = []
    total_pixels = np.count_nonzero(binary_crop_img)
    occupy_percentage = 0

    if highpass:
        gray_crop_img[np.where(binary_crop_img == 0)] = 0
        test_img = gray_crop_img.copy()

        threshold_value = 0
        while occupy_percentage < target_percentage and np.count_nonzero(test_img) != 0:
            test_img[np.where(test_img == threshold_value)] = 0
            threshold_value = test_img.max()
            num_pixels = np.count_nonzero(test_img == threshold_value)
            accumulate_pixels = num_pixels + sum(array_number_pixels)
            occupy_percentage = accumulate_pixels / total_pixels
            array_number_pixels.append(num_pixels)
    else:
        gray_crop_img[np.where(binary_crop_img == 0)] = 255
        test_img = gray_crop_img.copy()

        threshold_value = 255
        while occupy_percentage < target_percentage and len(test_img)-np.count_nonzero(test_img) != 0:
            test_img[np.where(test_img == threshold_value)] = 255
            threshold_value = test_img.min()
            num_pixels = np.count_nonzero(test_img == threshold_value)
            accumulate_pixels = num_pixels + sum(array_number_pixels)
            occupy_percentage = accumulate_pixels / total_pixels
            array_number_pixels.append(num_pixels)
    return threshold_value


def get_glcm(patch):
    glcm = greycomatrix(patch, [5], [0], 256, symmetric=True, normed=True)
    diss = greycoprops(glcm, 'dissimilarity')[0, 0]
    corr = greycoprops(glcm, 'correlation')[0, 0]
    homo = greycoprops(glcm, 'homogeneity')[0, 0]
    eng = greycoprops(glcm, 'energy')[0, 0]
    cont = greycoprops(glcm, 'contrast')[0, 0]
    print("[INFO] dissimilarity: %f, correlation: %f, homogeneity: %f, energy: %f, contrast: %f," % (diss, corr, homo, eng, cont))
    return [diss, corr, homo, eng, cont]


def faster_bradley_threshold(src, threshold=75, window_r=3, invert=False):
    percentage = threshold / 80
    diam = 2*window_r + 3
    img = np.array(src).astype(np.float)
    means = ndimage.uniform_filter(img, diam)
    height, width = img.shape[:2]
    result = np.zeros((height,width), np.uint8)
    result[img >= percentage * means] = 255
    if invert:
        result = cv2.bitwise_not(result)
    return result


def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return edged


def find_edges_grey(frame, img, rect_roi, threshold, horizontal=True, forward=True, level=0.5, positive=True):
    fnd_edges = []
    x,y = rect_roi.tL
    w = rect_roi.width
    h = rect_roi.height

    if horizontal:
        target_level = int(w * level)
    else:
        target_level = int(h * level)

    roi_img = get_crop_img(img, x, y, w, h)
    if len(roi_img.shape) > 2:
        roi_img = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)

    height_img, width_img = roi_img.shape

    if horizontal:
        if forward:
            start = 0
            end = height_img-1
            order = 1
        else:
            start = height_img-1
            end = 0
            order = -1

        for m in range(start, end, order):
            a = int(roi_img[m][target_level])
            b = int(roi_img[m+order][target_level])

            if positive:
                diff = b - a
            else:
                diff = a - b

            if diff >= threshold:
                count = 0

                for n in range(0, width_img-1):
                    c = roi_img[m][n]
                    d = roi_img[m+order][n]
                    if positive:
                        diff_adj = d - c
                    else:
                        diff_adj = c - d
                    if diff_adj >= threshold:
                        count += 1

                if count >= int(w*0.3):
                    edge_y = m + y
                    edge_x = target_level + x
                    fnd_edges.append((edge_x, edge_y))
    else:
        if forward:
            start = 0
            end = width_img - 1
            order = 1
        else:
            start = width_img - 1
            end = 0
            order = -1

        for m in range(start, end, order):
            a = int(roi_img[target_level][m])
            b = int(roi_img[target_level][m + order])

            if positive:
                diff = b - a
            else:
                diff = a - b

            if diff >= threshold:
                count = 0

                for n in range(0, height_img - 1):
                    c = roi_img[n][m]
                    d = roi_img[n + order][m]
                    if positive:
                        diff_adj = d - c
                    else:
                        diff_adj = c - d
                    if diff_adj >= threshold:
                        count += 1

                if count >= int(h * 0.1):
                    edge_x = m + x
                    edge_y = target_level + y
                    fnd_edges.append((edge_x, edge_y))

    return fnd_edges


def find_edges(binary_img, rect_roi, min_connected_pixel=10, horizontal=True, forward=True, level=0.5):
    fnd_edges = []
    x,y = rect_roi.tL
    w = rect_roi.width
    h = rect_roi.height

    if horizontal:
        target_level = int(w * level)
    else:
        target_level = int(h * level)

    roi_img = get_crop_img(binary_img, x, y, w, h)
    height_img, width_img = roi_img.shape

    if height_img < 1 or width_img < 1:
        imshow('roi_img', roi_img, 1)
        print("find edges failed")
        return fnd_edges

    if horizontal:
        if forward:
            start = 0
            end = height_img-1
            order = 1
        else:
            start = height_img-1
            end = 0
            order = -1

        for m in range(start, end, order):
            pixel_value = int(roi_img[m][target_level])
            if pixel_value == 255:
                edge_y = m + y
                edge_x = target_level + x
                fnd_edges.append((edge_x, edge_y))
    else:
        if forward:
            start = 0
            end = width_img - 1
            order = 1
        else:
            start = width_img - 1
            end = 0
            order = -1

        for m in range(start, end, order):
            pixel_value = int(roi_img[target_level][m])

            if pixel_value == 255:
                edge_x = m + x
                edge_y = target_level + y
                fnd_edges.append((edge_x, edge_y))

    return fnd_edges


def find_distance(pnt1, pnt2):
    return math.hypot(pnt2[0] - pnt1[0], pnt2[1] - pnt1[1])


def sort_point(point_list, x_axis=True):
    if x_axis:
        sorted_list = sorted(point_list, key=lambda k: (k[0], k[1]))
    else:
        sorted_list = sorted(point_list, key=lambda k: (k[1], k[0]))
    return sorted_list

def sort_rect(rect_list, x_axis=True):
    if x_axis:
        sorted_list = sorted(rect_list, key=lambda k: (k.left, k.top))
    else:
        sorted_list = sorted(rect_list, key=lambda k: (k.top, k.left))
    return sorted_list


def get_sub_roi_location(src_height, src_width, sub_roi_height, sub_roi_width):
    top_left_point = [0, 0]
    bot_right_point = [0, 0]
    array = []

    while (top_left_point[1] + sub_roi_height <= src_height):
        if (top_left_point[0] + sub_roi_width <= src_width):
            bot_right_point[0] = top_left_point[0] + sub_roi_width
            bot_right_point[1] = top_left_point[1] + sub_roi_height

            if (top_left_point[0] + 2 * sub_roi_width > src_width):
                bot_right_point[0] += src_width - (top_left_point[0] + sub_roi_width)
            if (top_left_point[1] + 2 * sub_roi_height > src_height):
                bot_right_point[1] += src_height - (top_left_point[1] + sub_roi_height)

            coordinate = [top_left_point, bot_right_point]
            array.append(coordinate)
            top_left_point[0] += sub_roi_width
        else:
            top_left_point[0] = 0
            top_left_point[1] += sub_roi_height
    return array


def get_sub_roi_dimension(src_height, src_width , column, row):
    sub_roi_height = int(src_height / row)
    sub_roi_width = int(src_width / column)
    return sub_roi_height, sub_roi_width


def remove_small_connected_pixels(binary_img, min_size):
    # find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(binary_img, connectivity=8)
    # connectedComponentswithStats yields every seperated component with information on each of them, such as size
    # the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
    sizes = stats[1:, -1]
    nb_components = nb_components - 1
    # min_size = minimum size of particles we want to keep (number of pixels)
    output_img = np.zeros(output.shape, np.uint8)
    # for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            output_img[output == i + 1] = 255
    return output_img


def find_object_angle(frame, binary_img):
    mean_angle = 0
    angle_array = []
    line_array = []
    lines = cv2.HoughLines(binary_img, 1, np.pi / 180, 100)
    if lines is not None:
        for ln in lines[0:2]:
            rho, theta = ln[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 0 * (-b))
            y1 = int(y0 + 0 * (a))
            x2 = int(x0 - 3000 * (-b))
            y2 = int(y0 - 3000 * (a))
            if x2-x1 == 0:
                m = 0
            else:
                m = (y2 - y1) / (x2 - x1)
            c = int(y1 - (m * x1))
            start_x = 0
            start_y = c
            end_x = binary_img.shape[1]
            end_y = int((m * end_x) + c)
            angle = math.degrees(math.atan2(y2-y1, x2-x1))
            angle_array.append(angle)
            line = CLine((x1, y1), (x2, y2))
            line_array.append(line)
            cv2.line(frame, (x1,y1), (x2,y2), YELLOW, thickness=2, lineType=8)
        mean_angle = sum(angle_array)/len(angle_array)
    return mean_angle, line_array


def get_averaging_point(edges_array):
    point = None
    if edges_array:
        avg_y = int(sum(y for x, y in edges_array) / len(edges_array))
        avg_x = int(sum(x for x, y in edges_array) / len(edges_array))
        point = (avg_x, avg_y)
    return point


def convert_input_to_dictionary(pocket_edge_list):
    sorted_list = sorted(pocket_edge_list, key=lambda k: (k[0][1], k[0][0]))
    row_count = 0
    row = []
    new_dict = {}

    if len(pocket_edge_list)>1:
        for n in range(0, len(sorted_list)-1):
            ref1 = sorted_list[n]
            ref2 = sorted_list[n+1]

            if not is_new_row(ref1, ref2):
                row.append(ref1)
            else:
                row.append(ref1)
                row = sorted(row, key=lambda k: (k[0][0], k[0][1]))
                new_dict[row_count] = row
                row_count += 1
                row = []

            if n == len(sorted_list)-2:
                row.append(ref2)
                row = sorted(row, key=lambda k: (k[0][0], k[0][1]))
                new_dict[row_count] = row

    elif len(pocket_edge_list)==1:
        row.append(pocket_edge_list[0])
        new_dict[row_count] = row

    return new_dict


def is_new_row(set_edge1, set_edge2):
    bool = False
    rect1 = CRect((set_edge1[LFT][0], set_edge1[TOP][1]), (set_edge1[RGH][0], set_edge1[BOT][1]))
    rect2 = CRect((set_edge2[LFT][0], set_edge2[TOP][1]), (set_edge2[RGH][0], set_edge2[BOT][1]))
    if rect2.center[1] > rect1.bottom or rect2.center[1] < rect1.top:
        bool = True
    return bool

def combine_contour_rect_iteratively(canny_img, iteration):
    mask = np.zeros(canny_img.shape, np.uint8)
    binary_img = canny_img.copy()

    for n in range(iteration):
        contours = cv2.findContours(binary_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        for cnt in contours:
            cnt_x, cnt_y, cnt_w, cnt_h = cv2.boundingRect(cnt)
            rect_contour = CRect((cnt_x, cnt_y), (cnt_x + cnt_w, cnt_y + cnt_h))
            cv2.rectangle(mask, rect_contour.tL, rect_contour.bR, WHITE, -1)
        binary_img = mask
    return binary_img

def refine_pocket_mask_area(canny_img, rough_pocket_mask, estimate_pocket_height,estimate_pocket_width):
    contours = cv2.findContours(canny_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    for cnt in contours:
        cnt_x, cnt_y, cnt_w, cnt_h = cv2.boundingRect(cnt)
        rect_contour = CRect((cnt_x, cnt_y), (cnt_x + cnt_w, cnt_y + cnt_h))
        if cnt_h < estimate_pocket_height*0.3 or cnt_w < estimate_pocket_width*0.3:
            cv2.rectangle(rough_pocket_mask, rect_contour.tL, rect_contour.bR, BLACK, -1)
    return rough_pocket_mask

def find_inner_or_outer_pocket_four_side_edges(pocket_rect, canny_img, enhanced_canny_img, outer=True):
    rect_dict = {}
    edge_dict = {}

    if outer:
        dir_dict = { TOP: True, BOT: False, LFT: True, RGH: False}
        # dir_dict[TOP] = True
        # dir_dict[BOT] = False
        # dir_dict[LFT] = True
        # dir_dict[RGH] = False
    else:
        dir_dict = {TOP: False, BOT: True, LFT: False, RGH: True}
        # dir_dict[TOP] = False
        # dir_dict[BOT] = True
        # dir_dict[LFT] = False
        # dir_dict[RGH] = True

    for side in range(4):
        offset_x = int(pocket_rect.width / 3)
        offset_y = int(pocket_rect.height / 3)
        sm = 3

        if side == TOP:
            top_left = (pocket_rect.left + offset_x, pocket_rect.top - sm)
            bot_right = (pocket_rect.right - offset_x, pocket_rect.top + offset_y)
        elif side == BOT:
            top_left = (pocket_rect.left + offset_x, pocket_rect.bottom - offset_y)
            bot_right = (pocket_rect.right - offset_x, pocket_rect.bottom + sm)
        elif side == LFT:
            top_left = (pocket_rect.left - sm , pocket_rect.top + offset_y)
            bot_right = (pocket_rect.left + offset_x, pocket_rect.bottom - offset_y)
        else:
            top_left = (pocket_rect.right - offset_x, pocket_rect.top + offset_y)
            bot_right = (pocket_rect.right + sm, pocket_rect.bottom - offset_y)

        rect = CRect(top_left, bot_right)
        rect.validate(canny_img)
        rect_dict[side] = rect

    for side in [LFT, RGH]:
        rect = rect_dict[side]
        dir = dir_dict[side]
        edges = find_edges(canny_img, rect, horizontal=False, forward=dir)
        if edges:
            edge_dict[side] = edges[0]

    edge_dict = cross_check_pocket_edges(canny_img, enhanced_canny_img, edge_dict, rect_dict, dir_dict, [LFT, RGH])

    if len(edge_dict) == 2:
        center_pnt = (edge_dict[LFT][0] + edge_dict[RGH][0]) * 0.5
        center_current = (rect_dict[TOP].bR[0] + rect_dict[TOP].tL[0]) * 0.5
        shift = int(center_pnt - center_current)
        rect_dict[TOP] = rect_dict[TOP].translateX(shift)
        rect_dict[BOT] = rect_dict[BOT].translateX(shift)

    for side in [TOP, BOT]:
        rect = rect_dict[side]
        dir = dir_dict[side]
        edges = find_edges(canny_img, rect, horizontal=True, forward=dir)
        if edges:
            edge_dict[side] = edges[0]

    edge_dict = cross_check_pocket_edges(canny_img, enhanced_canny_img, edge_dict, rect_dict, dir_dict, [TOP, BOT])
    return edge_dict


def cross_check_pocket_edges(canny_img, enhanced_canny_img, edge_dict, rect_dict, dir_dict, side_list):
    for side in side_list:
        if not side in edge_dict:
            rect = rect_dict[side]
            if side == LFT or side == RGH:
                horz = False
            else:
                horz = True
            edges = find_edges(enhanced_canny_img, rect, horizontal=horz, forward=dir_dict[side])

            if edges:
                edge_dict[side] = edges[0]
            else:
                fnd_edge = fine_search_edges(canny_img, rect, horizontal=horz, forward=dir_dict[side], number_roi=2)
                if fnd_edge:
                    edge_dict[side] = fnd_edge

    return edge_dict



def fine_search_edges(img, rect_roi, horizontal= True, forward=True, number_roi=1):
    fnd_edges = []
    x, y = rect_roi.tL
    w = rect_roi.width
    h = rect_roi.height

    if horizontal:
        sub_roi_size = int(w / number_roi)
        for lap in range(0, number_roi):
            top_left_x = x + (lap * sub_roi_size)
            top_left_y = y
            bot_right_y = top_left_y + h
            if lap is number_roi - 1:
                bot_right_x = rect_roi.right
            else:
                bot_right_x = top_left_x + sub_roi_size

            sub_rect = CRect((top_left_x, top_left_y), (bot_right_x, bot_right_y))
            edges = find_edges(img, sub_rect, forward=forward, min_connected_pixel=(sub_rect.width * 0.4))

            if len(edges) > 0:
                fnd_edges.append(edges[0])
    else:
        sub_roi_size = int(h / number_roi)
        for lap in range(0, number_roi):
            top_left_x = x
            top_left_y = y + (lap * sub_roi_size)

            bot_right_x = top_left_x + w

            if lap is number_roi - 1:
                bot_right_y = rect_roi.bottom
            else:
                bot_right_y = top_left_y + sub_roi_size

            sub_rect = CRect((top_left_x, top_left_y), (bot_right_x, bot_right_y))
            edges = find_edges(img, sub_rect, horizontal=False, forward=forward, min_connected_pixel=(sub_rect.height * 0.4))

            if len(edges) > 0:
                fnd_edges.append(edges[0])

    output_edge = None
    if len(fnd_edges) > 0:

        if horizontal:
            pos_x = rect_roi.center[0]
            pos_y = int( sum([k[1] for k in fnd_edges]) / len(fnd_edges) )
        else:
            pos_y = rect_roi.center[1]
            pos_x = int( sum([k[0] for k in fnd_edges]) / len(fnd_edges) )

        output_edge = (pos_x, pos_y)
        print("[INFO] pos_x = ", pos_x)
        print("[INFO] pos_y = ", pos_y)
    return output_edge


def time_usage(func):
    def wrapper(*args, **kwargs):
        beg_ts = time.time()*1000
        retval = func(*args, **kwargs)
        end_ts = time.time()*1000
        print("Function %s - elapsed time: %.2f (ms)" % (func.__name__, end_ts - beg_ts))
        return retval
    return wrapper


def safe_run(func):
    def wrapper(*args, **kwargs):
        try:
           return func(*args, **kwargs)
        except Exception:
            traceback.print_exc()
            return None
    return wrapper


def imshow(window_string , image_source, time=None):
    cv2.namedWindow(window_string, cv2.WINDOW_NORMAL)
    cv2.imshow(window_string, image_source)
    cv2.waitKey(time) if time is not None else None


def enhance(img):
   YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
   # cv2.imshow("YCrCb", YCrCb)
   #enhancement of colors from CLAHE for YCrCb images
   YCrCb_planes = cv2.split(YCrCb)
   clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
   YCrCb_planes[0] = clahe.apply(YCrCb_planes[0])
   YCrCb_planes[1] = clahe.apply(YCrCb_planes[1])
   YCrCb_planes[2] = clahe.apply(YCrCb_planes[2])
   YCrCb = cv2.merge(YCrCb_planes)
   bgr1 = cv2.cvtColor(YCrCb, cv2.COLOR_YCrCb2BGR)
   return bgr1

def enhance2(img):
   # converting image to LAB space
   lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
   # cv2.imshow("lab", lab)
   # enhancement of colors from CLAHE for LAB images
   lab_planes = cv2.split(lab)
   clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
   lab_planes[0] = clahe.apply(lab_planes[0])
   lab = cv2.merge(lab_planes)
   bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
   return bgr


def enhance_gray_img(gray_img, clip_limit = 2.0):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    process_img = clahe.apply(gray_img)
    return process_img


def increase_contrast_img(gray_img, contrast=0, brightness=0):
    img = np.int16(gray_img)
    img = img * (contrast / 127 + 1) - contrast + brightness
    img = np.clip(img, 0, 255)
    img = np.uint8(img)
    return img


def is_linux_os():
    if 'linux' in sys.platform.lower():
        return True
    else:
        return False

def slash():
    if is_linux_os():
        slash = "/"
    else:
        slash = "\\"
    return slash


def realign_img(train_img, insp_img, insp_img_rgb):
    im2 = insp_img_rgb.copy()

    # Convert images to grayscale
    im1_gray = train_img.copy()
    im2_gray = insp_img.copy()

    # Find size of image1
    sz = im1_gray.shape

    # Define the motion model
    warp_mode = cv2.MOTION_TRANSLATION

    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Specify the number of iterations.
    number_of_iterations = 5000

    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-10

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv2.findTransformECC(im1_gray, im2_gray, warp_matrix, warp_mode, criteria)

    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        # Use warpPerspective for Homography
        im2_aligned = cv2.warpPerspective(im2, warp_matrix, (sz[1], sz[0]),
                                          flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else:
        # Use warpAffine for Translation, Euclidean and Affine
        im2_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        im2_aligned = cv2.cvtColor(im2_aligned, cv2.COLOR_BGR2GRAY)
    return im2_aligned




def nothing():
    pass

def create_track_bar(input_img, items_dict, window_string):
    dict_value = {}
    cv2.namedWindow(window_string, cv2.WINDOW_NORMAL)
    for item in items_dict:
        cv2.createTrackbar(item.get("name"), window_string, item.get("min"), item.get("max"), nothing)
        dict_value[item.get("name")] = cv2.getTrackbarPos(item.get("name"), window_string)
    imshow(window_string, input_img, 1)

def create_threshold_track_bar(src):
    window_str = "Threshold Trackbar"
    cv2.namedWindow(window_str, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("Threshold", window_str, 0, 255, nothing)
    threshold = cv2.getTrackbarPos("Threshold", window_str)
    output = cv2.threshold(src, threshold, 255, cv2.THRESH_BINARY)[1]
    imshow(window_str, output, 1)

def create_contrast_track_bar(src):
    window_str = "Contrast Trackbar"
    cv2.namedWindow(window_str, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("Contrast", window_str, 1, 255, nothing)
    cv2.createTrackbar("Brightness", window_str, 0, 255, nothing)
    cv2.createTrackbar("Threshold", window_str, 1, 255, nothing)
    c = cv2.getTrackbarPos("Contrast", window_str)
    b = cv2.getTrackbarPos("Brightness", window_str)
    t = cv2.getTrackbarPos("Threshold", window_str)
    output = increase_contrast_img(src, contrast=c, brightness=b, threshold=t)
    create_threshold_track_bar(output)
    imshow(window_str, output)

def threshold(img, lowThresholdRatio=0.05, highThresholdRatio=0.09):
    highThreshold = img.max() * highThresholdRatio
    lowThreshold = highThreshold * lowThresholdRatio

    M, N = img.shape
    res = np.zeros((M, N), dtype=np.int32)

    weak = np.int32(25)
    strong = np.int32(255)

    strong_i, strong_j = np.where(img >= highThreshold)
    zeros_i, zeros_j = np.where(img < lowThreshold)

    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    return (res, weak, strong)
