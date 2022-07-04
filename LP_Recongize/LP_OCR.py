import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
import sys

MIN_AREA = 80
MIN_WIDTH, MIN_HEIGHT = 2, 8
MIN_RATIO, MAX_RATIO = 0.25, 1.0

MAX_DIAG_MULTIPLYER = 5
MAX_ANGLE_DIFF = 12.0
MAX_AREA_DIFF = 0.5
MAX_WIDTH_DIFF = 0.8
MAX_HEIGHT_DIFF = 0.2
MIN_N_MATCHED = 5

PLATE_WIDTH_PADDING = 1.3 # 1.3
PLATE_HEIGHT_PADDING = 1.5 # 1.5
MIN_PLATE_RATIO = 3
MAX_PLATE_RATIO = 10

def draw_by_cv2(img):
    resize_img = cv2.resize(img, (960, 540))

    cv2.imshow("img", resize_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def find_chars(contour_list, possible_contours):
    matched_result_idx = []

    for d1 in contour_list:
        matched_contours_idx = []

        for d2 in contour_list:
            if d1['idx'] == d2['idx']:
                continue
        
            dx = abs(d1['cx'] - d2['cx'])
            dy = abs(d1['cy'] - d2['cy'])

            diagonal_length1 = np.sqrt(d1['w']**2 + d1['h']**2)
            distance = np.linalg.norm(np.array([d1['cx'], d1['cy']]) - np.array([d2['cx'], d2['cy']]))

            if dx == 0:
                angle_diff= 90
            else:
                angle_diff = np.degrees(np.arctan(dy / dx))

            area_diff = abs(d1['w']*d1['h'] - d2['w']*d2['h']) / (d1['w']*d1['h'])
            width_diff = abs(d1['w'] - d2['w']) / d1['w']
            height_diff = abs(d1['h'] - d2['h']) / d1['h']

            cond = [
                distance < diagonal_length1 * MAX_DIAG_MULTIPLYER, 
                angle_diff < MAX_ANGLE_DIFF,
                area_diff < MAX_AREA_DIFF,
                width_diff < MAX_WIDTH_DIFF,
                height_diff < MAX_HEIGHT_DIFF]

            if False not in cond:
                matched_contours_idx.append(d2['idx'])

        matched_contours_idx.append(d1['idx'])

        if len(matched_contours_idx) < MIN_N_MATCHED:
            continue

        matched_result_idx.append(matched_contours_idx)
        unmatched_contour_idx = []

        for d4 in contour_list:
            if d4['idx'] not in matched_contours_idx:
                unmatched_contour_idx.append(d4['idx']) 

        unmatched_contour = np.take(possible_contours, unmatched_contour_idx)
        recursive_contour_list = find_chars(unmatched_contour, possible_contours)

        for idx in recursive_contour_list:
            matched_result_idx.append(idx)

        break
    
    return matched_result_idx

def find_plates(img, matched_result):
    plate_imgs = []
    plate_infos = []

    height, width = img.shape

    for i, matched_chars in enumerate(matched_result):
        sorted_chars = sorted(matched_chars, key=lambda x: x['cx'])

        plate_cx = (sorted_chars[0]['cx'] + sorted_chars[-1]['cx']) / 2
        plate_cy = (sorted_chars[0]['cy'] + sorted_chars[-1]['cy']) / 2
        
        plate_width = (sorted_chars[-1]['x'] + sorted_chars[-1]['w'] - sorted_chars[0]['x']) * PLATE_WIDTH_PADDING
        
        sum_height = 0
        for d in sorted_chars:
            sum_height += d['h']

        plate_height = int(sum_height / len(sorted_chars) * PLATE_HEIGHT_PADDING)
        
        triangle_height = sorted_chars[-1]['cy'] - sorted_chars[0]['cy']
        triangle_hypotenus = np.linalg.norm(
            np.array([sorted_chars[0]['cx'], sorted_chars[0]['cy']]) - 
            np.array([sorted_chars[-1]['cx'], sorted_chars[-1]['cy']])
        )
        
        angle = np.degrees(np.arcsin(triangle_height / triangle_hypotenus))
        
        rotation_matrix = cv2.getRotationMatrix2D(center=(plate_cx, plate_cy), angle=angle, scale=1.0)
        
        img_rotated = cv2.warpAffine(img, M=rotation_matrix, dsize=(width, height))
        
        img_cropped = cv2.getRectSubPix(
            img_rotated, 
            patchSize=(int(plate_width), int(plate_height)), 
            center=(int(plate_cx), int(plate_cy))
        )
        
        cond = [
            img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO,
            img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO > MAX_PLATE_RATIO
        ]
        if True in cond:
            continue
        
        plate_imgs.append(img_cropped)
        plate_infos.append({
            'x': int(plate_cx - plate_width / 2),
            'y': int(plate_cy - plate_height / 2),
            'w': int(plate_width),
            'h': int(plate_height)
        })

    return plate_imgs, plate_infos

def ocr(plate_imgs):
    longest_idx, longest_text = -1, 0
    plate_chars = []

    for i, plate_img in enumerate(plate_imgs):
        plate_img = cv2.resize(plate_img, dsize=(0, 0), fx=1.6, fy=1.6)
        _, plate_img = cv2.threshold(plate_img, thresh=0.0, maxval=255.0, type=cv2.THRESH_BINARY|cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(plate_img, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)

        plate_min_x, plate_min_y = plate_img.shape[1], plate_img.shape[0]
        plate_max_x, plate_max_y = 0, 0

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            area = w * h
            ratio = w / h

            cond = [
                area > MIN_AREA,
                w > MIN_WIDTH, 
                h > MIN_HEIGHT,
                MIN_RATIO < ratio < MAX_RATIO
            ]
            if False not in cond:
                if x < plate_min_x:
                    plate_min_x = x
                if y < plate_min_y:
                    plate_min_y = y
                if x + w > plate_max_x:
                    plate_max_x = x + w
                if y + h > plate_max_y:
                    plate_max_y = y + h

        img_result = plate_img[plate_min_y:plate_max_y, plate_min_x:plate_max_x]

        img_result = cv2.GaussianBlur(img_result, ksize=(3, 3), sigmaX=0)
        _, img_result = cv2.threshold(img_result, thresh=0.0, maxval=255.0, type=cv2.THRESH_BINARY|cv2.THRESH_OTSU)
        img_result = cv2.copyMakeBorder(img_result, top=10, bottom=10, left=10, right=10, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))

        pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
        chars = pytesseract.image_to_string(img_result, lang='kor', config='--psm 7 --oem 0')

        result_chars = ''
        has_digit = False

        for c in chars:
            if ord('가') <= ord(c) <= ord('힣') or c.isdigit():
                if c.isdigit():
                    has_digit = True
                result_chars += c
        
        plate_chars.append(result_chars)

        if has_digit and len(result_chars) > longest_text:
            longest_idx = i
        

    return longest_idx, plate_chars

def main(path, state, pos):
    img_ori = cv2.imread(path)
    img_h, img_w, img_c = img_ori.shape

    if(state == 1):
        dw = 1./img_w
        dh = 1./img_h

        cw = round(pos[2] / dw)
        ch = round(pos[3] / dh)

        cx = round(pos[0] / dw - cw/2)
        cy = round(pos[1] / dh - ch/2)
    else:
        cx, cy = 0, 0
        cw, ch = img_w, img_h

    # print("%d %d %d %d"%(cx, cy, cw, ch))
    img = img_ori[cy:cy+ch, cx:cx+cw]
    height, width, channel = img.shape

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_thresh = cv2.adaptiveThreshold(gray, maxValue=255.0, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresholdType=cv2.THRESH_BINARY_INV, blockSize=19, C=9)

    contours, _ = cv2.findContours(img_thresh, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)
    contours_dict = []
    temp_result = np.zeros((height, width, channel), dtype=np.uint8)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        #cv2.rectangle(temp_result, pt1=(x, y), pt2=(x+w, y+h), color=(255,255,255), thickness=2)
        contours_dict.append({
            'contour': contour,
            'x': x,
            'y': y,
            'w': w,
            'h': h,
            'cx': x + (w/2),
            'cy': y + (h/2)
        })

    #draw_by_cv2(temp_result)

    #######################
    possible_contours = []
    matched_result = []

    cnt = 0
    for d in contours_dict:
        area = d['w'] * d['h']
        ratio = d['w'] / d['h']

        cond = [
            area > MIN_AREA,
            d['w'] > MIN_WIDTH, d['h'] > MIN_HEIGHT,
            MIN_RATIO < ratio < MAX_RATIO
        ]
        if False not in cond:
            d['idx'] = cnt
            cnt += 1
            possible_contours.append(d)

    # for d in possible_contours:
    #     cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255, 255, 255), thickness=2)

    # draw_by_cv2(temp_result)
    result_idx = find_chars(possible_contours, possible_contours)

    for idx_list in result_idx:
        matched_result.append(np.take(possible_contours, idx_list))

    # temp_result = np.zeros((height, width, channel), dtype=np.uint8)

    # for r in matched_result:
    #     for d in r:
    #         cv2. rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255, 255, 255), thickness=2)
    # draw_by_cv2(temp_result)
    
    
    ########################
    plate_imgs, plate_infos = find_plates(img_thresh, matched_result)

    longest_idx, plate_chars = ocr(plate_imgs)

    info = plate_infos[longest_idx]
    chars = plate_chars[longest_idx]

    print(chars)

    # f = open('./LP.txt', 'w')
    # f.write(chars)
    # f.close()

    img_out = img_ori.copy()

    cv2.rectangle(img_out, pt1=(cx+info['x'], cy+info['y']), pt2=(cx+info['x']+info['w'], cy+info['y']+info['h']), color=(0, 0, 255), thickness=2)
    cv2.imwrite("LP.jpg", img_out)
    #plt.savefig('LicensePlate.jpg', bbox_inches='tight')
    #draw_by_cv2(img_out)

def pos_convert(pos):
    convert_pos = list()

    for i in range(len(pos)):
        convert_pos.append(float(pos[i]))
    
    return convert_pos


if __name__ == '__main__':
    if len(sys.argv) == 2:
        pos = [0, 0, 0, 0]
        main(sys.argv[1], 0, pos)
    elif len(sys.argv) == 6:
        pos = pos_convert(sys.argv[2:6])
        main(sys.argv[1], 1, pos)
    else:
        print("Argument Error")