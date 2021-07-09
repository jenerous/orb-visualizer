#!/usr/bin/env python
# coding: utf-8

import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import math
import json
from time import sleep



# initial threshold for FAST feature (difference to center point)
iniThFast = 20
# reduce threshold for FAST, if not enough feature points were found to this
minThFast = 5

# original patch size for rotation estimation
PATCH_SIZE = 31
HALF_PATCH_SIZE = 15

# how wide shall the window image be
window_width = 1500

# initialize the fast detector, will be used later
fast = cv2.FastFeatureDetector_create(iniThFast, True)


pattern = json.load(open('pattern.json'))
# https://github.com/raulmur/ORB_SLAM2/blob/master/src/ORBextractor.cc#L150

modes = ["fast", "full", "pattern"]

def limit(val, lower, upper):
    # clip given value to lower or upper limit
    return min(upper, max(lower, val))

def pixel_circle(r):
    # find out, which points belong to a pixel circle around a certain point
    d = round(math.pi - (2 * r))
    x = 0
    y = r

    cpoints = []
    while x <= y:
        cpoints.append((x, -y))
        cpoints.append((y, -x))
        cpoints.append((y, x))
        cpoints.append((x, y))
        cpoints.append((-x, y))
        cpoints.append((-y, x))
        cpoints.append((-y, -x))
        cpoints.append((-x, -y))

        if d < 0:
            d += (math.pi * x) + (math.pi * 2)
        else:
            d += math.pi * (x - y) + (math.pi * 3)
            y -= 1
        x += 1
    return list(set(cpoints))

def calc_umax():
    # This relates to https://github.com/raulmur/ORB_SLAM2/blob/f2e6f51cdc8d067655d90a78c06261378e07e8f3/src/ORBextractor.cc#L452
    # This is for orientation
    # pre-compute the end of a row in a circular patch
    umax = [0] * (HALF_PATCH_SIZE + 1)
    vmax = int(np.floor(HALF_PATCH_SIZE * np.sqrt(2) / 2 + 1))

    vmin = int(np.ceil(HALF_PATCH_SIZE * np.sqrt(2) / 2))


    hp2 = HALF_PATCH_SIZE*HALF_PATCH_SIZE;
    for v in range(vmax + 1):
        umax[v] = int(np.round(np.sqrt(hp2 - v * v)))

    # Make sure we are symmetric
    v0 = 0
    for v in range(HALF_PATCH_SIZE, vmin-1, -1):
        while umax[v0] == umax[v0 + 1]:
            v0 += 1
        umax[v] = v0
        v0 += 1

    print('umax:', umax)
    return umax

def IC_Angle(image, pt,  u_max):
    # this relates to https://github.com/raulmur/ORB_SLAM2/blob/master/src/ORBextractor.cc#L77
    if image.ndim > 2:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    cpx = int(round(pt[1]))
    cpy = int(round(pt[0]))
    print('cpx/y/val', cpx, cpy, image[cpy, cpx])
    m_01 = int(0)
    # Treat the center line differently, v=0
    m_10 = sum([u * image[cpy, cpx + u] for u in range(-HALF_PATCH_SIZE, HALF_PATCH_SIZE + 1)])
    m_00 = sum([image[cpy, cpx + u] for u in range(-HALF_PATCH_SIZE, HALF_PATCH_SIZE + 1)])

    # Go line by line in the circuI853lar patch
    for v in range(1, HALF_PATCH_SIZE + 1):
        # Proceed over the two lines
        v_sum = 0;
        d = u_max[v];
        for u in range(-d, d + 1):
            val_plus = int(image[cpy + v, cpx + u])
            val_minus = int(image[cpy - v, cpx + u])
            v_sum += (val_plus - val_minus)
            m_10 += u * (val_plus + val_minus)
            m_00 += val_plus + val_minus
        m_01 += v * v_sum
    # print('m_01, m_10, m_00', m_01, m_10, m_00)
    angle = cv2.fastAtan2(m_01, m_10)

    if m_00 == 0 or not m_00:
        centerpoint_x = 0
        centerpoint_y = 0
    else:
        centerpoint_x = int(m_10/m_00)
        centerpoint_y = int(m_01/m_00)


    return angle, centerpoint_x + HALF_PATCH_SIZE, centerpoint_y + HALF_PATCH_SIZE

def put_text_on_enlarged(text, x, y, thickness=1):
    # a wrapper for positioning text in the upscaled version of the image
    global overlay, canvas, resized
    textsize = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=text_scale, thickness=thickness)[0]

    xshift = int(((canvas.shape[1] - ls * resized.shape[1] + x * f)  + (canvas.shape[1] - ls * resized.shape[1] + (x + 1) * f)) / 2)
    xshift -= textsize[0] // 2

    yshift = int(y * f + f / 2)
    yshift += textsize[1] // 2

    overlay = cv2.putText(
        overlay,
        text,
        (xshift , yshift),
        cv2.FONT_HERSHEY_COMPLEX,
        text_scale,
        (255),
        thickness=thickness
    )

# start of main
source = ""

while source != "q":
    source = input("Filepath or camera index: ")

    try:
        source = int(source)
        in_mode = 'live'
        webcam = cv2.VideoCapture(0)
        ret, video_frame = webcam.read()
        video_frame = cv2.flip(video_frame, 1)
        fast_ex = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)
        break
    except:
        if os.path.exists(source.strip()):
            in_mode = 'file'
            fast_ex = cv2.imread(source.strip(), cv2.IMREAD_GRAYSCALE)
            break
        else:
            print("Could not find given path or Camera Device for {}".format(source))
            exit()


center = (fast_ex.shape[0] // 2 , fast_ex.shape[1] // 2)
mode = modes[0]
umax = calc_umax()

while True:
    if in_mode == 'live':
        ret, video_frame = webcam.read()
        video_frame = cv2.flip(video_frame, 1)
        fast_ex = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)

    # circle radius
    r = HALF_PATCH_SIZE

    # calculating the text scale on base of radius. This was just interpolated with two examples
    text_scale = -0.04*r+0.87

    # how many pixels to pad around cirle
    padding = 2

    # the representation of the cropped area shall be *scale* times larger than the original height
    # keeping cropped separate, to later iterate over it
    scale = 1.2
    cropped = fast_ex[center[0]-r-padding:center[0]+r+padding+1,
                      center[1]-r-padding:center[1]+r+padding+1]
    resized = cv2.resize(cropped, (int(fast_ex.shape[0] *scale), int(fast_ex.shape[0]*scale)), interpolation=cv2.INTER_NEAREST)
    vKeysCell = fast.detect(fast_ex[center[0]-r-padding:center[0]+r+padding+1,
                                    center[1]-r-padding:center[1]+r+padding+1])

    # create a new canvas, to paste everything into
    canvas = np.ndarray((resized.shape[0], fast_ex.shape[1]+20+resized.shape[1]))
    canvas.fill(255)

    # where to paste the original image
    paste_x1 = 0
    paste_x2 = fast_ex.shape[1]

    paste_y1 = int(canvas.shape[0] / 2 - fast_ex.shape[0] / 2)
    paste_y2 = int(canvas.shape[0] / 2 - fast_ex.shape[0] / 2 + fast_ex.shape[0])

    # paste original image
    canvas[paste_y1: paste_y2, paste_x1:paste_x2] = fast_ex

    # paste resized crop
    canvas[:, -resized.shape[1]:] = resized

    # scale up everything to make lines smoother
    ls = int(np.ceil(window_width/canvas.shape[1]))
    canvas = cv2.resize(canvas, (0, 0), fx=ls, fy=ls, interpolation=cv2.INTER_NEAREST)

    # pasting things into an overlay, to later increase contrast (black & white)
    overlay = np.ndarray(canvas.shape)

    # use 128 to indicate emtpy spaces later
    overlay.fill(128)

    # line from rectangle to top left corner of crop
    overlay = cv2.line(
        overlay,
        (ls * (paste_x1 + center[1]+r+padding), ls * (paste_y1 + center[0]-r-padding)),
        (canvas.shape[1] - ls * resized.shape[1], 0), (255),
        thickness=2
    )

    # line from rectangle to bottom left corner of crop
    overlay = cv2.line(
        overlay,
        (ls * (paste_x1 + center[1]+r+padding+1), ls * (paste_y1 + center[0]+r+padding+1)),
        (canvas.shape[1]- ls * resized.shape[1], canvas.shape[0]), (255),
        thickness=2
    )

    # rectangle to indicate crop in original image
    overlay = cv2.rectangle(
        overlay,
        (ls * (paste_x1 + center[1]-r-padding), ls * (paste_y1 + center[0]-r-padding)),
        (ls * (paste_x1 + center[1]+r+padding+1), ls * (paste_y1 + center[0]+r+padding+1)), (255),
        thickness=2
    )

    # scale factor from original crop to resized version, after scaling up everything
    f = (resized.shape[0]) / cropped.shape[0] * ls
    pc = pixel_circle(r)

    # create vertical lines
    for cx in range(cropped.shape[1]):
        xshift = int(canvas.shape[1] - ls * resized.shape[1] + cx * f )
        overlay = cv2.line(
            overlay,
            (xshift, 0),
            (xshift, canvas.shape[0]),
            255
        )

    # create horizontal lines
    for cy in range(cropped.shape[0]):
        overlay = cv2.line(
            overlay,
            (canvas.shape[1] - ls * resized.shape[1], int((1+cy) * f)),
            (canvas.shape[1], int((1+cy) * f)),
            255
        )

    # outer circle
    overlay = cv2.circle(
        overlay,
        (int(canvas.shape[1] - ls * resized.shape[1] + cropped.shape[1] / 2 * f ), int(cropped.shape[0] / 2 * f)),
        int((r + 0.6) * f),
        255
    )

    # inner circle
    overlay = cv2.circle(
        overlay,
        (int(canvas.shape[1] - ls * resized.shape[1] + cropped.shape[1] / 2 * f ), int(cropped.shape[0] / 2 * f)),
        int((r - 0.55) * f),
        255
    )

    if mode == "full":

        # circle through all points of the circle and insert the values
        for cy in range(-r, r + 1):
            yp = [p[0] for p in pc if p[1] == cy]
            yshift = cy+r+padding
            for cx in range(min(yp), max(yp)+1):
                # calculating center of upscaled pixels, with respect to the text size
                thick = 1 # 2 if cy == 0 and cx == 0 else 1
                textsize = cv2.getTextSize(str(cropped[cy+r+padding, cx+r+padding]), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=text_scale, thickness=thick)[0]

                xshift = int(((canvas.shape[1] - ls * resized.shape[1] + (cx+r+padding) * f)  + (canvas.shape[1] - ls * resized.shape[1] + (cx + r + padding + 1) * f)) / 2)
                xshift -= textsize[0] // 2

                yshift = int((cy+r+padding) * f + f / 2)
                yshift += textsize[1] // 2

                overlay = cv2.putText(
                    overlay,
                    str(cropped[cy+r+padding, cx+r+padding]),
                    (xshift , yshift),
                    cv2.FONT_HERSHEY_COMPLEX,
                    text_scale,
                    (255),
                    thickness=thick
                )
                if cy == 0 and cx == 0:
                    overlay = cv2.rectangle(
                        overlay,
                        (overlay.shape[1] - int((r+1+padding) * f + 2), int((r+padding) * f)),
                        (overlay.shape[1] - int((r+padding) * f - 1), int((r+1+padding) * f)),
                        (255),
                        2
                    )

    elif mode == "fast":
        # show which pixels would count into the FAST feature detection
        # put values of pixels into cropped / resized image
        for cx in range(cropped.shape[1]):
            for cy in range(cropped.shape[0]):
                if (cx-r-padding, cy-r-padding) in pc or (cx-r-padding == 0 and cy-r-padding == 0):
                    put_text_on_enlarged(str(cropped[cy, cx]), cx, cy)
                    if (cx-r-padding == 0 and cy-r-padding == 0):
                        # add info to point in the center
                        put_text_on_enlarged("[p]", cx+0.75, cy+0.25,thickness=2)


        nb_angle = 2 * np.pi / len(pc)
        r_plus = 1.15
        for nb, nba in enumerate(np.arange(0.0, 2 * np.pi, nb_angle)):
            textsize = cv2.getTextSize('[{}]'.format(nb + 1), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=text_scale, thickness=2)[0]
            nba_x = int(np.sin(nba) * (r + r_plus) * f - overlay.shape[0] / 2 + overlay.shape[1] - textsize[0] / 2)
            nba_y = int(-np.cos(nba) * (r + r_plus) * f + overlay.shape[0] / 2 + textsize[1] / 2)

            overlay = cv2.putText(
                overlay,
                '[{}]'.format(nb + 1),
                (nba_x, nba_y),
                cv2.FONT_HERSHEY_COMPLEX,
                text_scale,
                (255),
                thickness=2
            )


    elif mode == "pattern":
        # show the first x pattern overlayed
        pmax = 10
        descriptor = []
        d_count = 0
        for a_x, a_y, b_x, b_y in pattern:
            if a_x > padding + r or a_x < - padding - r or \
                      a_y > padding + r or a_y < - padding - r or \
                      b_x > padding + r or b_x < - padding - r or \
                      b_y > padding + r or b_y < - padding - r:
                continue

            if d_count > pmax:
                break

            if fast_ex[center[0]+ a_y, center[1]+ a_x] < fast_ex[center[0]+ b_y, center[1]+ b_x]:
                descriptor.append("1")
                put_text_on_enlarged("{}a".format(d_count), a_x+padding+r, a_y+padding+r)
                put_text_on_enlarged("{}b".format(d_count), b_x+padding+r, b_y+padding+r, thickness=2)

            else:
                descriptor.append("0")
                # if fast_ex[center[0]+ a_y, center[1]+ a_x] == fast_ex[center[0]+ b_y, center[1]+ b_x]:
                #     put_text_on_enlarged("{}a".format(d_count), a_x+padding+r, a_y+padding+r)
                # else:
                put_text_on_enlarged("{}a".format(d_count), a_x+padding+r, a_y+padding+r, thickness=2)
                put_text_on_enlarged("{}b".format(d_count), b_x+padding+r, b_y+padding+r)

            d_count += 1

        # Also print this onto the image
        overlay = cv2.putText(
            overlay,
            "Descriptor: " + " | ".join(descriptor) + " ...",
            (20, overlay.shape[0] - 20),
            cv2.FONT_HERSHEY_COMPLEX,
            text_scale,
            (255),
            thickness=1
        )
        print("Descriptor: " + " | ".join(descriptor) + " ...")


    # turning overlay into white (255) pixels, where the underlying image is darker
    # and into black (0) pixels, where the underlying image is lighter
    overlay[overlay != 128] = np.where(canvas > 150, 50, 200)[overlay != 128]

    # pasting in the overlay
    canvas[overlay != 128] = overlay[overlay != 128]

    # calculate the momentums and angle of a circular patch
    a, cpx, cpy = IC_Angle(fast_ex, center, umax)

    print("returned", a, cpx, cpy, np.sin(np.deg2rad(a)), np.cos(np.deg2rad(a)))

    # initialize an RGB canvas
    rgb = cv2.cvtColor(canvas.astype('uint8'), cv2.COLOR_GRAY2BGR)

    # draw a line according to the angle
    xshift = int(((canvas.shape[1] - ls * resized.shape[1] + (padding + r) * f)  + (canvas.shape[1] - ls * resized.shape[1] + ((padding + r) + 1) * f)) / 2)
    yshift = int((padding + r) * f + f / 2)

    r_red = 1
    # drawing reference line and arc
    rgb = cv2.line(rgb,
        (xshift, yshift),
        (int(xshift + (r - r_red)  * f), yshift),
        (255, 200, 128), 3)

    rgb = cv2.ellipse(rgb, (xshift, yshift), ( int((r-r_red-1) * f), int((r-r_red-1) * f)),
           0, 0, a, (210, 50, 128), 3)

    # drawing arrow
    a_x = int(np.cos(np.deg2rad(a)) * (r - r_red) * f - overlay.shape[0] / 2 + overlay.shape[1])
    a_y = int(np.sin(np.deg2rad(a)) * (r - r_red) * f + overlay.shape[0] / 2)

    rgb = cv2.line(rgb,
        (xshift, yshift),
        (a_x, a_y),
        (128, 150, 0), 3)

    a_x_l = int(np.cos(np.deg2rad((a - 150) % 360)) * (1) * f + a_x)
    a_y_l = int(np.sin(np.deg2rad((a - 150) % 360)) * (1) * f + a_y)

    rgb = cv2.line(rgb,
        (a_x, a_y),
        (a_x_l, a_y_l),
        (128, 150, 0), 3)

    a_x_r = int(np.cos(np.deg2rad(a + 150)) * (1) * f + a_x)
    a_y_r = int(np.sin(np.deg2rad(a + 150)) * (1) * f + a_y)

    rgb = cv2.line(rgb,
        (a_x, a_y),
        (a_x_r, a_y_r),
        (128, 150, 0), 3)


    cpx += padding
    cpy += padding

    # add the angle
    rgb = cv2.putText(rgb,
        '{:.2f}'.format(a),
        (int(xshift + (r - r_red)  * f), int(yshift + f / 2)),
        cv2.FONT_HERSHEY_COMPLEX,
        text_scale * 2,
        (255, 200, 128),
        2
    )

    # draw centroid
    xshift = int(((canvas.shape[1] - ls * resized.shape[1] + cpx * f)  + (canvas.shape[1] - ls * resized.shape[1] + (cpx + 1) * f)) / 2)
    yshift = int(cpy * f + f / 2)

    rgb = cv2.circle(rgb,
         (
             xshift,
             yshift
         ), 3,(50, 80, 255), 3)

    rgb = cv2.putText(rgb,
        "C",
        (int(xshift + f), int(yshift + f)),
        cv2.FONT_HERSHEY_COMPLEX,
        text_scale * 2,
        (50, 80, 255),
        2
    )


    # draw keypoints
    vKeysCellShifted = []
    for vit in vKeysCell:
        xshift = int(((canvas.shape[1] - ls * resized.shape[1] + vit.pt[0] * f)  + (canvas.shape[1] - ls * resized.shape[1] + (vit.pt[0] + 1) * f)) / 2)
        yshift = int(vit.pt[1] * f + f / 2)
        vit.pt = (xshift, yshift)
        vKeysCellShifted.append(vit)

    rgb = cv2.drawKeypoints(rgb, vKeysCellShifted, rgb, (255, 0, 0))

    # add some information beneath the image
    rgb_info = np.zeros((rgb.shape[0] + 50, rgb.shape[1], rgb.shape[2]), dtype=np.uint8)
    rgb_info[:rgb.shape[0], :, :] = rgb
    rgb_info = cv2.putText(
        rgb_info,
        'PATCH: {} | KP: {} || w, a, s, d to position center || +/- to in/decrease patch size || q to quit'.format(PATCH_SIZE, len(vKeysCellShifted)),
        (20, rgb.shape[0] + 35),
        cv2.FONT_HERSHEY_COMPLEX,
        1,
        (255, 255, 255)
    )

    # rescale the window
    fscale = window_width / rgb_info.shape[1]
    fscale = 1.0 if fscale > 1 else fscale

    rgb_info = cv2.resize(rgb_info, (0, 0), fx=fscale, fy=fscale, interpolation=cv2.INTER_LINEAR)

    cv2.imshow('rgb', rgb_info)

    pressedKey = cv2.waitKey(1) & 0xFF
    if pressedKey == ord('q'):
        # quit
        cv2.destroyAllWindows()
        break
    if pressedKey == ord('w'):
        # set center pixel one higher
        center = (center[0] - 1, center[1])
        continue
    if pressedKey == ord('s'):
        # set center pixel one lower
        center = (center[0] + 1, center[1])
        continue
    if pressedKey == ord('a'):
        # set center pixel one left
        center = (center[0], center[1] - 1)
        continue
    if pressedKey == ord('d'):
        # set center pixel one right
        center = (center[0], center[1] + 1)
        continue

    if pressedKey == ord('m'):
        # toggle through modes
        mode = modes[modes.index(mode) + 1] if modes.index(mode) + 1 < len(modes) else modes[0]

    if pressedKey == ord('p'):
        # save a screenshot of the image
        cv2.imwrite('live.png', rgb)

    if pressedKey == ord('+'):
        # increase the patch area
        HALF_PATCH_SIZE += 1
        PATCH_SIZE += 2
        HALF_PATCH_SIZE = limit(HALF_PATCH_SIZE, 2, 20)
        PATCH_SIZE = limit(PATCH_SIZE, 5, 41)
        umax = calc_umax()
        continue

    if pressedKey == ord('-'):
        # decrease the patch area
        HALF_PATCH_SIZE -= 1
        PATCH_SIZE -= 2
        HALF_PATCH_SIZE = limit(HALF_PATCH_SIZE, 2, 20)
        PATCH_SIZE = limit(PATCH_SIZE, 5, 41)
        umax = calc_umax()
        continue
    #sleep(0.001)
