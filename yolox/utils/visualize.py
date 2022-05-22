#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import cv2
import numpy as np

__all__ = ["vis"]


def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None):

    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return img


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color

clients = {}
clients_ids = []

def plot_tracking(image, tlwhs, obj_ids, scores=None, frame_id=0, fps=0., ids2=None):
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255

    #text_scale = max(1, image.shape[1] / 1600.)
    #text_thickness = 2
    #line_thickness = max(1, int(image.shape[1] / 500.))
    text_scale = 2
    text_thickness = 3
    line_thickness = 3

    workers = 0
    queue = 0
        
    # visual parametres 
    visualization = True
    cashbox_border = 0.62 # коэффициент расположения линии, разделяющей прилавок и торговый зал
    cashbox = tuple(map(int, (im_w*0.39, im_h*cashbox_border, im_w*0.62, im_h*0.85)))
    queuebox = tuple(map(int, (cashbox[0], cashbox[1], im_w*0.95, cashbox[3])))
    

    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        dot = tuple(map(int, (x1+w/2, y1+h/2)))
        obj_id = int(obj_ids[i])
        id_text = '{}'.format(int(obj_id))
        if ids2 is not None:
            id_text = id_text + ', {}'.format(int(ids2[i]))
        color = get_color(abs(obj_id))
        cv2.circle(im, dot, 5, color=color, thickness=-1)
        if visualization == True:
            cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
            cv2.putText(im, id_text, (intbox[0], intbox[1]), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 0),
                        thickness=text_thickness)
        if dot[1]<im_h*cashbox_border:
            workers += 1
        if dot[0]>cashbox[0] and dot[0]<cashbox[2] and dot[1]>im_h*cashbox_border and dot[1]<cashbox[3]:
            if obj_id not in list(clients.keys()):
                clients.update([[obj_id, 0]])
            if obj_id in list(clients.keys()):
                clients[obj_id] += 1
                if clients[obj_id] > 250 and obj_id not in clients_ids:
                    clients_ids.append(obj_id)
        if dot[0]>queuebox[0] and dot[0]<queuebox[2] and dot[1]>im_h*cashbox_border and dot[1]<queuebox[3]:
            queue += 1
    

    if visualization == True:
        cv2.line(im, (0, round(im_h*cashbox_border)), (im_w, round(im_h*cashbox_border)), (0, 0, 255), thickness=line_thickness)
        cv2.rectangle(im, queuebox[0:2], queuebox[2:4], (255, 153, 51), thickness=line_thickness) # queue box
        cv2.rectangle(im, cashbox[0:2], cashbox[2:4], (0, 153, 0), thickness=line_thickness) # cashbox
    
    cv2.rectangle(im, (0, 0), (round(im_w*0.6), round(im_h*0.15)), color=(255, 255, 255), thickness=-1)
    cv2.putText(im, f'Employees: {workers} Total Clients: {len(clients_ids)}', # Potential: {len(tlwhs)-workers-client}
                (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 0), thickness=text_thickness)
    cv2.putText(im, f'Queue: {queue}',
                (0, int(30 * text_scale)), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 0), thickness=text_thickness)

    return im


_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)
