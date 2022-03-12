from copy import copy

import cv2
import numpy as np

from intelligent_placer_lib.preprocessing_module import working
from intelligent_placer_lib.recognition import recognise_objects


def try_insert(x_start: int, y_start: int, list: np.array, img: np.array):
    height = img[0, :].size
    width = img[:, 0].size

    list_copy = copy(list)

    for y in range(y_start, y_start + height):
        for x in range(x_start, x_start + width):
            if list_copy[x, y] == 0 and img[x - x_start, y - y_start] != 0:
                return [], False
            if list_copy[x, y] != 0 and img[x - x_start, y - y_start] != 0:
                list_copy[x, y] = 0

    return list_copy, True

def find(elements: np.array, list: np.array, x_start: int, y_start: int):
    if len(elements) == 0:
        return True

    height = list[0, :].size
    width = list[:, 0].size

    img_height = elements[0][0, :].size
    img_width = elements[0][:, 0].size

    while height - y_start > img_height or width - x_start > img_width:
        new_list, result = try_insert(x_start, y_start, list, elements[0])
        if not result:
            if width - x_start > img_width:
                x_start += 1
            else:
                x_start = 0
                y_start += 1
        else:
            result = find(elements[1:], new_list, 0, 0)
            if result:
                return True
            else:
                if width - x_start > img_width:
                    x_start += 1
                else:
                    y_start += 1

    return False

def algorithm(elements: np.array, pts: np.array, size):
    list = np.zeros((size[0], size[1]))
    cv2.drawContours(list, [pts], -1, 255, -1)

    x, y = np.where(list != 0)
    list = list[min(x): max(x), min(y): max(y)]

    return find(elements, list, 0, 0)

def intelligent_placer(path: str, pts: np.array):
    objects = working()
    src = cv2.imread(path)
    recs = recognise_objects(src)

    matches = {}
    for object in objects:
        match = []
        for rec in recs:
            if rec.size != 0 and rec[:, 0, 0].size * rec[0, :, 0].size >= 8000:
                match.append(objects[object].recognise(rec))
        matches[object] = match

    detects = []

    for i in range(len(matches[next(iter(matches))])):
        choose = next(iter(matches))
        for match in matches:
            if matches[match][i] > matches[choose][i]:
                choose = match
        if matches[choose][i] != 0:
            matches.pop(choose)
            detects.append(choose)

    elements = []

    size = np.zeros(2, dtype=int)

    for detect in detects:
        map = objects[detect].get_map(10)
        size[0], size[1] = map[:, 0].size, map[0, :].size
        x, y = np.where(map != 0)
        elements.append(map[min(x): max(x), min(y): max(y)])

    for i in range(len(pts)):
        pts[i, 0] = int(pts[i, 0] * (size[0] / 10))
        pts[i, 1] = int(pts[i, 1] * (size[1] / 10))

    return algorithm(elements, pts, size)
