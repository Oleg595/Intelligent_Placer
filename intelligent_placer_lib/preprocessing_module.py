import cv2
import numpy as np

from intelligent_placer_lib.recognition import list_paper, delete_fone


class Shape:
    def __init__(self, img, start_img, name):
        x, y = np.where(img[:, :, 0] != 0)
        self.img = cv2.resize(img, (1000, 1000))
        self.start_img = start_img
        self.name = name
        self.data = []

    def get_image(self):
        return self.img

    def recognise(self, src):
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(src, None)
        kp2, des2 = orb.detectAndCompute(self.start_img, None)

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        correct_matches = []
        for m, n in matches:
            if m.distance < 0.8 * n.distance:
                correct_matches.append([m])

        return len(correct_matches)

    def get_size(self):
        return min(self.img[:, 0, 0].size, self.img[0, :, 0].size), max(self.img[:, 0, 0].size, self.img[0, :, 0].size)

    def get_map(self, eps: int):
        height = 1000 // eps
        width = 1000 // eps

        map = np.zeros((height, width))

        for i in range(0, height):
            for j in range(0, width):
                if len(np.where(self.img[eps * i: eps * (i + 1), eps * j: eps * (j + 1)] != 0)[0]) != 0:
                    map[i, j] = 255
                else:
                    map[i, j] = 0

        return map

def cv_show(name, img):
    imS = cv2.resize(img, (600, 600))
    cv2.imshow(name, imS)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_mask(image):
    src = image
    gr = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    bl = cv2.medianBlur(gr, 19)
    canny = cv2.Canny(bl, 0, 5)

    contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    pap_list = \
    sorted(contours, key=lambda tup: max(tup[:, 0, 0]) + max(tup[:, 0, 1]) - min(tup[:, 0, 0]) - min(tup[:, 0, 1]),
           reverse=True)[0]

    max(pap_list[:, 0, 0])
    mask = np.zeros(src.shape, dtype=np.uint8)
    cv2.drawContours(image=mask, contours=[pap_list], color=(255, 255, 255), thickness=cv2.FILLED, contourIdx=0)

    return mask

def working():
    images = ['book', 'check_book', 'clock', 'cup', 'key', 'lighter', 'mouse', 'pen', 'pin', 'scissors']

    elements = {}

    for path in images:
        start_img = cv2.imread('./images/items/' + path + '.jpg')
        start_img = list_paper(start_img)
        img = delete_fone(start_img)
        elements[path] = Shape(img=img, start_img=start_img, name=path)

    return elements

