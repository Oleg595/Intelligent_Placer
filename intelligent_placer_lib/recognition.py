import cv2
import numpy as np

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

def list_paper(img):
    mask = get_mask(img)
    mask = mask[:, :, 0]

    x, y = np.where(mask != 0)

    up = min(x)
    bottom = max(x)
    left = min(y)
    right = max(y)

    #img = cv2.bitwise_and(img, img, mask=mask)
    img = img[up:bottom, left:right]

    return img

def cv_show_rec(name, img):
    imS = cv2.resize(img, (600, 600))
    cv2.imshow(name, imS)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def only_mask(src):
    height = src[:, 0, 0].size
    width = src[0, :, 0].size

    return src[30: height - 30, 30: width - 30, :]

def delete_fone(src):
    mask = only_mask(src)
    detected_edges = cv2.GaussianBlur(src=mask, ksize=(9, 9), sigmaX=10, dst=50)
    canny = cv2.Canny(detected_edges, 10, 100, apertureSize=3)
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)

    mask = np.zeros(src.shape, dtype=np.uint8)
    cv2.drawContours(image=mask, contours=contours, color=(255, 255, 255), thickness=cv2.FILLED, contourIdx=-1)

    return mask

def recognise_objects(img):
    paper = list_paper(img)
    paper = only_mask(paper)
    canny = cv2.Canny(paper, 150, 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 35))
    closed = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel)

    contours = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    cv2.drawContours(closed, contours, -1, (255, 255, 255), -1)

    result = []
    for contour in contours:
        image = np.zeros(closed.shape, dtype=np.uint8)
        cv2.drawContours(image, [contour], -1, (255, 255, 255), -1)
        x, y = np.where(image != 0)
        image = paper[min(x): max(x), min(y): max(y)]
        result.append(image)

    return result
