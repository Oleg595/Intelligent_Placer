import numpy as np
from intelligent_placer_lib.intelligent_placer_module import intelligent_placer

'''Координаты точек принимают значения от 0 до 10 по каждой из осей. 
Точка (0, 0) соответствует верхнему левому краю листа, где оси x 
соответствует самая длинная сторона листа. Точка (10, 10) - 
нижнему правому. Количество углов в многоугольнике ограничено 10 - ю. 
Возможные изображения для обработки находятся в папке examples.'''
print(intelligent_placer('./images/examples/image10.jpg', np.array([[0, 10], [10, 10], [10, 0]])))
print(intelligent_placer('./images/examples/image10.jpg', np.array([[0, 0], [0, 10], [10, 10], [10, 0]])))
print(intelligent_placer('./images/examples/image10.jpg', np.array([[3, 10], [0, 4], [5, 0], [10, 4], [8, 10]])))
print(intelligent_placer('./images/examples/image2.jpg', np.array([[3, 10], [0, 4], [5, 0], [10, 4], [8, 10]])))
