#!/usr/bin/env python
# coding: utf-8

u"""
Главный файл программного продукта
"""

import cv2
import qrcode
import numpy as np
import dlib
from matplotlib import pyplot as plt


# процесс формирования Bio QR-кода
# 1. Получить изображение лица (ИЛ)
def load_image(path_to_img):
    u"""
    Загружает изображение по указанному пути
    :param path_to_img: Путь к изображению
    :return: Изображение img
    """
    img = cv2.imread(path_to_img)
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return image


# 2. Нормализовать размер ИЛ
def make_square_img(image, size=None, silent=True):
    u"""
    Приводит изображение к пропорциям 1:1 по большей стороне
    :param image: изображение
    :return: new_image: исходное изображение, дополненное белым цветом
    до соотношения сторон 1:1
    """

    # Получаем высоту и ширину изображения
    shape = image.shape
    h = shape[0]
    w = shape[1]
    # Если изображение уже имеет пропорции 1:1..
    if h == w and not size:
        # Если размер нечетный, то уменьшаем изображение на 1 пиксель
        if h % 2:
            return cv2.resize(image, (h-1, h-1))
        # Иначе возвращаем исходное изображение
        return image

    # Определяем новые размеры по большей стороне
    # Флаги добавления границы заполнением (верх/низ и лево/право)
    tb = lr = False
    # Размер границы
    border_size_y = 0
    border_size_x = 0
    if size and size >= max(h, w):
        if size % 2:
            size -= 1
        new_h = new_w = size
        border_size_x = int((new_w - w) / 2)
        border_size_y = int((new_h - h) / 2)
        lr = True
        tb = True
    elif h > w:
        if h % 2:
            h -= 1
        new_h = new_w = h
        border_size_x = int((new_w - w) / 2)
        lr = True
    else:
        if w % 2:
            w -= 1
        new_w = new_h = w
        border_size_y = int((new_h - h) / 2)
        tb = True
    # Делаем заполнение границами
    img_filled = cv2.copyMakeBorder(
        image,
        top=border_size_y if tb else 0,
        bottom=border_size_y if tb else 0,
        left=border_size_x if lr else 0,
        right=border_size_x if lr else 0,
        borderType=cv2.BORDER_CONSTANT,
        value=[255, 255, 255]  # Заполняем белым цветом
    )
    if not silent:
        to_show = cv2.cvtColor(img_filled, cv2.COLOR_RGB2BGR)
        cv2.imshow("Filled image", to_show)
        cv2.waitKey(0)
    return img_filled


# 3. Определить координаты половин изображения лица
def get_middle_coords(image):
    u"""
    Определяет индекс середины изображения
    :param image: Исходное изображение
    :return: индекс середины
    """
    width = image.shape[1]
    if width % 2 > 0:
        return int((width - 1) / 2)
    return int(width / 2)


# 4. Создать слой-комбинацию из левой половины слоя R и правой слоя G
def generate_layers_combination(image):
    u"""
    Создает слой-комбинацию из половин слоев R и G
    :param image: Исходное изображение
    :return: слой-комбинация с левой половиной R и правой половиной G
    """
    # 1. Получаем значения компонент r и g для этого изображения
    G = image[:, :, 1]
    R = image[:, :, 0]
    # 2. Получаем индекс середины изображения
    middle = get_middle_coords(image)
    width = image.shape[1]
    # 3. Копируем компоненту r в отдельный массив
    r_copy = R.copy()
    # 4. В правую половину компоненты r записываем значения компоненты g
    for i in range(middle, width):
        r_copy[:, i] = G[:, i]
    return r_copy


# 5. Сгенерировать QR-код со слоями R=комбинация из п. 4, G=QR-код, B=слой B ИЛ
def generate_qr_pip(photo1, photo2, silent=True):
    # Изображение к данному моменту уже должно быть квадратным
    M = N = photo1.shape[0]
    # Оставляем запас для границ QR-кода
    MQR = M - 30

    # 1. Генерируем полутоновый код QR info
    qr_img = qrcode.make("Sample data", version=8)
    qr_numpy = np.array(qr_img)
    if qr_numpy.dtype == bool:
        qr_numpy = qr_numpy.astype(np.uint8) * 255
    # cv2_qr = cv2.cvtColor(qr_numpy, cv2.COLOR_RGB2GRAY)
    qr_info = cv2.resize(qr_numpy, (MQR, MQR), interpolation=cv2.INTER_LINEAR)
    qr_info = make_square_img(qr_info, silent=silent, size=M)

    # 4. Выбираем компоненты (для данной реализации для кода PIP)
    # QRpip(:, :, 1) = R = photo1,
    # QRpip(:, :, 2) = QRinfo = qr_info,
    # QRpip(:, :, 3) = B = photo2

    # 5. Формируем цветной BIO QR по выбранным компонентам
    result = np.zeros((M, N, 3), np.uint8)
    result[:, :, 0] = photo1
    result[:, :, 1] = qr_info
    result[:, :, 2] = photo2

    if not silent:
        # Создаем объект окна dlib
        win = dlib.image_window()
        win.set_title("Forming BIO QR-code")
        win.clear_overlay()
        win.set_image(result)
        win.wait_until_closed()

    return result


def form_bio_qr():
    # 1. Получить изображение лица (ИЛ)
    img_path = "test_images/test" + str(10) + ".jpg"
    image = load_image(img_path)
    # 2. Нормализовать размер ИЛ
    image_square = make_square_img(image)
    # 3. Определить координаты половин изображения лица
    # 4. Создать слой-комбинацию из левой половины слоя R и правой слоя G
    rg = generate_layers_combination(image_square)
    # 5. Сгенерировать QR-код со слоями R=комбинация из п. 4, G=QR-код, B=слой B ИЛ
    qr_result = generate_qr_pip(rg, image_square[:, :, 2])
    return image_square, qr_result


# процесс декодирования Bio QR-кода
def recreate_layers_RG(rg_layer):
    u"""
    Воссоздает слои R и G из слоя RG
    :param rg_layer: слой RG (левая часть со значениями R, правая - G)
    :return: восстановленные слои R и G
    """
    # 2. Сделаем отражение слоя по вертикали
    # При вертикальном отражении по оси Y в левой части слоя flipped
    # станет G слой, а в правой - R слой
    flipped = cv2.flip(rg_layer, 1)
    # 3. Получим данные об индексе середины изображения и его ширине
    middle = get_middle_coords(rg_layer)
    width = rg_layer.shape[1]
    # 4. Выделим половины изображений из слоев
    copyR = rg_layer[:, 0:middle]
    copyG = rg_layer[:, middle:width]
    fullR = np.concatenate((copyR, flipped[:, middle:width]), axis=1)
    fullG = np.concatenate((flipped[:, 0:middle], copyG), axis=1)
    return fullR, fullG


# 1. Получить изображение Bio QR-кода (см. load_image)
def decode_qr_pip(qr_image, silent=True):
    u"""
    Проводит декодирование PIP QR-кода и восстанавливает исходное изображение лица
    :param qr_image: исходный BIO QR-код типа PIP
    :return:
    """
    height, width, _ = qr_image.shape
    # 2. Разбить изображение на слои
    RG = qr_image[:, :, 0]
    qr_info = qr_image[:, :, 1]
    B = qr_image[:, :, 2]
    # 3. Выделить половины из слоя R
    # 4. Провести зеркальное отражение каждой из половин и сформировать
    # соответствующие цветные слои ИЛ
    restoredR, restoredG = recreate_layers_RG(RG)
    # 5. Разместить восстановленные слои R и G вместе со слоем B из Bio QR-кода
    result = np.zeros((height, width, 3), np.uint8)
    result[:, :, 0] = restoredR
    result[:, :, 1] = restoredG
    result[:, :, 2] = B
    # 6. Выполнить сведение полученных слоев и сформировать результирующее изображение
    if not silent:
        win = dlib.image_window()
        win.set_title("Recreated image")
        win.clear_overlay()
        win.set_image(result)
        win.wait_until_closed()
    return result


def make_hist(orig, result):
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        histr = cv2.calcHist([orig], [i], None, [256], [0, 256])
        histr_my = cv2.calcHist([result], [i], None, [256], [0, 256])

        plt.plot(histr, color=col, label="{c} comp of Original".format(c=col))
        plt.plot(histr_my, 'g--', color=col, label="{c} comp of Decoded".format(c=col))
        plt.legend()
        plt.xlim([0, 256])
    plt.show()


def calculate_diff(orig, result):
    # --- take the absolute difference of the images ---
    res = cv2.absdiff(orig, result)

    # --- convert the result to integer type ---
    res = res.astype(np.uint8)

    # --- find percentage difference based on number of pixels that are not zero ---
    percentage = (np.count_nonzero(res) * 100) / res.size
    return percentage


def find_face_landmarks(face_image):
    u"""
    Поиск ключевых точек лица
    :param img_path: путь к изображению
    :return:
    """

    # Загружаем ИЛ и конвертируем его в grayscale (полутона)
    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)

    # p - файл с уже обученной моделью для поиска ключевых точек на ИЛ (от dlib)
    p = "shape_predictor_68_face_landmarks.dat"
    # Инициализируем детектор лиц dlib (HOG-based)
    # и создаем predictor для контуров лица на основе имеющейся модели
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(p)

    # Ищем лица в полутоновом изображении
    # The 1 in the second argument indicates that we should
    # upsample the image 1 time.
    # This will make everything bigger and allow us to detect more faces.
    rects = detector(gray, 1)
    print(u"---------- Метод find_face_landmarks ----------")
    # Выводим количество найденных лиц
    print(u"Количество обнаруженных лиц: {}".format(len(rects)))

    # Создаем объект окна dlib
    win = dlib.image_window()
    win.set_title("Finding face landmarks")
    win.clear_overlay()
    win.set_image(face_image)

    points = None
    landmark_detection = None
    # Для каждого найденного лица..
    for (i, rect) in enumerate(rects):
        # Находим ключевые точки для каждого найденного лица
        shape = predictor(gray, rect)

        # Преобразовываем полученные точки в numpy список координат (x, y)
        # shape = face_utils.shape_to_np(shape)
        # Для всех полученных координат..
        points = [shape.part(i) for i in range(0, 68)]
        landmark_detection = dlib.full_object_detection(
            rect, points
        )
        # Загружаем изображение в окно и добавляем сверху найденные
        # ключевые точки лица
        win.add_overlay(landmark_detection)
    win.wait_until_closed()
    return points, landmark_detection


normal_image, qr_image = form_bio_qr()
restored = decode_qr_pip(qr_image)
normal_qr = np.concatenate((normal_image, qr_image), axis=1)
summary = np.concatenate((normal_qr, restored), axis=1)
# Создаем объект окна dlib
win = dlib.image_window()
win.set_title("Result of BIO QR-code decoding")
win.clear_overlay()
win.set_image(summary)
win.wait_until_closed()
make_hist(normal_image, restored)
diff = round(calculate_diff(normal_image, restored), 3)
print(u'Восстановленное изображение отличается от оригинального на {p}%'.format(p=diff))
points, det = find_face_landmarks(normal_image)
points_rest, det_rest = find_face_landmarks(restored)
