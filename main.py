#!/usr/bin/env python
# coding: utf-8

u"""
Главный файл программного продукта
"""

import cv2
import qrcode
import numpy as np
import dlib


# процесс формирования Bio QR-кода
# 1. Получить изображение лица (ИЛ)
def load_face_image(path_to_img):
    u"""
    Загружает изображение по указанному пути
    :param path_to_img: Путь к изображению
    :return: Изображение img
    """
    img = cv2.imread(path_to_img)
    return img


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
    # Если изображение уже имеет пропорции 1:1, то возвращаем его же
    if h == w and not size:
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
        cv2.imshow("Filled image", img_filled)
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
    R = image[:, :, 2]
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
    cv2_qr = cv2.cvtColor(qr_numpy, cv2.COLOR_RGB2GRAY)
    qr_info = cv2.resize(cv2_qr, (MQR, MQR), interpolation=cv2.INTER_LINEAR)
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

# процесс декодирования Bio QR-кода
# 1. Получить изображение Bio QR-кода
# 2. Разбить изображение на слои
# 3. Выделить половины из слоя R
# 4. Провести зеркальное отражение каждой из половин и сформировать
# соответствующие цветные слои ИЛ
# 5. Разместить восстановленные слои R и G вместе со слоем B из Bio QR-кода
# 6. Выполнить сведение полученных слоев и сформировать результирующее изображение
