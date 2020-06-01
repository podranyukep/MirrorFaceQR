#!/usr/bin/env python
# coding: utf-8

u"""
Главный файл программного продукта
Автор: Екатерина Подранюк
"""

import cv2
import qrcode
import numpy as np
import dlib
import sys
from matplotlib import pyplot as plt
from tabulate import tabulate


def show_images(images, title):
    u"""
    Отображает изображения из списка, добавляя разделяющую черту
    :param images: список изображений
    :param title: заголовок окна
    :return:
    """
    # Создаем объект окна dlib
    win = dlib.image_window()
    win.set_title(title)
    win.clear_overlay()
    for_view = None
    black_bar = None
    for image in images:
        if for_view is None:
            for_view = image
        else:
            if black_bar is None:
                H = for_view.shape[0]
                if len(for_view.shape) > 2:
                    shape = (H, 3, 3)
                else:
                    shape = (H, 3)
                # Создаем разделитель
                black_bar = np.zeros(shape, dtype=np.uint8)
            for_view = np.concatenate((for_view, black_bar), axis=1)
            for_view = np.concatenate((for_view, image), axis=1)
    win.set_image(for_view)
    win.wait_until_closed()


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
    if size:
        dsize = (size, size)
    else:
        dsize = (new_h, new_w)
    img_filled = cv2.resize(img_filled, dsize)
    if not silent:
        show_images([img_filled], "Filled image")
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
    u"""
    Генерирует цветной BIO QR-код типа PIP
    :param photo1: компонента Photo для слоя R
    :param photo2: компонента Photo для слоя B
    :param silent: флаг отображения промежуточных результатов
    :return: сгенерированный BIO QR-код
    """
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
        show_images([photo1, qr_info, photo2], "Forming BIO QR-code")

    return result


def form_bio_qr(path_to_image):
    u"""
    Сформировать цветной BIO QR-код и подготовить для него входные данные
    :return: нормализованное ИЛ, сгенерированный код
    """
    # 1. Получить изображение лица (ИЛ)
    image = load_image(path_to_image)
    # 2. Нормализовать размер ИЛ
    image_square = make_square_img(image)
    # 3. Определить координаты половин изображения лица
    # 4. Создать слой-комбинацию из левой половины слоя R и правой слоя G
    rg = generate_layers_combination(image_square)
    show_images([rg], "RG component")
    # 5. Сгенерировать QR-код со слоями R=комбинация из п. 4, G=QR-код, B=слой B ИЛ
    qr_result = generate_qr_pip(rg, image_square[:, :, 2], silent=False)
    return image_square, qr_result


# процесс декодирования Bio QR-кода
def recreate_layers_RG(rg_layer):
    u"""
    Воссоздает слои R и G из слоя RG
    :param rg_layer: слой RG (левая часть со значениями R, правая - G)
    :return: восстановленные слои R и G
    """
    # 1. Сделаем отражение слоя по вертикали
    # При вертикальном отражении по оси Y в левой части слоя flipped
    # станет G слой, а в правой - R слой
    flipped = cv2.flip(rg_layer, 1)
    # 2. Получим данные об индексе середины изображения и его ширине
    middle = get_middle_coords(rg_layer)
    width = rg_layer.shape[1]
    # 3. Выделим половины изображений из слоев
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
        show_images([RG, restoredR, restoredG], "Recreated image")
        show_images([result], "Recreated image")
    return result


def make_hist(orig, result):
    u"""
    Сформировать гистограмму для двух изображений
    :param orig: оригинальное изображение
    :param result: изображение-результат вычислений
    :return:
    """
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
    u"""
    Рассчитать абсолютную разницу между двумя изображениями
    (code by Jeru Luke)
    :param orig: оригинальное изображение
    :param result: изображение-результат вычислений
    :return: процентная разница
    """
    # --- take the absolute difference of the images ---
    res = cv2.absdiff(orig, result)

    # --- convert the result to integer type ---
    res = res.astype(np.uint8)

    # --- find percentage difference based on number of pixels that are not zero ---
    percentage = (np.count_nonzero(res) * 100) / res.size
    return percentage


def find_face_landmarks(face_image, silent=True):
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
    if not silent and landmark_detection:
        # Создаем объект окна dlib
        win = dlib.image_window()
        win.set_title("Finding face landmarks")
        win.clear_overlay()
        win.set_image(face_image)
        win.add_overlay(landmark_detection)
        win.wait_until_closed()
    return points, landmark_detection


def calculate_distance_between_landmarks(first, second):
    u"""
    Вычисляет евклидово расстояние между соответствующими
    антропометрическими точками (АПТ)
    :param first: первый набор АПТ
    :param second: второй набор АПТ
    :return: сумма евклидовых расстояний
    """
    length = len(first)
    dist = 0
    for i in range(0, length):
        p1 = first[i]
        p2 = second[i]
        dist += pow(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2), 0.5)
    return dist


def calculate_antro(landmarks):
    u"""
    Расчитывает основные антропометрические характеристики лица
    :param landmarks: набор АПТ
    :return: значение антропометрических характеристик
    """
    lm = {
        'x': [point.x for point in landmarks],
        'y': [point.y for point in landmarks]
    }
    top_lip = lm['y'][50] if lm['y'][50] >= lm['y'][52] else lm['y'][52]
    return {
        'height': max(lm['y']) - min(lm['y']),
        'width': max(lm['x']) - min(lm['x']),
        'eye_width': abs(lm['x'][36] - lm['x'][39]),
        'nose_width': abs(lm['x'][31] - lm['x'][35]),
        'lips_width': abs(lm['x'][48] - lm['x'][54]),
        'nose_height': abs(lm['y'][33] - lm['y'][27]),
        'lips_height': abs(top_lip - lm['y'][57])
    }


def detect_antro_diff(landmarks1, landmarks2):
    u"""
    Вычислить разницу между антропометрическими характеристиками
    двух различных наборов АПТ
    :param landmarks1: первый набор АПТ
    :param landmarks2: второй набор АПТ
    :return:
    """
    face1 = calculate_antro(landmarks1)
    face2 = calculate_antro(landmarks2)
    diff_dict = {}
    for k in face1.keys():
        delta = abs(face1[k] - face2[k])
        diff_dict[k] = [face1[k], face2[k], delta,
                        round(delta/min(face1[k], face2[k]), 3)*100]
    return diff_dict


# Получаем путь к исходному ИЛ
if len(sys.argv) > 1:
    img_path = sys.argv[1]
else:
    img_path = "test_images/test" + str(12) + ".jpg"

# Генерируем цветной BIO QR-код
normal_image, qr_image = form_bio_qr(img_path)

# Реконструируем ИЛ на основе сформированного кода
restored = decode_qr_pip(qr_image, silent=False)
normal_qr = np.concatenate((normal_image, qr_image), axis=1)
summary = np.concatenate((normal_qr, restored), axis=1)

# Отображаем результат
show_images([normal_image, qr_image, restored], "Result of BIO QR-code decoding")

# Отображаем гистограмму для оригинального и восстановленного изображения
make_hist(normal_image, restored)

# Расчитываем абсолютную разницу между изображениями
diff = round(calculate_diff(normal_image, restored), 3)
print(u'Восстановленное изображение отличается от оригинального '
      u'на {p}%'.format(p=diff))

# Выполняем расчет и анализ антропометрических характеристик для
# исходного и реконструированного ИЛ
points, det = find_face_landmarks(normal_image)
points_rest, det_rest = find_face_landmarks(restored)
sum_distance = round(calculate_distance_between_landmarks(points, points_rest),
                     3)
print(u'Сумма Евклидовых расстояний между антропометрическими точками: '
      u'{s}'.format(s=sum_distance))
diff = detect_antro_diff(points, points_rest)
columns = [u'Высота лица', u'Ширина лица', u'Ширина глазной щели', u'Ширина носа',
           u'Ширина губ', u'Высота носа', u'Высота от нижней губы до верхней']
rows = [u'Оригинал', u'Восстановленное', u'Разница', u'Разница %']
print(tabulate(diff, headers=columns, showindex=rows))
# Отображаем полученную антропометрику на исходном ИЛ для сравнения
# Создаем объект окна dlib
win = dlib.image_window()
win.set_title("Face landmarks")
win.clear_overlay()
win.set_image(normal_image)
# Исходные АПТ - синим цветом
win.add_overlay(det)
# АПТ реконструированного лица - красным цветом
win.add_overlay(det_rest, color=dlib.rgb_pixel(255, 0, 0))
win.wait_until_closed()

