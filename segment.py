import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def order_chars(contour_list):
    num_contours = len(contour_list)
    matched_result_idx = []  # Lista para armazenar índices de contornos correspondentes
    
    # Loop sobre todos os contornos
    for i in range(num_contours):
        matched_contours_idx = list(range(i, num_contours))  # Inicie com todos os contornos
        matched_result_idx.append(matched_contours_idx)  # Adicione a lista de índices de contornos correspondentes

    if len(matched_result_idx) == 0:
        return None  # Não foi possível encontrar duas seleções extremas de caracteres da placa
    else:
        matched_result_idx = sorted((contour_list[i] for i in matched_result_idx[0]), key=lambda x: x['cx'])
        return matched_result_idx


def rotate_image(frame):
    
    img_ori = frame.copy()
    height, width, _ = img_ori.shape

    gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)

    # Binarização adaptativa
    img_thresh = cv2.adaptiveThreshold(
        gray,
        maxValue=255.0,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=19,
        C=9
    )

    # Busca de contornos na imagem binarizada
    contours, _ = cv2.findContours(
        img_thresh,
        mode=cv2.RETR_LIST,
        method=cv2.CHAIN_APPROX_SIMPLE
    )

    possible_contours = []
    #image = gray.copy()
    # Filtro entre os contornos com base em critérios de área e proporção
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        area = w * h
        ratio = w / h
        
        if area > 50 and w > 2 and h > 8 and 0.0 < ratio < 0.9:
            print(area, w, h, ratio)
            possible_contours.append({
                'contour': contour,
                'x': x,
                'y': y,
                'w': w,
                'h': h,
                'cx': x + (w / 2),
                'cy': y + (h / 2)
            })
            #cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)

            
    #cv2.imshow("aklsdj", image)                
    #cv2.waitKey(0)
    sorted_chars = order_chars(possible_contours)
    # Classificação dos caracteres correspondentes pela posição central x
    if sorted_chars is not None:
        # Calculo das coordenadas do centro da placa para a rotação
        plate_cx = (sorted_chars[0]['cx'] + sorted_chars[-1]['cx']) / 2
        plate_cy = (sorted_chars[0]['cy'] + sorted_chars[-1]['cy']) / 2

        # Cálculo do angulo de rotação da placa
        triangle_height = sorted_chars[-1]['cy'] - sorted_chars[0]['cy']
        triangle_hypotenuse = np.linalg.norm(
            np.array([sorted_chars[0]['cx'], sorted_chars[0]['cy']]) -
            np.array([sorted_chars[-1]['cx'], sorted_chars[-1]['cy']])
        )
        #Conversão de radianos para graus.
        angle = np.degrees(np.arcsin(triangle_height / triangle_hypotenuse))

        # Matriz de rotação
        rotation_matrix = cv2.getRotationMatrix2D(center=(plate_cx, plate_cy), angle=angle, scale=1.0)

        # Rotação da imagem original
        img_rotated = cv2.warpAffine(gray, M=rotation_matrix, dsize=(width, height))

        # Calculo de largura da placa
        plate_width = (sorted_chars[-1]['x'] + sorted_chars[-1]['w'] - sorted_chars[0]['x']) * 1.15

        # Calculo da altura da placa
        sum_height = sum(d['h'] for d in sorted_chars)
        plate_height = int(sum_height / len(sorted_chars) * 1.3)

        # Recorte a região da placa da imagem rotacionada
        img_cropped = cv2.getRectSubPix(
            img_rotated,
            patchSize=(int(plate_width), int(plate_height)),
            center=(int(plate_cx), int(plate_cy))
        )
        # Redimensione a imagem recortada
        img_resized = cv2.resize(img_cropped, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

        cv2.imshow('resized_rotacionada', img_resized)
        cv2.waitKey(0)
        return img_resized

    return None

    img_rotated = cv2.resize(img_cropped, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    ret, img = cv2.threshold(img_rotated, 70, 255, cv2.THRESH_BINARY)
    img = cv2.GaussianBlur(img, (3,3),0)
    cv2.imshow('pos_r', img)
    cv2.waitKey(0)    
    return img
