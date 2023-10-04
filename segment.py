import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def find_chars(contour_list):
    num_contours = len(contour_list)
    matched_result_idx = []  # Lista para armazenar índices de contornos correspondentes
    
    # Loop sobre todos os contornos
    for i in range(num_contours):
        matched_contours_idx = list(range(i, num_contours))  # Inicie com todos os contornos subsequentes
        matched_result_idx.append(matched_contours_idx)  # Adicione a lista de índices de contornos correspondentes
    
    return matched_result_idx  # Retorna a lista de índices de contornos correspondentes


def rotate_image(frame):
    cv2.imshow('original', frame)
    cv2.waitKey(0)
    img_ori = frame.copy()
    height, width, _ = img_ori.shape

    gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)
    cv2.imshow('cinza', gray)
    cv2.waitKey(0)

    # Binarização adaptativa
    img_thresh = cv2.adaptiveThreshold(
        gray,
        maxValue=255.0,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=19,
        C=9
    )
    cv2.imshow('binarizacao_adaptativa', img_thresh)
    cv2.waitKey(0)

    # Encontre contornos na imagem binarizada
    contours, _ = cv2.findContours(
        img_thresh,
        mode=cv2.RETR_LIST,
        method=cv2.CHAIN_APPROX_SIMPLE
    )

    possible_contours = []

    # Filtre os contornos com base em critérios de área e proporção
    img_thresh_contorno = img_ori.copy()
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        area = w * h
        ratio = w / h

        if area > 75 and w > 2 and h > 5 and 0.0 < ratio < 0.9:
            possible_contours.append({
                'contour': contour,
                'x': x,
                'y': y,
                'w': w,
                'h': h,
                'cx': x + (w / 2),
                'cy': y + (h / 2)
            })

            cv2.rectangle(img_thresh_contorno,(x,y),(x+w,y+h),(0,255,0),1)

            
    cv2.imshow('contorno', img_thresh_contorno)
    cv2.waitKey(0)
    
    result_idx = find_chars(possible_contours)
    if len(result_idx) == 0:
        return None  # Não foi possível encontrar uma placa

    # Classifique os caracteres correspondentes pela posição x
    sorted_chars = sorted((possible_contours[i] for i in result_idx[0]), key=lambda x: x['cx'])

    # Calcule as coordenadas do centro da placa para a rotação
    plate_cx = (sorted_chars[0]['cx'] + sorted_chars[-1]['cx']) / 2
    plate_cy = (sorted_chars[0]['cy'] + sorted_chars[-1]['cy']) / 2

    # Calcule a largura da placa
    plate_width = (sorted_chars[-1]['x'] + sorted_chars[-1]['w'] - sorted_chars[0]['x']) * 1.05

    # Calcule a altura da placa
    sum_height = sum(d['h'] for d in sorted_chars)
    plate_height = int(sum_height / len(sorted_chars) * 1.3)

    # Calcule o ângulo de rotação da placa
    triangle_height = sorted_chars[-1]['cy'] - sorted_chars[0]['cy']
    triangle_hypotenuse = np.linalg.norm(
        np.array([sorted_chars[0]['cx'], sorted_chars[0]['cy']]) -
        np.array([sorted_chars[-1]['cx'], sorted_chars[-1]['cy']])
    )
    angle = np.degrees(np.arcsin(triangle_height / triangle_hypotenuse))

    # Crie uma matriz de rotação
    rotation_matrix = cv2.getRotationMatrix2D(center=(plate_cx, plate_cy), angle=angle, scale=1.0)

    # Rotação da imagem original
    img_rotated = cv2.warpAffine(gray, M=rotation_matrix, dsize=(width, height))
    cv2.imshow('rotacionada', img_rotated)
    cv2.waitKey(0)

    # Recorte a região da placa da imagem rotacionada
    img_cropped = cv2.getRectSubPix(
        img_rotated,
        patchSize=(int(plate_width), int(plate_height)),
        center=(int(plate_cx), int(plate_cy))
    )
    cv2.imshow('cortada', img_cropped)
    cv2.waitKey(0)

    # Redimensione a imagem recortada
    img_resized = cv2.resize(img_cropped, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

    cv2.imshow('resized', img_resized)
    cv2.waitKey(0)
    return img_resized

    img_rotated = cv2.resize(img_cropped, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    ret, img = cv2.threshold(img_rotated, 70, 255, cv2.THRESH_BINARY)
    img = cv2.GaussianBlur(img, (3,3),0)
    cv2.imshow('pos_r', img)
    cv2.waitKey(0)    
    return img
