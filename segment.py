import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

#Rotacionar
PLATE_WIDTH_PADDING = 1.3 # 1.3
PLATE_HEIGHT_PADDING = 1.5 # 1.5
MIN_PLATE_RATIO = 1
MAX_PLATE_RATIO = 10

#Selecionando os boxes pelo arranjamento dos contornos
MAX_DIAG_MULTIPLYER = 5 # 5
MAX_ANGLE_DIFF = 12.0 # 12.0
MAX_AREA_DIFF = 0.5 # 0.5
MAX_WIDTH_DIFF = 0.8
MAX_HEIGHT_DIFF = 0.2
MIN_N_MATCHED = 2 # 3

def find_chars(contour_list, img):
    matched_result_idx = []  # Lista para armazenar índices de contornos correspondentes
    
    # Loop sobre todos os contornos
    for i, d1 in enumerate(contour_list):
        matched_contours_idx = [i]  # Inicie com o próprio índice d1
        
        # Loop para comparar d1 com todos os outros contornos
        for j, d2 in enumerate(contour_list[i+1:]):
            j = j + i + 1  # Ajuste o índice para corresponder ao contorno real em contour_list
            
            dx = abs(d1['cx'] - d2['cx'])
            dy = abs(d1['cy'] - d2['cy'])


            diagonal_length1 = np.sqrt(d1['w'] ** 2 + d1['h'] ** 2)

            distance = np.linalg.norm(np.array([d1['cx'], d1['cy']]) - np.array([d2['cx'], d2['cy']]))
            angle_diff = 90 if dx == 0 else np.degrees(np.arctan(dy / dx))
            area_diff = abs(d1['w'] * d1['h'] - d2['w'] * d2['h']) / (d1['w'] * d1['h'])
            width_diff = abs(d1['w'] - d2['w']) / d1['w']
            height_diff = abs(d1['h'] - d2['h']) / d1['h']

            matched_contours_idx.append(j)
            '''# Verifique se os contornos correspondem com base em várias medidas
            if distance < diagonal_length1 * MAX_DIAG_MULTIPLYER \
            and angle_diff < MAX_ANGLE_DIFF and area_diff < MAX_AREA_DIFF \
            and width_diff < MAX_WIDTH_DIFF and height_diff < MAX_HEIGHT_DIFF:
                matched_contours_idx.append(j)  # Adicione o índice do contorno correspondente
                cv2.rectangle(img,(d2['x'],d2['y']),(d2['x']+d2['w'],d2['y']+d2['h']),(255,0,0),3)'''

        if len(matched_contours_idx) >= MIN_N_MATCHED:
            matched_result_idx.append(matched_contours_idx)  # Adicione a lista de índices de contornos correspondentes
    
    return matched_result_idx  # Retorna a lista de índices de contornos correspondentes


def rotate_image(frame):
    img_ori = frame.copy()

    height, width, _ = img_ori.shape
    print(width)
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

    # Encontre contornos na imagem binarizada
    contours, _ = cv2.findContours(
        img_thresh,
        mode=cv2.RETR_LIST,
        method=cv2.CHAIN_APPROX_SIMPLE
    )

    possible_contours = []

    # Filtre os contornos com base em critérios de área e proporção
    img = img_ori.copy()
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        #cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),3)
        
        area = w * h
        ratio = w / h

        if area > 85 and w > 9 and h > 8 and 0.0 < ratio < 0.7:
            possible_contours.append({
                'contour': contour,
                'x': x,
                'y': y,
                'w': w,
                'h': h,
                'cx': x + (w / 2),
                'cy': y + (h / 2)
            })

    
    result_idx = find_chars(possible_contours, img)
    cv2.imshow('contorno', img)
    cv2.waitKey(0)
    if len(result_idx) == 0:
        return None  # Não foi possível encontrar uma placa

    matched_result = []

    for idx_list in result_idx:
        matched_result.append(np.take(possible_contours, idx_list))
    
    # Classifique os caracteres correspondentes pela posição x
    sorted_chars = sorted(matched_result[0], key=lambda x: x['cx'])

    # Calcule as coordenadas do centro da placa para a rotação
    plate_cx = (sorted_chars[0]['cx'] + sorted_chars[-1]['cx']) / 2
    plate_cy = (sorted_chars[0]['cy'] + sorted_chars[-1]['cy']) / 2
    # Calcule a largura da placa
    plate_width = ((sorted_chars[-1]['x'] + sorted_chars[-1]['w']) - sorted_chars[0]['x']) * 1.15
    print(plate_width)
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
    cv2.waitKey(0)   
    img_rotated = cv2.warpAffine(gray, M=rotation_matrix, dsize=(width, height))

    # Recorte a região da placa da imagem rotacionada
    img_cropped = cv2.getRectSubPix(
        img_rotated,
        patchSize=(int(plate_width), int(plate_height)),
        center=(int(plate_cx), int(plate_cy))
    )

    img_rotated = cv2.resize(img_cropped, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    ret, img = cv2.threshold(img_rotated, 70, 255, cv2.THRESH_BINARY)
    img = cv2.GaussianBlur(img, (5,5),0)
    cv2.imshow('pos_r', img)
    cv2.waitKey(0)    
    return img
