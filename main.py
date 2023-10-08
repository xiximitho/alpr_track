import os.path
import re

import cv2
import numpy as np
import pytesseract
from pytesseract import Output
from sort.tracker import SortTracker

from segment import rotate_image

net_vehicle = cv2.dnn.readNet('./model/vehicle_novo.onnx')
net_plate = cv2.dnn.readNet('./model/placa.onnx')
WIDTH =  640
HEIGHT = 640

def get_detections(img,net):
    # 1.CONVERT IMAGE TO YOLO FORMAT
    detections = []
    image = img #.copy()
    row, col, d = image.shape

    max_rc = max(row,col)
    input_image = np.zeros((max_rc,max_rc,3),dtype=np.uint8)
    input_image[0:row,0:col] = image

    # 2. GET PREDICTION FROM YOLO MODEL
    img_width, img_height, _ = img.shape

    if img_width > 0 and img_height > 0:
        blob = cv2.dnn.blobFromImage(input_image,1/255,(WIDTH,HEIGHT),swapRB=True,crop=False)
        net.setInput(blob)
        preds = net.forward()
        detections = preds[0]

    return input_image, detections

def non_maximum_supression(input_image,detections):

    # 3. FILTER DETECTIONS BASED ON CONFIDENCE AND PROBABILIY SCORE

    # center x, center y, w , h, conf, proba
    boxes = []
    confidences = []

    image_w, image_h = input_image.shape[:2]
    x_factor = image_w/WIDTH
    y_factor = image_h/HEIGHT

    for i in range(len(detections)):
        row = detections[i]
        confidence = row[4] # confidence of detecting license plate
        
        if confidence > 0.7:
            class_score = row[5] # probability score of license plate
            
            if class_score > 0.4:
                cx, cy , w, h = row[0:4]

                left = int((cx - 0.5*w)*x_factor)
                top = int((cy-0.5*h)*y_factor)
                width = int(w*x_factor)
                height = int(h*y_factor)
                box = np.array([left,top,width,height])

                confidences.append(confidence)
                boxes.append(box)

    # 4.1 CLEAN
    boxes_np = np.array(boxes).tolist()
    confidences_np = np.array(confidences).tolist()

    # 4.2 NMS
    index = cv2.dnn.NMSBoxes(boxes_np,confidences_np,0.25,0.7)

    return boxes_np, confidences_np, index

def yolo_predictions(img, net, track_id_counter):
    # 1: Detecção
    input_image, detections = get_detections(img, net)
    # 2: Supressão Não Máxima
    boxes_np, confidences_np, index = non_maximum_supression(input_image, detections)

    # Create an array for SORT input
    detections_for_sort = []

    for ind in index:
        x_min, y_min, width, height = boxes_np[ind]  # Extract coordinates and dimensions
        x_max = x_min + width
        y_max = y_min + height

        detection = [x_min, y_min, x_max, y_max, confidences_np[ind], track_id_counter]  # Append track ID
        detections_for_sort.append(detection)

        # Increment the track ID counter for the next frame
        track_id_counter += 1
        
    return input_image, detections_for_sort, track_id_counter

def yolo_prediction_plate(img, net):
    images = []
    # 1: Detecção
    input_image, detections = get_detections(img, net)
    # 2: Supressão Não Máxima
    boxes_np, confidences_np, index = non_maximum_supression(input_image, detections)

    # Create an array for SORT input
    detections_for_sort = []

    for ind in index:
        x_min, y_min, width, height = boxes_np[ind]  # Extract coordinates and dimensions
        x_max = x_min + width
        y_max = y_min + height

        detection = [x_min, y_min, x_max, y_max, confidences_np[ind]]  # Append track ID
        detections_for_sort.append(detection)

        

        cropped_detection = img[y_min:y_max, x_min:x_max]
        images.append(cropped_detection)
        
    return images

def is_tracked (track_id) :
    if track_id in tracked:
        return True
    
    return False

def add_tracked (track_id, plate):
    tracked[track_id] = plate

def extract_plate(frame):
    cv2.imshow("Placa", frame)
    cv2.waitKey(0)
    img = rotate_image(frame=frame)
    tess_config = r" --oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"

    if img is not None:        

        saida = pytesseract.image_to_data(img, lang='eng', output_type=Output.DICT, config=tess_config)   
        for i in range(len(saida["text"])):
            if int(saida["conf"][i]) >= 0:

                formatted = ''.join(e for e in saida["text"][i] if e.isalnum() or e == '-')
                print('extract_plate:' , formatted) 
                if len(formatted) == 7:
                    print(formatted, saida["conf"][i])
                    return formatted
                
                    
    return None

def corrigir_orientacao(imagem):
    imagem_corrigida = cv2.rotate(imagem, cv2.ROTATE_90_CLOCKWISE)
    return imagem_corrigida

video_path = './videos/untitled.mp4'
tracker = SortTracker()
cap = cv2.VideoCapture(video_path)
track_id_counter = 0
tracked = {}
# Loop para processar cada quadro
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    #frame = corrigir_orientacao(frame)

    # Chamar a função de predição do YOLO para processar o quadro
    result_img, detections_for_sort, track_id_counter = yolo_predictions(frame, net_vehicle, track_id_counter)
    if detections_for_sort != []:
        
        dets = np.array(detections_for_sort)
        online_targets = tracker.update(dets, None)
        
        for d in online_targets:
            xmin, ymin, xmax, ymax, track_id, _, _ = map(int, d)

            if not is_tracked(track_id):
                plate_images = yolo_prediction_plate(result_img[ymin:ymax, xmin:xmax], net_plate)
                for plate_img in plate_images:
                    
                    plate = extract_plate(frame=plate_img)
                    if plate is not None:
                        add_tracked(track_id, plate)  # Adiciona o objeto rastreado
                        

            # Escreve na imagem a placa e o ID do objeto rastreado
            if track_id in tracked:
                cv2.rectangle(result_img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
                cv2.putText(result_img, f"ID {track_id} : {tracked[track_id]}", (xmin + 10, ymin + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (10, 10,255), 3)
                if not (os.path.isfile(f'./plates_extracted/{tracked[track_id]}' + '.png')):
                    cv2.imwrite(f'./plates_extracted/{tracked[track_id]}' + '.png', result_img)
                #cv2.waitKey(0)
            else:
                cv2.putText(result_img, f"ID {track_id} : ", (xmin + 10, ymin + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)
                cv2.rectangle(result_img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)


    cv2.imshow('main', result_img)
    cv2.waitKey(0)
    # Pressione 'q' para sair do loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
print(tracked)