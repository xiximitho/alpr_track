import re

import cv2
import numpy as np
import pytesseract
from pytesseract import Output
from sort.tracker import SortTracker

net_vehicle = cv2.dnn.readNet('./model/vehicle_novo.onnx')
net_plate = cv2.dnn.readNet('./model/placa.onnx')
WIDTH =  640
HEIGHT = 640



def preprocess_image(frame):
    # Realizar o pré-processamento da imagem (por exemplo, redimensionar, normalizar, etc.)
    processed_frame = cv2.resize(frame, (HEIGHT, WIDTH))
    return processed_frame

def get_detections(img,net):
    # 1.CONVERT IMAGE TO YOLO FORMAT
    image = img #.copy()
    row, col, d = image.shape

    max_rc = max(row,col)
    input_image = np.zeros((max_rc,max_rc,3),dtype=np.uint8)
    input_image[0:row,0:col] = image

    # 2. GET PREDICTION FROM YOLO MODEL
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

def drawings(image,boxes_np,confidences_np,index):
    # 5. Drawings
    for ind in index:
        x,y,w,h =  boxes_np[ind]
        bb_conf = confidences_np[ind]
        conf_text = 'plate: {:.0f}%'.format(bb_conf*100)
        #license_text = extract_text(image,boxes_np[ind])


        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,255),2)
        cv2.rectangle(image,(x,y-30),(x+w,y),(255,0,255),-1)
        cv2.rectangle(image,(x,y+h),(x+w,y+h+25),(0,0,0),-1)


        cv2.putText(image,conf_text,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),1)
        #cv2.putText(image,license_text,(x,y+h+27),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),1)

    return image

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
    copia = frame.copy()
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred_image = gray_image
    #blurred_image = cv2.GaussianBlur(gray_image, (1, 1), 0)
    
    # Aplica uma técnica de limiarização para binarizar a imagem
    _, threshold_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    '''
    # Aplica um filtro de suavização Gaussiano para reduzir o ruído
    kernel = np.ones((1, 1), np.uint8)
    # Realize a operação de abertura para realçar as áreas pretas
    opened_image = cv2.morphologyEx(threshold_image, cv2.MORPH_OPEN, kernel)
    img_erode = cv2.erode(opened_image, kernel, iterations=1)
    img_dilate = cv2.dilate(img_erode, kernel=kernel, iterations=1)
    # Inverte as cores para tornar o texto branco e o fundo preto'''
    
    fator_reducao = 1.8
    # Aplique a redução de cinza multiplicando todos os pixels pelo fator
    imagem_reduzida = np.clip(blurred_image * fator_reducao, 0, 255).astype(np.uint8)

    cv2.imshow("tg", imagem_reduzida)
    cv2.waitKey(0)

    #plate_str = pytesseract.image_to_string(imagem_reduzida,output_type=Output.DICT, config=' -l eng --oem 1 --psm 6 -c min_length=7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890')
    #print("STR:: ", plate_str)
    plate_text = pytesseract.image_to_data(imagem_reduzida, output_type=Output.DICT, config=' -l eng --oem 1 --psm 6 -c min_length=7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890')
    for i in range(len(plate_text["text"])):
        if plate_text["text"][i] != "" and int(plate_text["conf"][i]) > 0:
            #print(f"{plate_text['text'][i]} (Confidence: {plate_text['conf'][i]}%)")
            # Annotate the text and its confidence level
            plate_final = re.sub(r'[^a-zA-Z0-9]', '', plate_text["text"][i])
            if plate_text['conf'][i] > 30 and len(plate_final) == 7:
                text = f"{plate_final} {plate_text['conf'][i]}%)"
                print(f"{text}")
                return text
    
    return None

# Função para corrigir a orientação da imagem
def corrigir_orientacao(imagem):
    imagem_corrigida = cv2.rotate(imagem, cv2.ROTATE_90_CLOCKWISE)
    return imagem_corrigida

video_path = './videos/w2.mp4'
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
                        add_tracked(track_id, plate)
        

            #Escreve na imagem a placa + id
            if track_id in tracked:
                cv2.rectangle(result_img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(result_img, f"ID: {track_id} : {tracked[track_id]}", (xmin, ymin + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                cv2.putText(result_img, f"ID: {track_id} : ", (xmin, ymin + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.rectangle(result_img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    cv2.imshow("Frame", result_img)
    #cv2.waitKey(0)
    # Pressione 'q' para sair do loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
print(tracked)