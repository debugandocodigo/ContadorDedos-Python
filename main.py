import cv2  # pip install opencv-python
import mediapipe as mp  # pip install mediapipe
from mediapipe.python.solutions import hands, drawing_utils

video = cv2.VideoCapture(2)  # Inicia a captura de video

mp_hands = hands.Hands(max_num_hands=1)  # Inicia o detector de mãos
mp_drawing = drawing_utils  # Inicia o desenhador

while True:
    _, frame = video.read()  # Captura um frame
    frame = cv2.flip(frame, 1)  # Inverte o frame
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Converte o frame para RGB
    results = mp_hands.process(img_rgb)  # Processa o frame
    hand_points = results.multi_hand_landmarks  # Pega os pontos de referência das mãos
    h, w, _ = frame.shape  # Pega a altura e largura do frame
    pontos = []  # Lista para armazenar os pontos de referência das mãos
    if hand_points:
        for points in hand_points:
            # Desenha os pontos de referência das mãos
            mp_drawing.draw_landmarks(frame, points, hands.HAND_CONNECTIONS)
            for id, cord in enumerate(points.landmark):  # Percorre os pontos de referência das mãos
                cx, cy = int(cord.x * w), int(cord.y * h)  # Pega a posição dos pontos de referência
                cv2.putText(frame, str(id), (cx, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                pontos.append((cx,cy))  # Adiciona os pontos de referência na lista

            top_point_fingers = [8,12,16,20]  # Pontos de referência do topo dos dedos
            qt_fingers = 0  # Variável para armazenar a quantidade de dedos levantados
            if pontos:
                if pontos[4][0] < pontos[3][0]:  # Verifica se o dedo polegar está levantado
                    qt_fingers += 1
                for x in top_point_fingers:
                   if pontos[x][1] < pontos[x-2][1]:  # Verifica se os dedos estão levantados
                       qt_fingers +=1

            cv2.rectangle(frame, (80, 10), (200,110), (255, 0, 0), -1)
            cv2.putText(frame, str(qt_fingers),(100,100), cv2.FONT_HERSHEY_SIMPLEX,4,(255,255,255), 5)

    cv2.imshow('Imagem', frame)
    cv2.waitKey(1)