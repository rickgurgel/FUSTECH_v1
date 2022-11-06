#A Gender and Age Detection program by Mahesh Sawant

import cv2
import argparse
import numpy as np
import face_recognition as fr

from engine import get_rostos

import pyttsx3
engine = pyttsx3.init()

rostos_conhecidos, nomes_dos_rostos = get_rostos()

count = 0
estagio = 0

pessoas = []
opacity = 1
pos = (10, 20)

def transparentOverlay(src, overlay, pos=(0, 0), scale=1):
    overlay = cv2.resize(overlay, (0, 0), fx=scale, fy=scale)
    h, w, _ = overlay.shape  # Size of foreground
    rows, cols, _ = src.shape  # Size of background Image
    y, x = pos[0], pos[1]  # Position of foreground/overlay image
    # loop over all pixels and apply the blending equation
    for i in range(h):
        for j in range(w):
            if x + i >= rows or y + j >= cols:
                continue
            alpha = float(overlay[i][j][3] / 255.0)  # read the alpha channel
            src[x + i][y + j] = alpha * overlay[i][j][:3] + (1 - alpha) * src[x + i][y + j]
    return src


def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn=frame.copy()
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections=net.forward()
    faceBoxes=[]
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn,faceBoxes


parser=argparse.ArgumentParser()
parser.add_argument('--image')

args=parser.parse_args()

faceProto="opencv_face_detector.pbtxt"
faceModel="opencv_face_detector_uint8.pb"
ageProto="age_deploy.prototxt"
ageModel="age_net.caffemodel"


MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.905847746)


ageList=['(0-4)', '(5-10)', '(11-17)', '(18-24)', '(19-24)', '(26-33)', '(34-43)', '(50-100)']

faceNet = cv2.dnn.readNet(faceModel,faceProto)
ageNet = cv2.dnn.readNet(ageModel,ageProto)

while True:
    video = cv2.VideoCapture(args.image if args.image else 0)
    padding=20

    while cv2.waitKey(1)<0 and estagio == 0:
        hasFrame,frame=video.read()
        if not hasFrame:
            cv2.waitKey()
            break

        resultImg,faceBoxes=highlightFace(faceNet,frame)
        if not faceBoxes:
            print("No face detected")

        for faceBox in faceBoxes:
            face=frame[max(0,faceBox[1]-padding):
                           min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
                           :min(faceBox[2]+padding, frame.shape[1]-1)]

            blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)

            ageNet.setInput(blob)
            agePreds=ageNet.forward()
            age=ageList[agePreds[0].argmax()]

            if ageList.index(age) >= ageList.index("(18-24)"):
                cv2.putText(resultImg, 'Maior de Idade', (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow("Detectando idade", resultImg)
                count += 1

            else:
                cv2.putText(resultImg, 'Menor de Idade', (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow("Detectando idade", resultImg)

        if count > 60:
            estagio = 1
            count = 0
            cv2.destroyAllWindows()

    video = cv2.VideoCapture(0)

    engine.say(f"Usuário Maior de Idade.")
    engine.runAndWait()
    engine.stop()
    count_ap = 0
    count_rep = 0

    while estagio == 1:

        pessoa = False
        ret, frame = video.read()

        waterImg = cv2.imread('img/uniateneu.png', -1)

        overlay = transparentOverlay(frame, waterImg, pos)
        cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)

        rgb_frame = frame[:, :, ::-1]

        localizacao_dos_rostos = fr.face_locations(rgb_frame)
        rosto_desconhecidos = fr.face_encodings(rgb_frame, localizacao_dos_rostos)

        for (top, right, bottom, left), rosto_desconhecido in zip(localizacao_dos_rostos, rosto_desconhecidos):
            resultados = fr.compare_faces(rostos_conhecidos, rosto_desconhecido)

            face_distances = fr.face_distance(rostos_conhecidos, rosto_desconhecido)

            melhor_id = np.argmin(face_distances)

            if resultados[melhor_id] == True:
                nome = nomes_dos_rostos[melhor_id]
                for i in range(len(pessoas)):
                    if pessoas[i] == nome:
                        pessoa = True
                        count_ap += 1
                        break

                if not pessoa:
                    pessoas.append(nome)

                # Acima
                cv2.rectangle(frame, (left, top - 35), (right, top), (0, 255, 0), cv2.FILLED)
                font = cv2.FONT_HERSHEY_SIMPLEX
                # Autorizado
                cv2.putText(frame, 'AUTORIZADO', (left + 6, top - 6), font, 0.7, (0, 0, 0), 1)

            else:
                # Acima
                cv2.rectangle(frame, (left, top - 35), (right, top), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_SIMPLEX
                # Autorizado
                cv2.putText(frame, 'BLOQUEADO', (left + 6, top - 6), font, 0.7, (0, 0, 0), 1)
                nome = "Desconhecido"
                count_rep += 1

            # Ao redor do rosto
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 0), 2)

            # Embaixo
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_SIMPLEX

            # Texto com nome
            cv2.putText(frame, nome, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            cv2.imshow('Life2Coding', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                start = 1
                break

        if count_ap > 10:
            count_ap = 0
            count_rep = 0
            estagio = 2
            cv2.destroyAllWindows()

        if count_rep > 15:
            count_ap = 0
            count_rep = 0
            estagio = 0
            cv2.destroyAllWindows()

    if estagio == 2:
        video = cv2.VideoCapture(0)
        engine.say(f"Motorista cadastrado.")
        engine.runAndWait()
        engine.stop()
    else:
        engine.say(f"Motorista não cadastrado, procure a administração.")
        engine.runAndWait()
        engine.stop()

    while estagio == 2:
        video.read()
        print("Estagio 3")