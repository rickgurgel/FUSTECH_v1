#A Gender and Age Detection program by Mahesh Sawant

import dlib
import sys
import cv2
import argparse
import numpy as np
import face_recognition as fr
import time
from engine import get_rostos
from scipy.spatial import distance as dist
from threading import Thread
import playsound
import queue
import pyttsx3
import PySimpleGUI as sg
import pyautogui

engine = pyttsx3.init()

rostos_conhecidos, nomes_dos_rostos = get_rostos()

count = 0
estagio = 0

pessoas = []
opacity = 1
pos = (10, 20)

FACE_DOWNSAMPLE_RATIO = 1.5
RESIZE_HEIGHT = 460

thresh = 0.27
modelPath = "models/shape_predictor_70_face_landmarks.dat"
sound_path = "alarm.wav"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(modelPath)

leftEyeIndex = [36, 37, 38, 39, 40, 41]
rightEyeIndex = [42, 43, 44, 45, 46, 47]

blinkCount = 0
drowsy = 0
state = 0
count = 0
menor = 0
blinkTime = 1.0  # 150ms 0.1
drowsyTime = 1.0  # 1200ms 1.5
ALARM_ON = False
GAMMA = 1.5
threadStatusQ = queue.Queue()

invGamma = 1.0 / GAMMA
table = np.array([((i / 255.0) ** invGamma) * 255 for i in range(0, 256)]).astype("uint8")

def janela_iniciar():
    sg.theme('Black')
    layout = [
        [sg.Button('Iniciar')]
    ]
    return sg.Window('Iniciando...', layout = layout, modal = True, finalize = True)

# def janela_fechar():
#    sg.theme('Reddit')
#    layout = [
#       [sg.Button('Fechar')]
#    ]
#    return sg.Window('Fechando...', layout = layout, finalize = True)

janela1 = janela_iniciar()

def gamma_correction(image):
    return cv2.LUT(image, table)


def histogram_equalization(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(gray)


def soundAlert(sound_path, threadStatusQ):
    while True:
        if not threadStatusQ.empty():
            FINISHED = threadStatusQ.get()
            if FINISHED:
                break
        playsound.playsound(sound_path)


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)

    return ear


def checkEyeStatus(landmarks):
    mask = np.zeros(frame.shape[:2], dtype=np.float32)

    hullLeftEye = []
    for i in range(0, len(leftEyeIndex)):
        hullLeftEye.append((landmarks[leftEyeIndex[i]][0], landmarks[leftEyeIndex[i]][1]))

    cv2.fillConvexPoly(mask, np.int32(hullLeftEye), 255)

    hullRightEye = []
    for i in range(0, len(rightEyeIndex)):
        hullRightEye.append((landmarks[rightEyeIndex[i]][0], landmarks[rightEyeIndex[i]][1]))

    cv2.fillConvexPoly(mask, np.int32(hullRightEye), 255)

    #############################################################################
    leftEAR = eye_aspect_ratio(hullLeftEye)
    rightEAR = eye_aspect_ratio(hullRightEye)

    ear = (leftEAR + rightEAR) / 2.0
    #############################################################################

    eyeStatus = 1  # 1 -> Open, 0 -> closed
    if (ear < thresh):
        eyeStatus = 0

    return eyeStatus


def checkBlinkStatus(eyeStatus):
    global state, blinkCount, drowsy
    if (state >= 0 and state <= falseBlinkLimit):
        if (eyeStatus):
            state = 0

        else:
            state += 1

    elif (state >= falseBlinkLimit and state < drowsyLimit):
        if (eyeStatus):
            blinkCount += 1
            state = 0

        else:
            state += 1


    else:
        if (eyeStatus):
            state = 0
            drowsy = 1
            blinkCount += 1

        else:
            drowsy = 1


def getLandmarks(im):
    imSmall = cv2.resize(im, None,
                         fx=1.0 / FACE_DOWNSAMPLE_RATIO,
                         fy=1.0 / FACE_DOWNSAMPLE_RATIO,
                         interpolation=cv2.INTER_LINEAR)

    rects = detector(imSmall, 0)
    if len(rects) == 0:
        return 0

    newRect = dlib.rectangle(int(rects[0].left() * FACE_DOWNSAMPLE_RATIO),
                             int(rects[0].top() * FACE_DOWNSAMPLE_RATIO),
                             int(rects[0].right() * FACE_DOWNSAMPLE_RATIO),
                             int(rects[0].bottom() * FACE_DOWNSAMPLE_RATIO))

    points = []
    [points.append((p.x, p.y)) for p in predictor(im, newRect).parts()]
    return points

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
    window, event, values = sg.read_all_windows()
    video = cv2.VideoCapture(args.image if args.image else 0)
    padding=20
    if window == janela1 and event == sg.WIN_CLOSED:
        janela1.hide
        break
    if window == janela1 and event == 'Iniciar':
        while cv2.waitKey(1)<0 and estagio == 0:
            hasFrame, frame = video.read()

            waterImg = cv2.imread('img/uniateneu.png', -1)

            overlay = transparentOverlay(frame, waterImg, pos)
            cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)

            rgb_frame = frame[:, :, ::-1]

            if not hasFrame:
                cv2.waitKey()
                break

            resultImg, faceBoxes = highlightFace(faceNet,frame)
            if not faceBoxes:
                print("No face detected")

            for faceBox in faceBoxes:
                face = frame[max(0,faceBox[1]-padding):
                               min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
                               :min(faceBox[2]+padding, frame.shape[1]-1)]

                blob = cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)

                ageNet.setInput(blob)
                agePreds=ageNet.forward()
                age=ageList[agePreds[0].argmax()]

                if ageList.index(age) >= ageList.index("(18-24)"):
                    cv2.putText(resultImg, 'Conferindo Idade', (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                    (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.namedWindow("Detectando idade", cv2.WND_PROP_FULLSCREEN)
                    cv2.setWindowProperty("Detectando idade", cv2.WND_PROP_FULLSCREEN,
                                          cv2.WINDOW_FULLSCREEN)
                    cv2.imshow("Detectando idade", resultImg)
                    count += 1

                else:
                    cv2.putText(resultImg, 'Menor de Idade', (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                    (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.namedWindow("Detectando idade", cv2.WND_PROP_FULLSCREEN)
                    cv2.setWindowProperty("Detectando idade", cv2.WND_PROP_FULLSCREEN,
                                          cv2.WINDOW_FULLSCREEN)
                    cv2.imshow("Detectando idade", resultImg)
                    menor += 1

            if count > 50:
                estagio = 1
                count = 0
                engine.say(f"Usuário Maior de Idade.")
                engine.runAndWait()
                engine.stop()
                count_ap = 0
                count_rep = 0

            if menor > 60:
                count = 0
                menor = 0
                engine.say(f"Usuário Menor de Idade.")
                engine.runAndWait()
                engine.stop()



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
                    cv2.putText(frame, 'AUTORIZANDO', (left + 6, top - 6), font, 0.7, (0, 0, 0), 1)

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

                cv2.namedWindow("Life2Coding", cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty("Life2Coding", cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN)
                cv2.imshow('Life2Coding', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    start = 1
                    break

            if count_ap > 7:
                count_ap = 0
                count_rep = 0
                estagio = 2

            if count_rep > 10:
                count_ap = 0
                count_rep = 0
                estagio = 0
                cv2.destroyAllWindows()

        if estagio == 2:
            engine.say(f"Motorista {nome} cadastrado.")
            engine.runAndWait()
            engine.stop()
        else:
            engine.say(f"Motorista não cadastrado, procure a administração.")
            engine.runAndWait()
            engine.stop()

        for i in range(10):
            ret, frame = video.read()

        totalTime = 0.0
        validFrames = 0
        dummyFrames = 100

        while (validFrames < dummyFrames):
            validFrames += 1
            t = time.time()
            ret, frame = video.read()
            height, width = frame.shape[:2]
            IMAGE_RESIZE = np.float32(height) / RESIZE_HEIGHT
            frame = cv2.resize(frame, None,
                               fx=1 / IMAGE_RESIZE,
                               fy=1 / IMAGE_RESIZE,
                               interpolation=cv2.INTER_LINEAR)

            adjusted = histogram_equalization(frame)

            landmarks = getLandmarks(adjusted)
            timeLandmarks = time.time() - t

            if landmarks == 0:
                validFrames -= 1
                cv2.putText(frame, "Face não detectada... Ajuste a iluminação...", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                            (0, 0, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, "or decrease FACE_DOWNSAMPLE_RATIO", (10, 50), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                            (0, 0, 255), 1, cv2.LINE_AA)
                cv2.namedWindow("Blink Detection Demo", cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty("Blink Detection Demo", cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN)
                cv2.imshow("Blink Detection Demo", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    sys.exit()

            else:
                totalTime += timeLandmarks


        print("CALIBRADO")

        spf = totalTime / dummyFrames
        print("Current SPF (seconds per frame) is {:.2f} ms".format(spf * 1000))

        drowsyLimit = drowsyTime / spf
        falseBlinkLimit = blinkTime / spf
        print("drowsy limit: {}, false blink limit: {}".format(drowsyLimit, falseBlinkLimit))

        while estagio == 2:
            if __name__ == "__main__":
                vid_writer = cv2.VideoWriter('output-low-light-2.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 15,
                                             (frame.shape[1], frame.shape[0]))
                while (1):
                    try:
                        t = time.time()
                        ret, frame = video.read()

                        waterImg = cv2.imread('img/uniateneu.png', -1)

                        overlay = transparentOverlay(frame, waterImg, pos)
                        cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)

                        rgb_frame = frame[:, :, ::-1]

                        height, width = frame.shape[:2]
                        IMAGE_RESIZE = np.float32(height) / RESIZE_HEIGHT
                        frame = cv2.resize(frame, None,
                                           fx=1 / IMAGE_RESIZE,
                                           fy=1 / IMAGE_RESIZE,
                                           interpolation=cv2.INTER_LINEAR)

                        # adjusted = gamma_correction(frame)
                        adjusted = histogram_equalization(frame)

                        landmarks = getLandmarks(adjusted)
                        if landmarks == 0:
                            validFrames -= 1
                            cv2.putText(frame, "Face não detectada... Ajuste a iluminação...", (10, 30),
                                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                            cv2.putText(frame, "or decrease FACE_DOWNSAMPLE_RATIO", (10, 50), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                                        (0, 0, 255), 1, cv2.LINE_AA)
                            cv2.namedWindow("Blink Detection Demo", cv2.WND_PROP_FULLSCREEN)
                            cv2.setWindowProperty("Blink Detection Demo", cv2.WND_PROP_FULLSCREEN,
                                                  cv2.WINDOW_FULLSCREEN)
                            cv2.imshow("Blink Detection Demo", frame)
                            if cv2.waitKey(1) & 0xFF == 27:
                                break
                            continue

                        eyeStatus = checkEyeStatus(landmarks)
                        checkBlinkStatus(eyeStatus)

                        for i in range(0, len(leftEyeIndex)):
                            cv2.circle(frame, (landmarks[leftEyeIndex[i]][0], landmarks[leftEyeIndex[i]][1]), 1,
                                       (0, 0, 255), -1, lineType=cv2.LINE_AA)

                        for i in range(0, len(rightEyeIndex)):
                            cv2.circle(frame, (landmarks[rightEyeIndex[i]][0], landmarks[rightEyeIndex[i]][1]), 1,
                                       (0, 0, 255), -1, lineType=cv2.LINE_AA)

                        if drowsy:
                            cv2.putText(frame, "ESTADO DE SONOLENCIA", (70, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255),
                                        2, cv2.LINE_AA)
                            if not ALARM_ON:
                                ALARM_ON = True
                                threadStatusQ.put(not ALARM_ON)
                                thread = Thread(target=soundAlert, args=(sound_path, threadStatusQ,))
                                thread.setDaemon(True)
                                thread.start()


                        else:
                            cv2.putText(frame, "Monitorando.", (390, 80), cv2.FONT_HERSHEY_COMPLEX, 0.8,
                                        (0, 0, 255), 2, cv2.LINE_AA)
                            # (0, 400)
                            ALARM_ON = False
                        cv2.namedWindow("Blink Detection Demo", cv2.WND_PROP_FULLSCREEN)
                        cv2.setWindowProperty("Blink Detection Demo", cv2.WND_PROP_FULLSCREEN,
                                              cv2.WINDOW_FULLSCREEN)
                        cv2.imshow("Blink Detection Demo", frame)
                        vid_writer.write(frame)

                        k = cv2.waitKey(1)
                        if ALARM_ON:
                            state = 0
                            drowsy = 0
                            ALARM_ON = False
                            threadStatusQ.put(not ALARM_ON)

                        elif k == 27:
                            estagio = 0
                            break


                    except Exception as e:
                        print(e)

                video.release()
                vid_writer.release()
                cv2.destroyAllWindows()