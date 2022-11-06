import face_recognition as fr
from time import strftime

def reconhece_face(url_foto):
    foto = fr.load_image_file(url_foto)
    rostos = fr.face_encodings(foto)
    if(len(rostos) > 0):
        return True, rostos
    
    return False, []

def frequencia(Horario):

    if int(Horario) >= 10:
        return 1

    elif int(Horario) >= 9:
        return 2
    else:
        return 3



def get_rostos():
    rostos_conhecidos = []
    nomes_dos_rostos = []

    sandro = reconhece_face("./img/sandro.jpg")
    if(sandro[0]):
        rostos_conhecidos.append(sandro[1][0])
        nomes_dos_rostos.append("Sandro")


    ricardo = reconhece_face("./img/ricardo.jpg")
    if (ricardo[0]):
        rostos_conhecidos.append(ricardo[1][0])
        nomes_dos_rostos.append("Ricardo")

    return rostos_conhecidos, nomes_dos_rostos