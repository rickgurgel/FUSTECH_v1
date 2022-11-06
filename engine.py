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

    alisson = reconhece_face("./img/alisson.jpg")
    if (alisson[0]):
        rostos_conhecidos.append(alisson[1][0])
        nomes_dos_rostos.append("Alisson Braga")

    nickolas = reconhece_face("./img/nickolas.jpg")
    if (nickolas[0]):
        rostos_conhecidos.append(nickolas[1][0])
        nomes_dos_rostos.append("Nickolas")

    mauricio = reconhece_face("./img/mauricio.jpg")
    if (mauricio[0]):
        rostos_conhecidos.append(mauricio[1][0])
        nomes_dos_rostos.append("Mauricio")

    jonathan = reconhece_face("./img/jonathan.jpg")
    if (jonathan[0]):
        rostos_conhecidos.append(jonathan[1][0])
        nomes_dos_rostos.append("Jonathan")

    marcelo = reconhece_face("./img/marcelo.jpg")
    if (marcelo[0]):
        rostos_conhecidos.append(marcelo[1][0])
        nomes_dos_rostos.append("Marcelo")

    avila = reconhece_face("./img/avila.jpg")
    if (avila[0]):
        rostos_conhecidos.append(avila[1][0])
        nomes_dos_rostos.append("Avila")

    wesley = reconhece_face("./img/wesley.jpg")
    if (wesley[0]):
        rostos_conhecidos.append(wesley[1][0])
        nomes_dos_rostos.append("Wesley")

    return rostos_conhecidos, nomes_dos_rostos