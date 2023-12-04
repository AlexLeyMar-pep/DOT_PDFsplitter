import time

import pytesseract
from pdf2image import convert_from_path
import matplotlib.pyplot as plt
import pypdf
import cv2
import os
import PyPDF2
from PIL import Image
import easyocr
import torch
import warnings


def try2():
    path_docs = r"C:\Users\80943848\Pepsico\PFNA HR Strategy, Staffing, Technology & Transformation - Kelmar Test Files"
    #path_docs = r"C:\Users\80943848\Downloads\Undetected_faces"

    files = 0
    matches = 0

    for filename in os.listdir(path_docs):
        if filename.endswith('.pdf'):
            start = time.time()
            with open(os.path.join(path_docs, filename)) as file:
                found_flag = False
                #Creating folder
                gpid = str((file.name).split('\\')[-1].split(".")[0])
                folder = create_folder(gpid)
                files += 1
                #Reading PDF
                path = file.name
                pdf = pypdf.PdfReader(open(path, "rb"))
                try:
                    images = convert_from_path(path)
                except Exception as e:
                    print(e)
                    pass
                print("\n"+path, "<----")
                for i in range(len(pdf.pages)):
                    page = pdf.pages[i].page_number
                    temp_img_path = "Page " + str(page+1) + " in " + path.split('\\')[-1] + ".jpg"
                    image = images[int(page)]
                    image.save(temp_img_path)
                    print(page, end=" ")
                    time.sleep(.85)
                    if opencv(temp_img_path) is True:
                        found_flag = True
                        absolute_path = str(folder) + "\\" + "driver_license.jpg"
                        image.save(absolute_path)
                        matches += 1
                        print("Match in page " + str(page+1) + " in " + path.split('\\')[-1])
                        break

                #RE RUN PDF USING TORCH
                if not found_flag:
                    print("\nStarting torch")
                    for i in range(len(pdf.pages)):
                        page = pdf.pages[i].page_number
                        temp_img_path = "Page " + str(page + 1) + " in " + path.split('\\')[-1] + ".jpg"
                        image = images[int(page)]
                        image.save(temp_img_path)
                        #print(page, end=" ")
                        time.sleep(.85)
                        matches = ["CDL", "ILLNOIS", "ILLINOIS", "Secretary of State", "ILUNOIS"]
                        text = recognize_text_torch(temp_img_path)
                        warnings.filterwarnings('default')
                        #print(len(text))
                        if len(text) > 80:
                            os.remove(temp_img_path)
                        for word in matches:
                            if word in text:
                                #img = cv2.imread(temp_img_path)
                                #show_image(img)
                                print("FACE FOUND!")
                                print("Match in page " + str(page + 1) + " in " + path.split('\\')[-1])
                                found_flag = True
                                os.remove(temp_img_path)
                                break
                        if found_flag:
                            absolute_path = str(folder) + "\\" + "driver_license.jpg"
                            image.save(absolute_path)
                            os.remove(temp_img_path)
                            break



            end = time.time()
            print("Time:", round(end-start, 2), "seconds")

    print(str(matches), "matches out of", str(files), "files")
    print(str(round((matches/files)*100, 2)), "accuracy")



def opencv(path):

    img = cv2.imread(path)

    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face = use_classifier(img, gray_image, "haarcascade_frontalface_default.xml")

    #eyes = use_classifier(img, gray_image, "haarcascade_eye.xml")

    face_alt = use_classifier(img, gray_image, "haarcascade_frontalface_alt.xml")

    face_alt_2 = use_classifier(img, gray_image, "haarcascade_frontalface_alt2.xml")

    try:
        if face.shape[0] > 5 or face_alt.shape[0] > 2 or face_alt_2.shape[0] > 3:       #ECONTRAR 7 de 9
            #print("\nFalse" + str(face.shape), str(face_alt.shape), str(face_alt_2.shape))
            #show_image(img)
            os.remove(path)
            return False

        #DETECT TEXT WITH OCR
        target = ["403", "DT"]


        text = recognize_text_OCR(path)
        #print(type(text))
        #print(text)
        for i in target:
            if i in text:
                print("False postive", end= ' ')
                #show_image(img)
                os.remove(path)
                return False


        #show_image(img)
        #print(face.shape, face_alt.shape, face_alt_2.shape)
        print("\nFACE FOUND!")
        os.remove(path)
        return True

    except Exception as e:
        os.remove(path)
        return False


def show_image(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 5))
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()


# 'GeeksForGeeks' code
def create_folder(name):
    # Directory
    directory = str(name)

    # Parent Directory path
    parent_dir = r"C:\Users\80943848\PycharmProjects\DOT_PDFsplitter"

    # Path
    path = os.path.join(parent_dir, directory)

    # Create the directory
    os.mkdir(path)
    #print("Directory '% s' created" % directory)

    return path


def test(path):
    img = cv2.imread(path)

    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face = use_classifier(img, gray_image, "haarcascade_frontalface_default.xml")
    #eyes = use_classifier(img, gray_image, "haarcascade_eye.xml")
    #l_eye = use_classifier(img, gray_image, "haarcascade_lefteye_2splits.xml")
    #r_eye = use_classifier(img, gray_image, "haarcascade_righteye_2splits.xml")
    face_alt = use_classifier(img, gray_image, "haarcascade_frontalface_alt.xml")
    face_alt_2 = use_classifier(img, gray_image, "haarcascade_frontalface_alt2.xml")
    #face_alt_tree = use_classifier(img, gray_image, "haarcascade_frontalface_alt_tree.xml")


    print(face_alt.shape)
    print(face_alt_2.shape)
    print("face: " + str(face.shape))

    show_image(img)




def use_classifier(img, gray_image,classifier_name):
    object_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + classifier_name
    )

    object = object_classifier.detectMultiScale(
        gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )

    for (x, y, w, h) in object:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)

    return object

def recognize_text_torch(path): #RECEIVES IMG
    #OCR
    reader = easyocr.Reader(['en'])
    result = reader.readtext(path, detail=0)
    return result #Array

def recognize_text_OCR(path): #RECEIVES IMG
    text = str(((pytesseract.image_to_string((Image.open(path))))))
    return text #String

def main():
    #opencv(r"C:\Users\80943848\PycharmProjects\DOT_PDFsplitter\Page_41.jpg")
    #test(r"C:\Users\80943848\Downloads\uld3.jpg")
    try2()
    #recognize_text_torch(r"C:\Users\80943848\Downloads\uld.jpg")


main()