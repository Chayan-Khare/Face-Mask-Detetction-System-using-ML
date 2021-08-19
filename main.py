from tensorflow.keras.models import load_model
import cv2
import numpy as np
import tkinter
from tkinter import messagebox
import smtplib
import imghdr
from email.message import EmailMessage
import os

root = tkinter.Tk()
root.withdraw()

model = load_model('Face_Mask_Model.h5')

mask_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

vid_source = cv2.VideoCapture(0)

# Dictionaries containing detail of wearing mask as well as putting rectangle around face
text_dict = {0: "Mask ON", 1: "No Mask"}
rect_color = {0: (0, 255, 0), 1: (0, 0, 255)}

Subject = "Breach of Vistor Policy Detected"
Text = "Vistor without face mask detected. Check camerato recognize vistor"

while (True):
    ret, img = vid_source.read()
    img = np.array(img[:, ::-1])  # cast as array
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = mask_classifier.detectMultiScale(grayscale_img, 1.3, 5)
    for (x, y, w, h) in faces:
        face_img = grayscale_img[y:y + w, x:x + w]
        resized_img = cv2.resize(face_img, (112, 112))
        normal_png = resized_img / 255.0
        reshaped_img = np.reshape(normal_png, (1, 112, 112, 1))
        result = model.predict(reshaped_img)

        label = np.argmax(result, axis=1)[0]

        cv2.rectangle(img, (x, y), (x + w, y + h), rect_color[label], 2)
        cv2.rectangle(img, (x, y - 40), (x + w, y), rect_color[label], -1)
        cv2.putText(img, text_dict[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        if (label == 1):
           cv2.imwrite("NewPicture.jpg", img)
            messagebox.showwarning("Warning", "Visitor Policy Violated")

            Sender_Email = "Enter your email"
            Reciever_Email = "Enter Sender Email"
            Sender_Password = "Enter Your Password"
            Password = "Enter Your Password"
            newMessage = EmailMessage()
            newMessage['Subject'] = "Warning Visitor Policy Violated"
            newMessage['From'] = Sender_Email
            newMessage['To'] = Reciever_Email
            newMessage.set_content('Visiter without face mask detected.')
            with open('NewPicture.jpg', 'rb') as f:
                image_data = f.read()
                image_type = imghdr.what(f.name)
                image_name = f.name
            newMessage.add_attachment(image_data, maintype='image', subtype=image_type, filename=image_name)
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:

                smtp.login(Sender_Email, Password)
                smtp.send_message(newMessage)


          

        else:
            pass
            break
    cv2.imshow('Live Feed', img)
    key = cv2.waitKey(1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
folder_path = (r'C:\Filepath')
#using listdir() method to list the files of the folder
test = os.listdir(folder_path)
#taking a loop to remove all the images
#using ".png" extension to remove only png images
#using os.remove() method to remove the files
for images in test:
    if images.endswith(".jpg"):
        os.remove(os.path.join(folder_path, images))
cv2.destroyAllWindows()
