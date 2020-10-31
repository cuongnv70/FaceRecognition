''''
Đào tạo nhiều khuôn mặt có trong dataset
	==> Mỗi khuôn mặt phải có một ID số nguyên bằng số duy nhất là 1, 2, 3, v.v.
'''

import cv2
import numpy as np
from PIL import Image
import os

# Đường dẫn ảnh khuôn mặt
path = 'dataset'

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Chức năng lấy hình ảnh và dữ liệu nhãn
def getImagesAndLabels(path, directory=None):

    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    faceSamples=[]
    ids = []

    for imagePath in imagePaths:

        PIL_img = Image.open(imagePath).convert('L') # Chuyển ảnh đã chụp sang thang độ xám
        img_numpy = np.array(PIL_img,'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)

        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)

    return faceSamples,ids

print ("\n [!] Đang đào tạo các khuôn mặt. Vui lòng đợi ...")
faces, ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))

# Lưu mô hình vào trainer/trainer.yml
recognizer.write('trainer/trainer.yml')

#In số lượng khuôn mặt được đào tạo và kết thúc chương trình
print("\n [!] {0} Khuôn mặt được đào tạo. Thoát khỏi chương trình".format(len(np.unique(ids))))
