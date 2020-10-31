import cv2
import os

cam = cv2.VideoCapture(0)
cam.set(3, 640) #Đặt chiều rộng video
cam.set(4, 480) #Đặt chiều cao video

#Đảm bảo 'haarcascade_frontalface_default.xml' nằm trong cùng thư mục với mã này
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

face_id = input('\n Nhập vào số ID người dùng : ')

print("\n [!] Đang khởi tạo chụp khuôn mặt. Nhìn vào máy ảnh và đợi ...")
# Khởi tạo số lượng khuôn mặt lấy mẫu riêng lẻ
count = 0

#Bắt đầu phát hiện khuôn mặt và tiến hành chụp 20 ảnh
while(True):

    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
        count += 1

        # Lưu hình ảnh đã chụp vào thư mục dataset với định dạng jpg
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
        cv2.imshow('Camera Set Data', img)

    k = cv2.waitKey(100) & 0xff # Bấm 'ESC' để thoát camera
    if k == 27:
        break
    elif count >= 20: # Chụp đủ 20 ảnh sẽ dừng chụp
         break

# Dọn dẹp
print("\n [!] Mẫu đã lấy xong, thoát khỏi chương trình.")
cam.release()
cv2.destroyAllWindows()
