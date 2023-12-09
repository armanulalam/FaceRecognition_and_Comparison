import cv2
import face_recognition

img = cv2.imread("images\Arman2.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (350,350))
face_locations = face_recognition.face_locations(img)
# print(face_locations)
imgEncoding1 = face_recognition.face_encodings(img, face_locations)[0]

img2 = cv2.imread("images\Arman1.jpg")
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img2 = cv2.resize(img2, (350,350))
face_locations2 = face_recognition.face_locations(img2)
# print(face_locations2)
imgEncoding2 = face_recognition.face_encodings(img2, face_locations2)[0]

result = face_recognition.compare_faces([imgEncoding1], imgEncoding2)

# print(type(result[0]))

if result[0] == bool(True):
    print("These persons are same.")
elif result[0] == bool(False):
    print("These persons are different.")

cv2.imshow("Image", img)
cv2.imshow("Image 2", img2)
cv2.waitKey(0)