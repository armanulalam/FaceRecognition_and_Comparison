import cv2
from test import SimpleFacerec

obj = SimpleFacerec()
obj.load_encoding_images("images/")

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
cap.set(10, 150)

names = []

while True:
    success, image = cap.read()

    face_locations, face_names = obj.detect_known_faces(image)

    for face_loc, name in zip(face_locations, face_names):
        top, right, bottom, left = [face_loc[i] for i in range(4)]
        cv2.putText(image, name, (left, top - 15), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 3)
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

        if face_names not in names:
            if face_names == ["Unknown"]:
                continue
            names.append(face_names)
        
        # print(name)

    cv2.imshow("Image", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(names)

cap.release()
cv2.destroyAllWindows()