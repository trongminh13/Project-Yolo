from ultralytics import YOLO
import cv2
model = YOLO('../Yolo-Weights/yolov8l.pt')
results = model("Images/3.png")
annotated_image = results[0].plot()
cv2.imshow("YOLO Detection Result", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()