from imageai.Detection import ObjectDetection
import os
# https://towardsdatascience.com/object-detection-with-10-lines-of-code-d6cb4d86f606

execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(os.path.join(
    execution_path, "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image=os.path.join(
    execution_path, "images/2.png"), output_image_path=os.path.join(execution_path, "images/2-new.png"))

for x,y in detections:
    
    print(eachObject["name"] , " : " , eachObject["percentage_probability"] )
    # print(eachObject["name"], " : ", eachObject["percentage_probability"])
    # cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # sub_img = image[y:y + h, x:x + w]
    # sub_img = cv2.cvtColor(sub_img, cv2.COLOR_BGR2GRAY)
    # shape = sub_img.shape
    # for row in range(shape[0]):
    #     for col in range(shape[1]):
    #         rnd = random.randint(0, 1)
    #         sub_img[row, col] = rnd * 255

