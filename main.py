# !/usr/bin/env python

'''
Calculates Region of Interest(ROI) by receiving points from mouse event and transform prespective so that
we can have top view of scene or ROI. This top view or bird eye view has the property that points are
distributed uniformally horizontally and vertically(scale for horizontal and vertical direction will be
 different). So for bird eye view points are equally distributed, which was not case for normal view.
YOLO V3 is used to detect humans in frame and by calculating bottom center point of bounding boxe around humans,
we transform those points to bird eye view. And then calculates risk factor by calculating distance between
points and then drawing birds eye view and drawing bounding boxes and distance lines between boxes on frame.
'''


# imports
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import cv2
import numpy as np
import time
import argparse
import imutils
import os
import urllib
import insightface
# own modules
import utills, plot

confid = 0.5
thresh = 0.5
mouse_pts = []
FACE_DETECTION_LIKELIHOOD_THRESHOLD = 0.5
FACE_DETECTION_SCALE = 1.0 #ESP32:1.0



# Function to get points for Region of Interest(ROI) and distance scale. It will take 8 points on first frame using mouse click
# event.First four points will define ROI where we want to moniter social distancing. Also these points should form parallel
# lines in real world if seen from above(birds eye view). Next 3 points will define 6 feet(unit length) distance in
# horizontal and vertical direction and those should form parallel lines with ROI. Unit length we can take based on choice.
# Points should pe in pre-defined order - bottom-left, bottom-right, top-right, top-left, point 5 and 6 should form
# horizontal line and point 5 and 7 should form verticle line. Horizontal and vertical scale will be different.

# Function will be called on mouse events

def get_mouse_points(event, x, y, flags, param):
    global mouse_pts
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(mouse_pts) < 4:
            cv2.circle(image, (x, y), 5, (0, 0, 255), 10)
        else:
            cv2.circle(image, (x, y), 5, (255, 0, 0), 10)

        if len(mouse_pts) >= 1 and len(mouse_pts) <= 3:
            cv2.line(image, (x, y), (mouse_pts[len(mouse_pts) - 1][0], mouse_pts[len(mouse_pts) - 1][1]), (70, 70, 70),
                     2)
            if len(mouse_pts) == 3:
                cv2.line(image, (x, y), (mouse_pts[0][0], mouse_pts[0][1]), (70, 70, 70), 2)

        if "mouse_pts" not in globals():
            mouse_pts = []
        mouse_pts.append((x, y))


def detect_and_predict_mask(frame, faceNet, maskNet):
    # grab the dimensions of the frame
    (h, w) = frame.shape[:2]
    # blob = cv2.dnn.blobFromImage(frame, 1.0, (300,300), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    bboxes, landmarks = faceNet.detect(frame, threshold=FACE_DETECTION_LIKELIHOOD_THRESHOLD, scale=FACE_DETECTION_SCALE)

    # initialize our list of faces, their corresponding locations, and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []

    # loop over the detections
    for i in range(0, bboxes.shape[0]):
        # extract the confidence (i.e., probability) associated with the detection
        face = bboxes[i]
        confidence = face[4]

        # filter out weak bboxes by ensuring the confidence is greater than the minimum confidence
        if confidence > args["confidence"]:
            # compute the (x, y)-coordinates of the bounding box for the object
            # box = bboxes[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = face[0:4].astype("int") #box.astype("int")

            # ensure the bounding boxes fall within the dimensions of the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # add the face and bounding boxes to their respective lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        # for faster inference we'll make batch predictions on *all*
        # faces at the same time rather than one-by-one predictions
        # in the above `for` loop
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    # return a 2-tuple of the face locations and their corresponding
    # locations
    return (locs, preds)


def calculate_social_distancing(vid_path, net, ln1):
    count = 0
    vs = cv2.VideoCapture(vid_path)

    points = []
    global image

    while True:

        (grabbed, frame) = vs.read()
        frame = imutils.resize(frame, width=500)  # if changing width, must also change offset of face

        if not grabbed:
            print('here')
            break
        (H, W) = frame.shape[:2]

        # first frame will be used to draw ROI and horizontal and vertical 180 cm distance(unit length in both directions)
        if count == 0:
            while True:
                image = frame
                cv2.imshow("image", image)
                cv2.waitKey(1)
                if len(mouse_pts) == 8:
                    cv2.destroyWindow("image")
                    start1=time.time()
                    break

            points = mouse_pts

        if count % 50 == 0:
            (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
            # Using first 4 points or coordinates for perspective transformation. The region marked by these 4 points are
            # considered ROI. This polygon shaped ROI is then warped into a rectangle which becomes the bird eye view.
            # This bird eye view then has the property property that points are distributed uniformally horizontally and
            # vertically(scale for horizontal and vertical direction will be different). So for bird eye view points are
            # equally distributed, which was not case for normal view.
            src = np.float32(np.array(points[:4]))
            dst = np.float32([[0, H], [W, H], [W, 0], [0, 0]])
            prespective_transform = cv2.getPerspectiveTransform(src, dst)

            # using next 3 points for horizontal and vertical unit length(in this case 180 cm)
            pts = np.float32(np.array([points[4:7]]))
            warped_pt = cv2.perspectiveTransform(pts, prespective_transform)[0]

            # since bird eye view has property that all points are equidistant in horizontal and vertical direction.
            # distance_w and distance_h will give us 180 cm distance in both horizontal and vertical directions
            # (how many pixels will be there in 180cm length in horizontal and vertical direction of birds eye view),
            # which we can use to calculate distance between two humans in transformed view or bird eye view
            distance_w = np.sqrt((warped_pt[0][0] - warped_pt[1][0]) ** 2 + (warped_pt[0][1] - warped_pt[1][1]) ** 2)
            distance_h = np.sqrt((warped_pt[0][0] - warped_pt[2][0]) ** 2 + (warped_pt[0][1] - warped_pt[2][1]) ** 2)
            pnts = np.array(points[:4], np.int32)
            cv2.polylines(frame, [pnts], True, (70, 70, 70), thickness=2)

            ####################################################################################

            # YOLO v3
            blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
            net.setInput(blob)
            layerOutputs = net.forward(ln1)
            boxes = []
            confidences = []
            classIDs = []

            for (box, pred) in zip(locs, preds):
                # unpack the bounding box and predictions
                (startX, startY, endX, endY) = box
                (mask, withoutMask) = pred

                # determine the class label and color we'll use to draw the bounding box and text
                if mask > withoutMask:
                    if mask > 0.90:
                        label = "Mask"
                        color = (0, 255, 0)  # green
                        img1 = cv2.imread("emoji_happy.png", -1)

                    else:
                        label = "Mask not on Properly"
                        color = (0, 220, 220)  # yellow
                        img1 = cv2.imread("emoji_middle.png", -1)
                else:
                    label = "No Mask"
                    color = (0, 0, 255)  # red
                    img1 = cv2.imread("emoji_worried.png", -1)

            img1 = cv2.resize(img1, (int(img1.shape[1] * 0.25), int(img1.shape[0] * 0.25)))

            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
            cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            x_offset = y_offset = 50
            y1, y2 = y_offset, y_offset + img1.shape[0]
            x1, x2 = x_offset, x_offset + img1.shape[1]

            alpha_s = img1[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s

            for c in range(0, 3):
                frame[y1:y2, x1:x2, c] = (alpha_s * img1[:, :, c] +
                                          alpha_l * frame[y1:y2, x1:x2, c])
            for output in layerOutputs:
                for detection in output:
                    scores = detection[5:]
                    classID = np.argmax(scores)
                    confidence = scores[classID]
                    # detecting humans in frame
                    if classID == 0:

                        if confidence > confid:
                            box = detection[0:4] * np.array([W, H, W, H])
                            (centerX, centerY, width, height) = box.astype("int")
                            cv2.putText(frame, label, (startX, startY - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                            x = int(centerX - (width / 2))
                            y = int(centerY - (height / 2))

                            boxes.append([x, y-10, int(width), int(height)+25])
                            confidences.append(float(confidence))
                            classIDs.append(classID)

            idxs = cv2.dnn.NMSBoxes(boxes, confidences, confid, thresh)
            boxes1 = []
            for i in range(len(boxes)):
                if i in idxs:
                    boxes1.append(boxes[i])
                    x, y, w, h = boxes[i]

            if len(boxes1) == 0:
                count = count + 1
                continue

            # Here we will be using bottom center point of bounding box for all boxes and will transform all those
            # bottom center points to bird eye view
            person_points = utills.get_transformed_points(boxes1, prespective_transform)

            # Here we will calculate distance between transformed points(humans)
            distances_mat, bxs_mat = utills.get_distances(boxes1, person_points, distance_w, distance_h)
            risk_count = utills.get_count(distances_mat)

            frame1 = np.copy(frame)

            # Draw bird eye view and frame with bouding boxes around humans according to risk factor
            img = plot.social_distancing_view(frame1, bxs_mat, boxes1, risk_count)
            # Show/write image and videos
            cv2.imshow('Camera View', img)


        count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print(time.time()-start1)
    vs.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    # Receives arguements specified by user
    parser = argparse.ArgumentParser()

    parser.add_argument('-v', '--video_path', action='store', dest='video_path', default='./data/example3.mp4',
                        help='Path for input video')

    parser.add_argument('-m', '--model', action='store', dest='model', default='./models/',
                        help='Path for models directory')

    parser.add_argument('-u', '--uop', action='store', dest='uop', default='NO',
                        help='Use open pose or not (YES/NO)')

    values = parser.parse_args()

    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--face", type=str, default="face_detector",
                    help="path to face detector model directory")
    ap.add_argument("-m", "--model", type=str, default="mask_detector.model",
                    help="path to trained face mask detector model")
    ap.add_argument("-c", "--confidence", type=float, default=0.5,
                    help="minimum probability to filter weak detections")
    args = vars(ap.parse_args())

    # load our serialized face detector model from disk
    print("[INFO] loading face detector model...")
    faceNet = insightface.model_zoo.get_model('retinaface_r50_v1')
    faceNet.prepare(ctx_id=-1, nms=0.4)

    # load the face mask detector model from disk
    print("[INFO] loading face mask detector model...")
    maskNet = load_model(args["model"])

    model_path = values.model
    if model_path[len(model_path) - 1] != '/':
        model_path = model_path + '/'



    # load Yolov3 weights

    weightsPath = model_path + "yolov3.weights"
    configPath = model_path + "yolov3.cfg"

    net_yl = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    ln = net_yl.getLayerNames()
    ln1 = [ln[i[0] - 1] for i in net_yl.getUnconnectedOutLayers()]

    # set mouse callback

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", get_mouse_points)
    np.random.seed(42)

    calculate_social_distancing(values.video_path, net_yl, ln1)


