import cv2
import torch
import numpy as np

from irisdetection import IrisDetector
from modernboundingbox import Display

# camera indices
DEFAULT_CAMERA = 0
SECONDARY_CAMERA = 1

# available models:
# MiDaS_small: Fastest, low resource, medium accuracy
# DPT_Large: Slowest, max resource, highest accuracy
# DPT_Hybrid: medium
MODEL_OPTIONS = ["MiDaS_small", "DPT_Large", "DPT_Hybrid"]

DEPTH_MODEL = MODEL_OPTIONS[0] # adjust index
DEPTH_MODEL_PATH = "intel-isl/MiDaS"

# load depth estimation model
depth_model = torch.hub.load(DEPTH_MODEL_PATH, DEPTH_MODEL)

# Move model to GPU if available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
depth_model.to(device)
depth_model.eval()

# Load transforms to resize and normalize the image
transforms = torch.hub.load(DEPTH_MODEL_PATH, "transforms")

if DEPTH_MODEL == "DPT_Large" or DEPTH_MODEL == "DPT_Hybrid":
    transform = transforms.dpt_transform
else:
    transform = transforms.small_transform

def DepthMap(input_image):
    # Apply input transforms
    input_batch = transform(input_image).to(device)

    # Prediction and resize to original resolution
    with torch.no_grad():
        prediction = depth_model(input_batch)
        prediction = torch.nn.functional.interpolate(prediction.unsqueeze(1), size=input_image.shape[:2], mode="bicubic", align_corners=False).squeeze()

    # get network output and normalize the output image
    depth_map = prediction.cpu().numpy()
    depth_map = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)

    input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)

    # process depth map for display + colormap
    depth_map = (depth_map * 255).astype(np.uint8)
    depth_map = cv2.applyColorMap(depth_map , cv2.COLORMAP_MAGMA)

    return depth_map

def Depth2Iris(iris_depth):
    return -1.7 * iris_depth + 2

def DetectFaces(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    faces = classifier.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    return faces

def StackDisplay(image1, image2):
    return np.concatenate((image1, image2), axis=1)

def main():
    video_cap = cv2.VideoCapture(DEFAULT_CAMERA)

    while True:
        _, input_frame = video_cap.read()

        iris_detector = IrisDetector(input_frame)

        detections = iris_detector.DetectIris()
        depth_map = DepthMap(input_frame)

        detected_faces = DetectFaces(input_frame)
        for x, y, w, h in detected_faces:
            Display.BoundingBoxwithInfo(detections, f"Distance (IRIS-CAMERA): 25 cm", 0.6, x, y, w, h, color=(0, 255, 0), thickness=2)

        cv2.putText(detections, "IRIS DISTANCE MEASUREMENT", (40, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.82, color=(0, 255, 0))
        cv2.putText(depth_map, "DEPTH MAP", (40, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.82, color=(0, 255, 0))

        display = StackDisplay(detections, depth_map)

        cv2.imshow("IrisDistance v1.0", display)

        if cv2.waitKey(20) & 0xff == ord("q"):
            break

    video_cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

