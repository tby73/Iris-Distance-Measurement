import cv2
import mediapipe as mp
import numpy as np

class IrisDetector:
    def __init__(self, input_image, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.input_image = input_image

        # eye & iris mp ID's
        self.LEFT_IRIS_IDS = [474, 475, 476, 477]
        self.RIGHT_IRIS_IDS = [469, 470, 471, 472]

        # FaceMesh module
        self.mp_face = mp.solutions.face_mesh

        # FaceMesh params
        self.max_num_faces = max_num_faces
        self.refine_landmarks = refine_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

    def DetectIris(self):
        # convert to RGB first
        rgb = cv2.cvtColor(self.input_image, cv2.COLOR_BGR2RGB)

        with self.mp_face.FaceMesh(max_num_faces=self.max_num_faces, refine_landmarks=self.refine_landmarks, 
                                   min_detection_confidence=self.min_detection_confidence, min_tracking_confidence=self.min_tracking_confidence) as face_mesh:
            # get network output
            height, width = rgb.shape[:2]
            detections = face_mesh.process(rgb)

            if detections.multi_face_landmarks:
                # get mesh points
                mesh_points = np.array([np.multiply([p.x, p.y], [width, height]).astype(int) for p in detections.multi_face_landmarks[0].landmark])

                # draw polylines
                cv2.polylines(rgb, [mesh_points[self.LEFT_IRIS_IDS]], True, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.polylines(rgb, [mesh_points[self.RIGHT_IRIS_IDS]], True, (0, 255, 0), 1, cv2.LINE_AA)

                # find center iris coordinates
                (l_cx, l_cy), (l_radius) = cv2.minEnclosingCircle(mesh_points[self.LEFT_IRIS_IDS])
                (r_cx, r_cy), (r_radius) = cv2.minEnclosingCircle(mesh_points[self.RIGHT_IRIS_IDS])

                # construct iris circle
                cleft = np.array([l_cx, l_cy], dtype=np.int32)
                cright = np.array([r_cx, r_cy], dtype=np.int32)

                cv2.circle(rgb, cleft, int(l_radius), (255, 0, 255), 1, cv2.LINE_AA)
                cv2.circle(rgb, cright, int(r_radius), (255, 0, 255), 1, cv2.LINE_AA)

        # convert to BGR for display
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    
