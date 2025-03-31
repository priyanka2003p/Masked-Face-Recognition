import cv2
import numpy as np
import mediapipe as mp
import time
from face_alignment import FaceMaskDetection
from tools import model_restore_from_pb
import tensorflow

#----tensorflow version check
if tensorflow.__version__.startswith('1.'):
    import tensorflow as tf
else:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
print("Tensorflow version: ", tf.__version__)

img_format = {'png', 'jpg', 'bmp'}

# Initialize mediapipe for eye blink detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

def eye_blink_detection(landmarks):
    # Using the eye landmarks to detect blinking
    # Left eye landmarks: 33, 133, 160, 158, 133
    # Right eye landmarks: 362, 263, 386, 374, 263
    left_eye = [landmarks[33], landmarks[133], landmarks[160], landmarks[158], landmarks[133]]
    right_eye = [landmarks[362], landmarks[263], landmarks[386], landmarks[374], landmarks[263]]

    # Calculate the aspect ratio (EAR) for left and right eyes
    def eye_aspect_ratio(eye_points):
        A = np.linalg.norm(eye_points[1] - eye_points[5])
        B = np.linalg.norm(eye_points[2] - eye_points[4])
        C = np.linalg.norm(eye_points[0] - eye_points[3])
        return (A + B) / (2.0 * C)

    left_ear = eye_aspect_ratio(left_eye)
    right_ear = eye_aspect_ratio(right_eye)
    ear = (left_ear + right_ear) / 2.0
    return ear

def video_init(camera_source=0, resolution="480", to_write=False, save_dir=None):
    #----var
    writer = None
    resolution_dict = {"480": [480, 640], "720": [720, 1280], "1080": [1080, 1920]}

    #----camera source connection
    cap = cv2.VideoCapture(camera_source)

    #----resolution decision
    if resolution in resolution_dict.keys():
        width = resolution_dict[resolution][1]
        height = resolution_dict[resolution][0]
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    else:
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # default 480
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # default 640

    if to_write is True:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        save_path = 'demo.avi'
        if save_dir is not None:
            save_path = os.path.join(save_dir, save_path)
        writer = cv2.VideoWriter(save_path, fourcc, 30, (int(width), int(height)))

    return cap, height, width, writer


def stream(pb_path, node_dict, ref_dir, camera_source=0, resolution="480", to_write=False, save_dir=None):
    #----var
    frame_count = 0
    FPS = "loading"
    face_mask_model_path = r'face_mask_detection.pb'
    margin = 40
    id2class = {0: 'Mask', 1: 'NoMask'}
    batch_size = 32
    threshold = 0.8

    # Video streaming initialization
    cap, height, width, writer = video_init(camera_source=camera_source, resolution=resolution, to_write=to_write, save_dir=save_dir)

    # Face detection init
    fmd = FaceMaskDetection(face_mask_model_path, margin, GPU_ratio=None)

    # Face recognition init
    sess, tf_dict = model_restore_from_pb(pb_path, node_dict, GPU_ratio=None)
    tf_input = tf_dict['input']
    tf_phase_train = tf_dict['phase_train']
    tf_embeddings = tf_dict['embeddings']
    model_shape = tf_input.shape.as_list()
    feed_dict = {tf_phase_train: False}
    if 'keep_prob' in tf_dict.keys():
        tf_keep_prob = tf_dict['keep_prob']
        feed_dict[tf_keep_prob] = 1.0

    # Read images from the database
    paths = [file.path for file in os.scandir(ref_dir) if file.name[-3:] in img_format]
    len_ref_path = len(paths)

    # TensorFlow setting for calculating distance
    if len_ref_path > 0:
        with tf.Graph().as_default():
            tf_tar = tf.placeholder(dtype=tf.float32, shape=tf_embeddings.shape[-1])
            tf_ref = tf.placeholder(dtype=tf.float32, shape=tf_embeddings.shape)
            tf_dis = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(tf_ref, tf_tar)), axis=1))
            sess_cal = tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True))
            sess_cal.run(tf.global_variables_initializer())

        feed_dict_2 = {tf_ref: np.zeros([len_ref_path, tf_embeddings.shape[-1]], dtype=np.float32)}

    # Initialize mediapipe Face Mesh
    with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:

        while(cap.isOpened()):
            ret, img = cap.read()

            if ret is True:
                # Image processing
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_rgb = img_rgb.astype(np.float32)
                img_rgb /= 255

                # Face detection using face mask detection model
                img_fd = cv2.resize(img_rgb, fmd.img_size)
                img_fd = np.expand_dims(img_fd, axis=0)

                bboxes, re_confidence, re_classes, re_mask_id = fmd.inference(img_fd, height, width)
                if len(bboxes) > 0:
                    for num, bbox in enumerate(bboxes):
                        class_id = re_mask_id[num]
                        color = (0, 255, 0) if class_id == 0 else (0, 0, 255)
                        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), color, 2)

                        # Use mediapipe to detect face landmarks
                        results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

                        if results.multi_face_landmarks:
                            for face_landmarks in results.multi_face_landmarks:
                                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark])
                                ear = eye_blink_detection(landmarks)
                                if ear < 0.2:  # Threshold for blink detection
                                    cv2.putText(img, "Liveliness Detected", (bbox[0] + 2, bbox[1] - 2),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

                        # Face recognition
                        name = ""
                        img_fr = img_rgb[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2], :]
                        img_fr = cv2.resize(img_fr, (int(model_shape[2]), int(model_shape[1])))
                        img_fr = np.expand_dims(img_fr, axis=0)

                        feed_dict[tf_input] = img_fr
                        embeddings_tar = sess.run(tf_embeddings, feed_dict=feed_dict)
                        feed_dict_2[tf_tar] = embeddings_tar[0]
                        distance = sess_cal.run(tf_dis, feed_dict=feed_dict_2)
                        arg = np.argmin(distance)

                        if distance[arg] < threshold:
                            name = paths[arg].split("\\")[-1].split(".")[0]

                        cv2.putText(img, "{},{}".format(id2class[class_id], name), (bbox[0] + 2, bbox[1] - 2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                # Display the video stream
                cv2.imshow('Frame', img)

                if to_write is True:
                    writer.write(img)

                frame_count += 1
                FPS = time.time() - FPS
                if frame_count % 60 == 0:
                    FPS = 1.0 / FPS
                    frame_count = 0

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if to_write is True:
        writer.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    pb_path = 'face_mask_detection.pb'
    node_dict = {'input': 'input_node', 'phase_train': 'phase_train', 'embeddings': 'embeddings_node'}
    ref_dir = 'database'
    stream(pb_path, node_dict, ref_dir)
