import cv2
import os
import time
import math
import numpy as np
from face_alignment import FaceMaskDetection
from tools import model_restore_from_pb
from sklearn.decomposition import PCA
import tensorflow as tf

# Ensure compatibility with TensorFlow 1.x
if tf.__version__.startswith('2.'):
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()

print("TensorFlow version:", tf.__version__)

img_format = {'png', 'jpg', 'bmp'}

class FaceEmbeddingPCA:
    def __init__(self, n_components=50):
        self.pca = PCA(n_components=n_components)
        self.is_fitted = False

    def fit(self, embeddings):
        """Fit PCA on the embeddings"""
        self.pca.fit(embeddings)
        self.is_fitted = True
        explained_variance = np.sum(self.pca.explained_variance_ratio_) * 100
        print(f"Explained variance with {self.pca.n_components_} components: {explained_variance:.2f}%")

    def transform(self, embeddings):
        """Transform embeddings using PCA"""
        if not self.is_fitted:
            raise ValueError("PCA must be fitted before transform.")
        return self.pca.transform(embeddings)

    def inverse_transform(self, reduced_embeddings):
        """Transform back to original space"""
        if not self.is_fitted:
            raise ValueError("PCA must be fitted before inverse_transform.")
        return self.pca.inverse_transform(reduced_embeddings)


def video_init(camera_source=0, resolution="480", to_write=False, save_dir=None):
    resolutions = {
        "480": (640, 480),
        "720": (1280, 720),
        "1080": (1920, 1080),
    }
    width, height = resolutions.get(resolution, (640, 480))
    
    cap = cv2.VideoCapture(camera_source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    writer = None
    if to_write and save_dir:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        save_path = os.path.join(save_dir, f"output_{time.time()}.avi")
        writer = cv2.VideoWriter(save_path, fourcc, 20.0, (width, height))

    return cap, height, width, writer


def stream(pb_path, node_dict, ref_dir, camera_source=0, resolution="480", to_write=False, save_dir=None):
    frame_count = 0
    FPS = "loading"
    face_mask_model_path = r'face_mask_detection.pb'
    margin = 40
    id2class = {0: 'Mask', 1: 'NoMask'}
    threshold = 0.8

    face_pca = FaceEmbeddingPCA(n_components=50)

    cap, height, width, writer = video_init(camera_source, resolution, to_write, save_dir)
    fmd = FaceMaskDetection(face_mask_model_path, margin, GPU_ratio=None)
    sess, tf_dict = model_restore_from_pb(pb_path, node_dict, GPU_ratio=None)

    tf_input = tf_dict['input']
    tf_phase_train = tf_dict['phase_train']
    tf_embeddings = tf_dict['embeddings']
    model_shape = tf_input.shape.as_list()
    feed_dict = {tf_phase_train: False}

    if 'keep_prob' in tf_dict:
        tf_keep_prob = tf_dict['keep_prob']
        feed_dict[tf_keep_prob] = 1.0

    paths = [file.path for file in os.scandir(ref_dir) if file.name.split('.')[-1] in img_format]
    if not paths:
        print("No reference images found in", ref_dir)
        return

    embeddings_ref = []
    for path in paths:
        img = cv2.imread(path)
        if img is not None:
            img = cv2.resize(img, (model_shape[2], model_shape[1]))
            img = img[..., ::-1] / 255.0
            feed_dict[tf_input] = np.expand_dims(img, axis=0)
            embedding = sess.run(tf_embeddings, feed_dict=feed_dict)
            embeddings_ref.append(embedding[0])
        else:
            print(f"Failed to read image: {path}")

    embeddings_ref = np.array(embeddings_ref)
    face_pca.fit(embeddings_ref)
    embeddings_ref_pca = face_pca.transform(embeddings_ref)

    with tf.Graph().as_default():
        tf_tar = tf.placeholder(dtype=tf.float32, shape=(face_pca.pca.n_components_))
        tf_ref = tf.placeholder(dtype=tf.float32, shape=(len(paths), face_pca.pca.n_components_))
        tf_dis = tf.sqrt(tf.reduce_sum(tf.square(tf_ref - tf_tar), axis=1))

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess_cal = tf.Session(config=config)
        sess_cal.run(tf.global_variables_initializer())

    feed_dict_2 = {tf_ref: embeddings_ref_pca}

    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
        img_fd = cv2.resize(img_rgb, fmd.img_size)
        bboxes, _, _, re_mask_id = fmd.inference(np.expand_dims(img_fd, axis=0), height, width)

        for bbox, mask_id in zip(bboxes, re_mask_id):
            color = (0, 255, 0) if mask_id == 0 else (0, 0, 255)
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), color, 2)

            if embeddings_ref_pca is not None:
                face_img = img_rgb[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
                face_img = cv2.resize(face_img, (model_shape[2], model_shape[1]))
                feed_dict[tf_input] = np.expand_dims(face_img, axis=0)
                embedding_tar = sess.run(tf_embeddings, feed_dict=feed_dict)
                embedding_tar_pca = face_pca.transform(embedding_tar)
                feed_dict_2[tf_tar] = embedding_tar_pca[0]
                distances = sess_cal.run(tf_dis, feed_dict=feed_dict_2)
                name = paths[np.argmin(distances)].split(os.sep)[-1].split('.')[0] if min(distances) < threshold else "Unknown"
                cv2.putText(img, f"{id2class[mask_id]}, {name}", (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color)

        cv2.imshow("Video Stream", img)
        if writer:
            writer.write(img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    if writer:
        writer.release()


if __name__ == "__main__":
    pb_path = "weights_15.pb"
    node_dict = {
        'input': 'input:0',
        'keep_prob': 'keep_prob:0',
        'phase_train': 'phase_train:0',
        'embeddings': 'embeddings:0',
    }
    ref_dir = "database"
    stream(pb_path, node_dict, ref_dir, camera_source=0, resolution="720", to_write=False, save_dir=None)
