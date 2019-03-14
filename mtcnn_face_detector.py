import arrow
import argparse

import numpy as np
import mxnet as mx
from eyewitness.config import BoundedBoxObject
from eyewitness.detection_utils import DetectionResult
from eyewitness.object_detector import ObjectDetector
from eyewitness.image_utils import swap_channel_rgb_bgr, Image
from eyewitness.image_id import ImageId
from bistiming import SimpleTimer

from deploy.mtcnn_detector import MtcnnDetector


class MtcnnFaceDetector(ObjectDetector):
    def __init__(self, mtcnn_path, ctx):
        self.face_detector = MtcnnDetector(
            model_folder=mtcnn_path,
            ctx=ctx, num_worker=1, accurate_landmark=True,
            threshold=[0.0, 0.0, 0.2])

    def detect(self, image_obj):
        detected_objects = []
        frame = swap_channel_rgb_bgr(np.array(image_obj.pil_image_obj))
        ret = self.face_detector.detect_face(frame, det_type=0)
        bbox, points = ret

        # boundingboxes, points = detect_face(
        #     frame, self.minsize, self.PNet, self.RNet, self.ONet,
        #     self.threshold, False, self.factor)

        # # boundingboxes shape n, 5
        # for idx in range(boundingboxes.shape[0]):
        #     x1, y1, x2, y2, score = boundingboxes[idx]
        #     detected_objects.append(BoundedBoxObject(x1, y1, x2, y2, 'face', score, ''))

        image_dict = {
            'image_id': image_obj.image_id,
            'detected_objects': detected_objects,
        }
        detection_result = DetectionResult(image_dict)
        return detection_result

    @property
    def valid_labels(self):
        return set(['face'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='face model test')
    parser.add_argument('--mtcnn_path', default='./deploy/mtcnn-model/',
                        help='path to load model.')
    parser.add_argument('--gpu', default=0, type=int, help='gpu id')

    args = parser.parse_args()

    ctx = mx.gpu(args.gpu)
    model_name = 'MTCNN'
    with SimpleTimer("Loading model %s" % model_name):
        face_detector = MtcnnFaceDetector(args.mtcnn_path, ctx)

    raw_image_path = 'demo/183club/test_image.jpg'
    train_image_id = ImageId(channel='demo', timestamp=arrow.now().timestamp, file_format='jpg')
    train_image_obj = Image(train_image_id, raw_image_path=raw_image_path)

    face_detector.detect(train_image_obj)
