import argparse
import pickle

import arrow
import numpy as np
from eyewitness.config import BoundedBoxObject
from eyewitness.image_utils import resize_and_stack_image_objs
from eyewitness.image_id import ImageId
from eyewitness.image_utils import ImageHandler, Image
from eyewitness.object_classifier import ObjectClassifier
from eyewitness.detection_utils import DetectionResult
from bistiming import SimpleTimer

from deploy import face_model


class ArcFaceClassifier(ObjectClassifier):
    def __init__(self, args, registered_ids,
                 objects_frame=None, registered_images_embedding=None,
                 threshold=0.0):
        assert objects_frame is not None or registered_images_embedding is not None
        self.model = face_model.FaceModel(args)
        self.image_size = [int(i) for i in args.image_size.split(',')]
        if registered_images_embedding:
            self.registered_images_embedding = registered_images_embedding
        else:
            self.registered_images_embedding = self.model.get_faces_feature(objects_frame)

        self.registered_ids = registered_ids
        self.threshold = threshold
        self.unknown = 'unknown'

    def detect(self, image_obj, bbox_objs=None):
        if bbox_objs:
            objs = image_obj.fetch_bbox_pil_objs(bbox_objs)
            objects_frame = resize_and_stack_image_objs(self.image_size, objs)
        else:
            x2, y2 = image_obj.pil_image_obj.size
            bbox_objs = [BoundedBoxObject(0, 0, x2, y2, '', 0, '')]
            # resize and extend the size
            objects_frame = np.array(image_obj.pil_image_obj.resize(self.image_size))
            objects_frame = np.expand_dims(objects_frame, axis=0)

        objects_frame = np.transpose(objects_frame, (0, 3, 1, 2))
        objects_embedding = self.model.get_faces_feature(objects_frame)
        similar_matrix = objects_embedding.dot(self.registered_images_embedding.T)
        detected_idx = similar_matrix.argmax(1)
        result_objects = []
        for idx, bbox in enumerate(bbox_objs):
            x1, y1, x2, y2, _, _, _ = bbox
            label = self.registered_ids[detected_idx[idx]]
            score = similar_matrix[idx, detected_idx[idx]]
            if score < self.threshold:
                label = self.unknown
            result_objects.append(BoundedBoxObject(x1, y1, x2, y2, label, score, ''))

        image_dict = {
            'image_id': image_obj.image_id,
            'detected_objects': result_objects,
        }
        detection_result = DetectionResult(image_dict)
        return detection_result

    def valid_labels(self):
        return set(self.registered_ids + [self.unknown])

    def store_embedding_info(self, pkl_path):
        with open(pkl_path, 'wb') as f:
            pickle.dump((self.registered_images_embedding, self.registered_ids), file=f)

    @staticmethod
    def restore_embedding_info(pkl_path):
        with open(pkl_path, 'rb') as f:
            return pickle.load(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='face model test')
    parser.add_argument('--image-size', default='112,112', help='')
    parser.add_argument('--model', default='', help='path to load model.')
    parser.add_argument('--gpu', default=0, type=int, help='gpu id')

    args = parser.parse_args()

    raw_image_path = 'demo/183club/test_image.jpg'
    train_image_id = ImageId(channel='demo', timestamp=arrow.now().timestamp, file_format='jpg')
    train_image_obj = Image(train_image_id, raw_image_path=raw_image_path)

    register_image_bbox_objs = [
        BoundedBoxObject(x1=347, y1=138, x2=396, y2=202, label='5', score=1, meta=''),
        BoundedBoxObject(x1=230, y1=157, x2=279, y2=224, label='4', score=1, meta=''),
        BoundedBoxObject(x1=705, y1=138, x2=753, y2=209, label='3', score=1, meta=''),
        BoundedBoxObject(x1=540, y1=152, x2=587, y2=210, label='2', score=1, meta=''),
        BoundedBoxObject(x1=73.0, y1=162, x2=124, y2=232, label='1', score=1, meta='')
    ]

    objs = train_image_obj.fetch_bbox_pil_objs(register_image_bbox_objs)
    objects_frame = resize_and_stack_image_objs((112, 112), objs)
    objects_frame = np.transpose(objects_frame, (0, 3, 1, 2))
    registered_ids = [i.label for i in register_image_bbox_objs]

    arcface_classifier = ArcFaceClassifier(args, registered_ids, objects_frame=objects_frame)

    test_image_bbox_objs = [
        BoundedBoxObject(x1=367, y1=101, x2=412, y2=160, label='face', score=1, meta='1'),
        BoundedBoxObject(x1=311, y1=176, x2=364, y2=248, label='face', score=1, meta='4'),
        BoundedBoxObject(x1=114, y1=80, x2=165, y2=148, label='face', score=1, meta='3'),
        BoundedBoxObject(x1=219, y1=40, x2=272, y2=107, label='face', score=1, meta='5'),
        BoundedBoxObject(x1=159, y1=191, x2=212, y2=259, label='face', score=1, meta='2')
    ]
    raw_image_path = 'demo/183club/test_image2.jpg'
    test_image_id = ImageId(channel='demo', timestamp=arrow.now().timestamp, file_format='jpg')
    test_image_obj = Image(test_image_id, raw_image_path=raw_image_path)

    with SimpleTimer("Predicting image with classifier"):
        detection_result = arcface_classifier.detect(test_image_obj, test_image_bbox_objs)

    print("detected %s objects" % len(detection_result.detected_objects))
    correct_count = sum(
        1 for i, j in zip(test_image_bbox_objs, detection_result.detected_objects)
        if i.meta == j.label
    )
    n_test_objs = len(test_image_bbox_objs)
    print("accuracy = %s (%s / %s)" % (correct_count/n_test_objs, correct_count, n_test_objs))

    ImageHandler.draw_bbox(train_image_obj.pil_image_obj, register_image_bbox_objs)
    ImageHandler.save(train_image_obj.pil_image_obj, "detected_image/drawn_image_1.jpg")

    ImageHandler.draw_bbox(test_image_obj.pil_image_obj, detection_result.detected_objects)
    ImageHandler.save(test_image_obj.pil_image_obj, "detected_image/drawn_image_2.jpg")

    arcface_classifier.store_embedding_info('183_model.pkl')
    embedding, ids = ArcFaceClassifier.restore_embedding_info('183_model.pkl')
    print(embedding.shape)
