import argparse
import pickle
import logging
from pathlib import Path

import arrow
import numpy as np
from eyewitness.config import BoundedBoxObject
from eyewitness.image_utils import resize_and_stack_image_objs
from eyewitness.image_id import ImageId
from eyewitness.image_utils import ImageHandler, Image
from eyewitness.object_classifier import ObjectClassifier
from eyewitness.detection_utils import DetectionResult
from eyewitness.config import DATASET_ALL
from bistiming import SimpleTimer
from dchunk import chunk_with_index

from deploy import face_model

LOG = logging.getLogger(__name__)


class ArcFaceClassifier(ObjectClassifier):
    def __init__(self, args, registered_ids,
                 objects_frame=None, registered_images_embedding=None,
                 threshold=0.0, batch_size=20):
        assert objects_frame is not None or registered_images_embedding is not None
        self.model = face_model.FaceModel(args)
        self.image_size = [int(i) for i in args.image_size.split(',')]
        if registered_images_embedding is not None:
            self.registered_images_embedding = registered_images_embedding
        else:
            n_images = objects_frame.shape[0]
            registered_images_embedding_list = []
            for row_idx, batch_start, batch_end in chunk_with_index(range(n_images), batch_size):
                objects_embedding = self.model.get_faces_feature(
                    objects_frame[batch_start: batch_end])
                registered_images_embedding_list.append(objects_embedding)
            self.registered_images_embedding = np.concatenate(registered_images_embedding_list)

        LOG.info("registered_images_embedding shape: %s", self.registered_images_embedding.shape)
        self.registered_ids = registered_ids
        self.threshold = threshold
        self.unknown = 'unknown'

    def detect(self, image_obj, bbox_objs=None, batch_size=2):
        if bbox_objs is None:
            x2, y2 = image_obj.pil_image_obj.size
            bbox_objs = [BoundedBoxObject(0, 0, x2, y2, '', 0, '')]

        n_bbox = len(bbox_objs)
        result_objects = []
        for row_idx, batch_start, batch_end in chunk_with_index(range(n_bbox), batch_size):
            batch_bbox_objs = bbox_objs[batch_start:batch_end]
            objs = image_obj.fetch_bbox_pil_objs(batch_bbox_objs)
            objects_frame = resize_and_stack_image_objs(self.image_size, objs)
            objects_frame = np.transpose(objects_frame, (0, 3, 1, 2))
            objects_embedding = self.model.get_faces_feature(objects_frame)
            similar_matrix = objects_embedding.dot(self.registered_images_embedding.T)
            detected_idx = similar_matrix.argmax(1)

            for idx, bbox in enumerate(batch_bbox_objs):
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
        pkl_path_obj = Path(pkl_path)
        if not pkl_path_obj.exists():
            raise Exception('path %s not exist' % pkl_path_obj)

        if pkl_path_obj.is_dir():
            pkls = pkl_path_obj.glob('*.pkl')
            all_face_embedding_list = []
            all_face_ids = []
            for pkl in pkls:
                with open(str(pkl), 'rb') as f:
                    face_embedding, face_ids = pickle.load(f)
                    if face_embedding.shape[0] != len(face_ids):
                        LOG.warn('the pkl %s, without same shape embedding and face ids', pkl)
                        continue
                    all_face_ids.extend(face_ids)
                    all_face_embedding_list.append(face_embedding)

            all_face_embedding = np.concatenate(all_face_embedding_list)
            assert all_face_embedding.shape[0] == len(all_face_ids)

            return all_face_embedding, all_face_ids
        else:
            with open(str(pkl_path), 'rb') as f:
                return pickle.load(f)


def generate_dataset_arcface_embedding(args, dataset, output_path):
    objs = []
    registered_ids = []
    # image_id2_objs = dict(dataset.ground_truth_iterator(testing_set_only=False))
    for image_obj, gt_objs in dataset.dataset_iterator(mode=DATASET_ALL, with_gt_objs=True):
        objs += image_obj.fetch_bbox_pil_objs(gt_objs)
        registered_ids += [bbox.label for bbox in gt_objs]

    objects_frame = resize_and_stack_image_objs((112, 112), objs)
    print("object_frame shape:", objects_frame.shape)
    objects_frame = np.transpose(objects_frame, (0, 3, 1, 2))
    with SimpleTimer("extracting embedding for dataset %s" % (dataset.dataset_name)):
        arcface_classifier = ArcFaceClassifier(args, registered_ids, objects_frame=objects_frame)

    print("store embedding to %s" % (output_path))
    arcface_classifier.store_embedding_info(output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='face model test')
    parser.add_argument('--image-size', default='112,112', help='')
    parser.add_argument('--model', default='models/model-r100-ii/model,0',
                        help='path to load model.')
    parser.add_argument('--gpu', default=0, type=int, help='gpu id')

    args = parser.parse_args()

    raw_image_path = 'demo/183club/test_image.jpg'
    train_image_id = ImageId(channel='demo', timestamp=arrow.now().timestamp, file_format='jpg')
    train_image_obj = Image(train_image_id, raw_image_path=raw_image_path)

    register_image_bbox_objs = [
        BoundedBoxObject(x1=73, y1=162, x2=124, y2=232, label='1', score=1, meta=''),
        BoundedBoxObject(x1=230, y1=157, x2=279, y2=224, label='2', score=1, meta=''),
        BoundedBoxObject(x1=347, y1=138, x2=396, y2=202, label='3', score=1, meta=''),
        BoundedBoxObject(x1=540, y1=152, x2=587, y2=210, label='4', score=1, meta=''),
        BoundedBoxObject(x1=705, y1=138, x2=753, y2=209, label='5', score=1, meta=''),
    ]

    objs = train_image_obj.fetch_bbox_pil_objs(register_image_bbox_objs)
    objects_frame = resize_and_stack_image_objs((112, 112), objs)
    objects_frame = np.transpose(objects_frame, (0, 3, 1, 2))
    registered_ids = [i.label for i in register_image_bbox_objs]

    arcface_classifier = ArcFaceClassifier(args, registered_ids, objects_frame=objects_frame)

    test_image_bbox_objs = [
        BoundedBoxObject(x1=114, y1=80, x2=165, y2=148, label='face', score=1, meta='5'),
        BoundedBoxObject(x1=159, y1=191, x2=212, y2=259, label='face', score=1, meta='4'),
        BoundedBoxObject(x1=219, y1=40, x2=272, y2=107, label='face', score=1, meta='3'),
        BoundedBoxObject(x1=311, y1=176, x2=364, y2=248, label='face', score=1, meta='2'),
        BoundedBoxObject(x1=367, y1=101, x2=412, y2=160, label='face', score=1, meta='1'),
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
    ImageHandler.save(train_image_obj.pil_image_obj, "detected_image/183club/drawn_image_1.jpg")

    ImageHandler.draw_bbox(test_image_obj.pil_image_obj, detection_result.detected_objects)
    ImageHandler.save(test_image_obj.pil_image_obj, "detected_image/183club/drawn_image_2.jpg")

    arcface_classifier.store_embedding_info('183_model.pkl')
    embedding, _ = ArcFaceClassifier.restore_embedding_info('183_model.pkl')
    print("restored embedding shape", embedding.shape)
