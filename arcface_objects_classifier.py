import argparse

import arrow
import numpy as np
from eyewitness.config import BoundedBoxObject
from eyewitness.image_utils import resize_and_stack_image_objs
from eyewitness.image_id import ImageId
from eyewitness.image_utils import Image
from eyewitness.object_classifier import ObjectClassifier

from deploy import face_model


class ArcFaceClassifier(ObjectClassifier):
    def __init__(self, args):
        self.model = face_model.FaceModel(args)
        self.image_size = [int(i) for i in args.image_size.split(',')]
        # TODO: forward images and register images

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

        # TODO: recognize faces for each bbox
        # image_dict = {
        #     'image_id': image_obj.image_id,
        #     'detected_objects': detected_objects,
        # }
        # detection_result = DetectionResult(image_dict)

    def valid_labels(self):
        # TODO: return valid items
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='face model test')
    parser.add_argument('--image-size', default='112,112', help='')
    parser.add_argument('--model', default='', help='path to load model.')
    parser.add_argument('--gpu', default=0, type=int, help='gpu id')

    args = parser.parse_args()

    arcface_classifier = ArcFaceClassifier(args)

    bbox_objs = [
        BoundedBoxObject(x1=347, y1=138, x2=396, y2=202, label='face', score=1, meta=''),
        BoundedBoxObject(x1=230, y1=157, x2=279, y2=224, label='face', score=1, meta=''),
        BoundedBoxObject(x1=705, y1=138, x2=753, y2=209, label='face', score=1, meta=''),
        BoundedBoxObject(x1=540, y1=152, x2=587, y2=210, label='face', score=1, meta=''),
        BoundedBoxObject(x1=73.0, y1=162, x2=124, y2=232, label='face', score=1, meta='')
    ]

    raw_image_path = 'demo/test_image.jpg'
    image_id = ImageId(channel='demo', timestamp=arrow.now().timestamp, file_format='jpg')
    image_obj = Image(image_id, raw_image_path=raw_image_path)

    # img = cv2.imread('deploy/Tom_Hanks_54745.png')
    # model = face_model.FaceModel(args)
    # img = model.get_input(img)
    # f1 = model.get_feature(img)
    # print(f1[0:10])
