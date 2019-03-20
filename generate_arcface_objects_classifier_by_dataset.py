import argparse
import arrow

import numpy as np
import mxnet as mx
from eyewitness.image_utils import ImageHandler, Image, resize_and_stack_image_objs
from eyewitness.dataset_util import BboxDataSet
from eyewitness.image_id import ImageId
from bistiming import SimpleTimer

from arcface_objects_classifier import ArcFaceClassifier
from mtcnn_face_detector import MtcnnFaceDetector


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='face model test')
    parser.add_argument('--image-size', default='112,112', help='')
    parser.add_argument('--model', default='models/model-r100-ii/model,0',
                        help='path to load model.')
    parser.add_argument('--gpu', default=0, type=int, help='gpu id')
    parser.add_argument('--mtcnn_path', default='deploy/mtcnn-model/',
                        help='path to load model.')
    parser.add_argument('--dataset_folder', default='models/183',
                        help='dataset used to registered')
    parser.add_argument('--preload_model', default=None,
                        help='pre-generated arcgace embedding')
    parser.add_argument('--demo_image', default='demo/183club/test_image2.jpg',
                        help='dataset used to registered')

    args = parser.parse_args()

    if args.gpu > 0:
        ctx = mx.gpu(args.gpu)
    else:
        ctx = mx.cpu(0)
    model_name = 'MTCNN'
    with SimpleTimer("Loading model %s" % model_name):
        face_detector = MtcnnFaceDetector(args.mtcnn_path, ctx)

    dataset_name = 'faces'
    faces_dataset = BboxDataSet(args.dataset_folder, dataset_name)

    if args.preload_model is None:
        objs = []
        registered_ids = []
        image_id2_objs = dict(faces_dataset.ground_truth_iterator(testing_set_only=False))
        for image_obj in faces_dataset.image_obj_iterator(testing_set_only=False):
            image_bbox_objs = image_id2_objs.get(image_obj.image_id, [])
            if len(image_bbox_objs) != 1:
                continue
            objs += image_obj.fetch_bbox_pil_objs(image_bbox_objs)
            registered_ids += [image_obj.image_id.channel]

        objects_frame = resize_and_stack_image_objs((112, 112), objs)
        print(objects_frame.shape)
        objects_frame = np.transpose(objects_frame, (0, 3, 1, 2))

        arcface_classifier = ArcFaceClassifier(args, registered_ids, objects_frame=objects_frame)
        arcface_classifier.store_embedding_info('faces.pkl')
        embedding, _ = ArcFaceClassifier.restore_embedding_info('faces.pkl')
        print("restored embedding shape", embedding.shape)
    else:
        embedding, registered_ids = ArcFaceClassifier.restore_embedding_info(args.preload_model)
        arcface_classifier = ArcFaceClassifier(args, registered_ids,
                                               registered_images_embedding=embedding)

    raw_image_path = args.demo_image
    if raw_image_path:
        test_image_id = ImageId(channel='demo', timestamp=arrow.now().timestamp, file_format='jpg')
        test_image_obj = Image(test_image_id, raw_image_path=raw_image_path)

        face_detection_result = face_detector.detect(test_image_obj)
        with SimpleTimer("Predicting image with classifier"):
            detection_result = arcface_classifier.detect(
                test_image_obj, face_detection_result.detected_objects)

        ImageHandler.draw_bbox(test_image_obj.pil_image_obj, detection_result.detected_objects)
        ImageHandler.save(test_image_obj.pil_image_obj, "detected_image/demo.jpg")
