import argparse
import glob
import os
from pathlib import Path

import mxnet as mx
from bistiming import SimpleTimer
from eyewitness.image_utils import Image, ImageHandler
from eyewitness.image_id import ImageId
from eyewitness.result_handler.db_writer import (BboxPeeweeDbWriter, FalseAlertPeeweeDbWriter)
from eyewitness.config import RAW_IMAGE_PATH

from peewee import SqliteDatabase

from mtcnn_face_detector import MtcnnFaceDetector

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='face model test')
    parser.add_argument('--mtcnn_path', default='deploy/mtcnn-model/',
                        help='path to load model.')
    parser.add_argument('--gpu', default=0, type=int, help='gpu id')

    parser.add_argument('--db_path', type=str, default='::memory::',
                        help='the path used to store detection result records')

    parser.add_argument('--input_image_path', type=str, default='demo/183club',
                        help='the path used to store detection result records')

    args = parser.parse_args()

    if args.gpu >= 0:
        ctx = mx.gpu(args.gpu)
    else:
        ctx = mx.cpu(0)
    model_name = 'MTCNN'
    with SimpleTimer("Loading model %s" % model_name):
        face_detector = MtcnnFaceDetector(args.mtcnn_path, ctx)

    # detection result handlers
    result_handlers = []
    # update image_info drawn_image_path, insert detection result
    database = SqliteDatabase(args.db_path)
    bbox_sqlite_handler = BboxPeeweeDbWriter(database)
    result_handlers.append(bbox_sqlite_handler)

    # just call for create false_alert_table
    FalseAlertPeeweeDbWriter(database)

    raw_images = glob.glob(os.path.join(args.input_image_path, '*.jpg'))
    for raw_image_path in raw_images:
        filename = raw_image_path.split('/')[-1]
        train_image_id = ImageId.from_str(filename)
        train_image_obj = Image(train_image_id, raw_image_path=raw_image_path)

        # make sure the image_info were registered
        bbox_sqlite_handler.register_image(train_image_id, {RAW_IMAGE_PATH: raw_image_path})
        with SimpleTimer("detect a img %s" % filename):
            detection_results = face_detector.detect(train_image_obj)
            n_face = len(detection_results.detected_objects)
            print("detected %s objs" % n_face)
            ImageHandler.draw_bbox(
                train_image_obj.pil_image_obj, detection_results.detected_objects)

            output_file = "detected_image/%s/%s" % (n_face, filename)
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)

            ImageHandler.save(
                train_image_obj.pil_image_obj, output_file)

            for result_handler in result_handlers:
                result_handler.handle(detection_results)
