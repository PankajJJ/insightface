import argparse
import os
import time
import tempfile

import arrow
import cv2
import PIL
from eyewitness.config import (IN_MEMORY, BBOX, RAW_IMAGE_PATH)
from eyewitness.image_id import ImageId

from eyewitness.image_utils import (ImageProducer, swap_channel_rgb_bgr, ImageHandler, Image)
from eyewitness.result_handler.db_writer import BboxPeeweeDbWriter
from eyewitness.result_handler.line_detection_result_handler import LineAnnotationSender
from eyewitness.detection_utils import DetectionResultHandler
from miio.powerstrip import PowerStrip
from peewee import SqliteDatabase
from gtts import gTTS

from mtcnn_arcface_classifier import MtcnnArcFaceClassifier

parser = argparse.ArgumentParser(description='face model example')
# model config
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='models/model-r100-ii/model,0',
                    help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--mtcnn_path', default='deploy/mtcnn-model/',
                    help='path to load model.')
# embedding setting
parser.add_argument('--dataset_embedding_path', default='face.pkl',
                    help='pre-generated arcface embedding')

# end2end setting
parser.add_argument('--db_path', type=str, default='::memory::',
                    help='the path used to store detection result records')
parser.add_argument('--interval_s', type=int, default=3, help='the interval of image generation')
parser.add_argument('--raw_image_folder', type=str, default=None,
                    help='store raw image to folder if given')

# chuangmi setting
parser.add_argument('--chuang_mi_power_ip', type=str, default='',
                    help='the chuang_mi power ip')
parser.add_argument('--chuang_mi_power_token', type=str, default='',
                    help='the chuang_mi power token')


# prounce the detected users
parser.add_argument('--is_prounce', default=True, type=bool, help='prounce the detected_users')


def prounce_zh_text(text):
    language = 'zh-tw'
    with tempfile.TemporaryDirectory() as local_folder:
        myobj = gTTS(text=text, lang=language, slow=False)
        destination_path = os.path.join(local_folder, "tmp.mp3")
        myobj.save(destination_path)
        # it's quite dirty, but work
        os.system("mpg321 %s" % destination_path)


class ChuangmiPowerPlugHandler(DetectionResultHandler):
    def __init__(self, ip, token, is_prounce=True):
        """
        Parameters
        ----------
        db_path: str
            database path
        """
        self.power = PowerStrip(ip=ip, token=token)
        self.power_cut_timestamp = arrow.now().timestamp
        self.postpone_seconds = 30
        self.is_prounce = is_prounce

    @property
    def detection_method(self):
        """str: BBOX"""
        return BBOX

    def _handle(self, detection_result):
        """
        handle detection result, the logic here:
        1. check if valid_users there(Appier empolyees, and face bigger than a ratio)
        2. if valid_users there, postpone the watermark for stopping the powerplug
        2.1 power off if detected result without user, and the time exceed the watermark
        2.2 power on if now is off and there are detected user

        Parameters
        ----------
        detection_result: DetectionResult
            detection result
        """
        valid_detected_users = [
            i for i in detection_result.detected_objects if i.label != 'unknown']
        if valid_detected_users:
            print("valid_users: %s, lets postone the power_cut_timestamp for 30s"
                  % [user.label for user in valid_detected_users])
            self.power_cut_timestamp = arrow.now().timestamp + 30
            if not self.power.status().is_on:
                self.power.on()
                if self.is_prounce:
                    for user in valid_detected_users:
                        prounce_zh_text(user.label)
        else:
            if self.power.status().is_on:
                if arrow.now().timestamp > self.power_cut_timestamp:
                    print("timestamp exceeded, cutoff the power")
                    self.power.off()


class InMemoryImageProducer(ImageProducer):
    def __init__(self, video_path, interval_s):
        self.vid = cv2.VideoCapture(video_path)
        self.interval_s = interval_s
        if not self.vid.isOpened():
            raise IOError("Couldn't open webcam or video")

    def produce_method(self):
        return IN_MEMORY

    def produce_image(self):
        while True:
            # clean buffer hack: for Linux V4L capture backend with a internal fifo
            for iter_ in range(5):
                self.vid.grab()
            _, frame = self.vid.read()
            yield PIL.Image.fromarray(swap_channel_rgb_bgr(frame))
            time.sleep(self.interval_s)


def image_url_handler(drawn_image_path):
    """if site_domain not set in env, will pass a pickchu image"""
    site_domain = os.environ.get('site_domain')
    if site_domain is None:
        return 'https://upload.wikimedia.org/wikipedia/en/a/a6/Pok%C3%A9mon_Pikachu_art.png'
    else:
        return '%s/%s' % (site_domain, drawn_image_path)


def line_detection_result_filter(detection_result):
    """
    used to check if sent notification or not
    """
    return any(i.label == 'person' for i in detection_result.detected_objects)


if __name__ == '__main__':
    args = parser.parse_args()
    raw_image_folder = args.raw_image_folder
    # image producer from webcam
    image_producer = InMemoryImageProducer(0, interval_s=args.interval_s)

    # object detector
    object_detector = MtcnnArcFaceClassifier(args)

    # detection result handlers
    result_handlers = []

    # update image_info drawn_image_path, insert detection result
    database = SqliteDatabase(args.db_path)
    bbox_sqlite_handler = BboxPeeweeDbWriter(database)
    result_handlers.append(bbox_sqlite_handler)

    if args.chuang_mi_power_ip:
        result_handlers.append(
            ChuangmiPowerPlugHandler(
                args.chuang_mi_power_ip, args.chuang_mi_power_token, args.is_prounce))

    # setup your line channel token and audience
    channel_access_token = os.environ.get('LINE_CHANNEL_ACCESS_TOKEN')
    if channel_access_token:
        line_annotation_sender = LineAnnotationSender(
            channel_access_token=channel_access_token,
            image_url_handler=image_url_handler,
            detection_result_filter=line_detection_result_filter,
            detection_method=BBOX,
            update_audience_period=10,
            database=database)
        result_handlers.append(line_annotation_sender)

    for image in image_producer.produce_image():
        image_id = ImageId(channel='demo', timestamp=arrow.now().timestamp, file_format='jpg')

        # store the raw image or not
        if raw_image_folder:
            raw_image_path = "%s/%s_%s.%s" % (
                raw_image_folder, image_id.channel, image_id.timestamp, image_id.file_format)
            ImageHandler.save(image, raw_image_path)
        else:
            raw_image_path = None

        image_obj = Image(image_id, pil_image_obj=image)
        bbox_sqlite_handler.register_image(image_id, {RAW_IMAGE_PATH: raw_image_path})
        detection_result = object_detector.detect(image_obj)

        if len(detection_result.detected_objects) > 0:
            # draw and save image, update detection result
            drawn_image_path = "detected_image/%s_%s.%s" % (
                image_id.channel, image_id.timestamp, image_id.file_format)
            ImageHandler.draw_bbox(image, detection_result.detected_objects)
            ImageHandler.save(image, drawn_image_path)
            detection_result.image_dict['drawn_image_path'] = drawn_image_path

        for result_handler in result_handlers:
            result_handler.handle(detection_result)
