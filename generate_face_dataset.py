import argparse

from eyewitness.dataset_util import create_bbox_dataset_from_eyewitness
from eyewitness.dataset_util import BboxDataSet
from peewee import SqliteDatabase

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate face dataset')
    parser.add_argument('--db_path', type=str, default='183.sqlite',
                        help='the path used to store detection result records')
    parser.add_argument('--output_dataset_path', type=str, default='183_dataset')

    args = parser.parse_args()

    database = SqliteDatabase(args.db_path)
    valid_classes = set(['face'])
    dataset_name = 'faces'
    create_bbox_dataset_from_eyewitness(
        database, valid_classes, args.output_dataset_path, dataset_name)

    dataset_A = BboxDataSet(args.output_dataset_path, dataset_name)
    dataset_A.generate_train_test_list(train_ratio=0.5)
