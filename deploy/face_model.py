from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os

import numpy as np
import mxnet as mx
from sklearn import preprocessing


sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'common'))


def do_flip(data):
    for idx in xrange(data.shape[0]):
        data[idx, :, :] = np.fliplr(data[idx, :, :])


def get_model(ctx, image_size, model_str, layer):
    _vec = model_str.split(',')
    assert len(_vec) == 2
    prefix = _vec[0]
    epoch = int(_vec[1])
    print('loading', prefix, epoch)
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
    all_layers = sym.get_internals()
    sym = all_layers[layer+'_output']
    model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
    model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
    model.set_params(arg_params, aux_params)
    return model


class FaceModel:
    def __init__(self, args):
        self.args = args
        if args.gpu >= 0:
            ctx = mx.gpu(args.gpu)
        else:
            ctx = mx.cpu(0)
        _vec = args.image_size.split(',')
        assert len(_vec) == 2
        image_size = (int(_vec[0]), int(_vec[1]))
        self.model = get_model(ctx, image_size, args.model, 'fc1')

    def get_faces_feature(self, faces_frame):
        """
        Parameters:
        -----------
        faces_frame: np.array
            faces frame with shape (n, c, w, h)
        """
        data = mx.nd.array(faces_frame)
        db = mx.io.DataBatch(data=(data,))
        self.model.forward(db, is_train=False)
        embeddings = self.model.get_outputs()[0].asnumpy()
        embeddings = preprocessing.normalize(embeddings, axis=1)
        return embeddings
