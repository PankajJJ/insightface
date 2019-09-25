import os

import numpy as np
import tensorrt as trt
import pycuda.driver as cuda  # noqa, must be imported
import pycuda.autoinit  # noqa, must be imported
from sklearn import preprocessing

import common

TRT_LOGGER = trt.Logger()

class FaceModelWithTRT(object):
    def __init__(self, args):
        self.engine_file = args.model
        self.engine = None
        self.inputs = None
        self.outputs = None
        self.bindings = None
        self.stream = None
        self.context = None

    def build(self):
        with open(self.engine_file, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.inputs, self.outputs, self.bindings, self.stream = common.allocate_buffers(
            self.engine)
        self.context = self.engine.create_execution_context()

    def get_faces_feature(self, objects_frame):
        # lazy load implementation
        if self.engine is None:
            self.build()

        self.inputs[0].host = np.array(objects_frame, dtype=np.float32, order='C')
        trt_outputs = common.do_inference(
            self.context, bindings=self.bindings,
            inputs=self.inputs, outputs=self.outputs, stream=self.stream)
        
        embeddings = preprocessing.normalize(trt_outputs)
        return embeddings