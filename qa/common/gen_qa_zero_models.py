# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
from builtins import range
import os
import sys
import numpy as np

FLAGS = None
np_dtype_string = np.dtype(object)

def np_to_model_dtype(np_dtype):
    if np_dtype == np.bool:
        return "TYPE_BOOL"
    elif np_dtype == np.int8:
        return "TYPE_INT8"
    elif np_dtype == np.int16:
        return "TYPE_INT16"
    elif np_dtype == np.int32:
        return "TYPE_INT32"
    elif np_dtype == np.int64:
        return "TYPE_INT64"
    elif np_dtype == np.uint8:
        return "TYPE_UINT8"
    elif np_dtype == np.uint16:
        return "TYPE_UINT16"
    elif np_dtype == np.float16:
        return "TYPE_FP16"
    elif np_dtype == np.float32:
        return "TYPE_FP32"
    elif np_dtype == np.float64:
        return "TYPE_FP64"
    elif np_dtype == np_dtype_string:
        return "TYPE_STRING"
    return None

def np_to_tf_dtype(np_dtype):
    if np_dtype == np.bool:
        return tf.bool
    elif np_dtype == np.int8:
        return tf.int8
    elif np_dtype == np.int16:
        return tf.int16
    elif np_dtype == np.int32:
        return tf.int32
    elif np_dtype == np.int64:
        return tf.int64
    elif np_dtype == np.uint8:
        return tf.uint8
    elif np_dtype == np.uint16:
        return tf.uint16
    elif np_dtype == np.float16:
        return tf.float16
    elif np_dtype == np.float32:
        return tf.float32
    elif np_dtype == np.float64:
        return tf.float64
    elif np_dtype == np_dtype_string:
        return tf.string
    return None

def np_to_c2_dtype(np_dtype):
    if np_dtype == np.bool:
        return c2core.DataType.BOOL
    elif np_dtype == np.int8:
        return c2core.DataType.INT8
    elif np_dtype == np.int16:
        return c2core.DataType.INT16
    elif np_dtype == np.int32:
        return c2core.DataType.INT32
    elif np_dtype == np.int64:
        return c2core.DataType.INT64
    elif np_dtype == np.uint8:
        return c2core.DataType.UINT8
    elif np_dtype == np.uint16:
        return c2core.DataType.UINT16
    elif np_dtype == np.float16:
        return c2core.DataType.FLOAT16
    elif np_dtype == np.float32:
        return c2core.DataType.FLOAT
    elif np_dtype == np.float64:
        return c2core.DataType.DOUBLE
    elif np_dtype == np_dtype_string:
        return c2core.DataType.STRING
    return None

def np_to_trt_dtype(np_dtype):
    if np_dtype == np.int8:
        return trt.infer.DataType.INT8
    elif np_dtype == np.int32:
        return trt.infer.DataType.INT32
    elif np_dtype == np.float16:
        return trt.infer.DataType.HALF
    elif np_dtype == np.float32:
        return trt.infer.DataType.FLOAT
    return None

def create_tf_modelfile(
        create_savedmodel, models_dir, model_version, max_batch, dtype, shape):

    if not tu.validate_for_tf_model(dtype, dtype, dtype, shape, shape, shape):
        return

    tf_dtype = np_to_tf_dtype(dtype)

    # Create the model with a empty shape except for batch, [?]. The
    # shape is completely determined by the batch dimension. The model
    # just copies the input to the output.
    tf.reset_default_graph()
    if max_batch == 0:
        in0 = tf.placeholder(tf_dtype, tu.shape_to_tf_shape(shape), "INPUT")
    else:
        in0 = tf.placeholder(tf_dtype, [None,] + tu.shape_to_tf_shape(shape), "INPUT")

    output0 = tf.identity(in0, name="OUTPUT")

    # Use a different model name for the non-batching variant
    if create_savedmodel:
        model_name = tu.get_model_name(
            "savedmodel_nobatch" if max_batch == 0 else "savedmodel", dtype, dtype, dtype)
    else:
        model_name = tu.get_model_name(
            "graphdef_nobatch" if max_batch == 0 else "graphdef", dtype, dtype, dtype)

    model_version_dir = models_dir + "/" + model_name + "/" + str(model_version)

    try:
        os.makedirs(model_version_dir)
    except OSError as ex:
        pass # ignore existing dir

    if create_savedmodel:
        with tf.Session() as sess:
            input0_tensor = tf.get_default_graph().get_tensor_by_name("INPUT:0")
            output0_tensor = tf.get_default_graph().get_tensor_by_name("OUTPUT:0")
            tf.saved_model.simple_save(sess, model_version_dir + "/model.savedmodel",
                                       inputs={"INPUT": input0_tensor},
                                       outputs={"OUTPUT": output0_tensor})
    else:
        with tf.Session() as sess:
            graph_io.write_graph(sess.graph.as_graph_def(), model_version_dir,
                                 "model.graphdef", as_text=False)

def create_tf_modelconfig(
        create_savedmodel, models_dir, model_version, max_batch, dtype, shape):

    if not tu.validate_for_tf_model(dtype, dtype, dtype, shape, shape, shape):
        return

    # If max_batch is > 0 and shape is empty, then have the case where
    # full batch input shape is a vector [ batch-size ], but each
    # input of the batch still is a single element so set shape to [ 1
    # ] for input.
    shape_str = tu.shape_to_dims_str(shape)
    if (max_batch > 0) and (len(shape) == 0):
        shape_str = tu.shape_to_dims_str([1,])

    # Use a different model name for the non-batching variant
    if create_savedmodel:
        model_name = tu.get_model_name(
            "savedmodel_nobatch" if max_batch == 0 else "savedmodel", dtype, dtype, dtype)
    else:
        model_name = tu.get_model_name(
            "graphdef_nobatch" if max_batch == 0 else "graphdef", dtype, dtype, dtype)

    config_dir = models_dir + "/" + model_name
    config = '''
name: "{}"
platform: "{}"
max_batch_size: {}
input [
  {{
    name: "INPUT"
    data_type: {}
    dims: [ {} ]
  }}
]
output [
  {{
    name: "OUTPUT"
    data_type: {}
    dims: [ {} ]
  }}
]
'''.format(model_name,
           "tensorflow_savedmodel" if create_savedmodel else "tensorflow_graphdef",
           max_batch,
           np_to_model_dtype(dtype), shape_str,
           np_to_model_dtype(dtype), shape_str)

    try:
        os.makedirs(config_dir)
    except OSError as ex:
        pass # ignore existing dir

    with open(config_dir + "/config.pbtxt", "w") as cfile:
        cfile.write(config)


def create_netdef_modelfile(
        create_savedmodel, models_dir, model_version, max_batch, dtype, shape):

    if not tu.validate_for_c2_model(dtype, dtype, dtype, shape, shape, shape):
        return

    c2_dtype = np_to_c2_dtype(dtype)
    model_name = tu.get_model_name(
        "netdef_nobatch" if max_batch == 0 else "netdef", dtype, dtype, dtype)

    # The model just copies the input to the output.
    model = c2model_helper.ModelHelper(name=model_name)
    model.net.Copy("INPUT", "OUTPUT")

    model_version_dir = models_dir + "/" + model_name + "/" + str(model_version)

    try:
        os.makedirs(model_version_dir)
    except OSError as ex:
        pass # ignore existing dir

    with open(model_version_dir + "/model.netdef", "wb") as f:
        f.write(model.Proto().SerializeToString())
    with open(model_version_dir + "/init_model.netdef", "wb") as f:
        f.write(model.InitProto().SerializeToString())


def create_netdef_modelconfig(
        create_savedmodel, models_dir, model_version, max_batch, dtype, shape):

    if not tu.validate_for_c2_model(dtype, dtype, dtype, shape, shape, shape):
        return

    # If max_batch is > 0 and shape is empty, then have the case where
    # full batch input shape is a vector [ batch-size ], but each
    # input of the batch still is a single element so set shape to [ 1
    # ] for input.
    shape_str = tu.shape_to_dims_str(shape)
    if (max_batch > 0) and (len(shape) == 0):
        shape_str = tu.shape_to_dims_str([1,])

    model_name = tu.get_model_name(
        "netdef_nobatch" if max_batch == 0 else "netdef", dtype, dtype, dtype)
    config_dir = models_dir + "/" + model_name
    config = '''
name: "{}"
platform: "caffe2_netdef"
max_batch_size: {}
input [
  {{
    name: "INPUT"
    data_type: {}
    dims: [ {} ]
  }}
]
output [
  {{
    name: "OUTPUT"
    data_type: {}
    dims: [ {} ]
  }}
]
'''.format(model_name, max_batch,
           np_to_model_dtype(dtype), shape_str,
           np_to_model_dtype(dtype), shape_str)

    try:
        os.makedirs(config_dir)
    except OSError as ex:
        pass # ignore existing dir

    with open(config_dir + "/config.pbtxt", "w") as cfile:
        cfile.write(config)


def create_models(models_dir, dtype, shape, no_batch=True):
    model_version = 1

    if FLAGS.graphdef:
        create_tf_modelconfig(False, models_dir, model_version, 8, dtype, shape);
        create_tf_modelfile(False, models_dir, model_version, 8, dtype, shape);
        if no_batch:
            create_tf_modelconfig(False, models_dir, model_version, 0, dtype, shape);
            create_tf_modelfile(False, models_dir, model_version, 0, dtype, shape);

    if FLAGS.savedmodel:
        create_tf_modelconfig(True, models_dir, model_version, 8, dtype, shape);
        create_tf_modelfile(True, models_dir, model_version, 8, dtype, shape);
        if no_batch:
            create_tf_modelconfig(True, models_dir, model_version, 0, dtype, shape);
            create_tf_modelfile(True, models_dir, model_version, 0, dtype, shape);

    if FLAGS.netdef:
        create_netdef_modelconfig(True, models_dir, model_version, 8, dtype, shape);
        create_netdef_modelfile(True, models_dir, model_version, 8, dtype, shape);
        if no_batch:
            create_netdef_modelconfig(True, models_dir, model_version, 0, dtype, shape);
            create_netdef_modelfile(True, models_dir, model_version, 0, dtype, shape);


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models_dir', type=str, required=True,
                        help='Top-level model directory')
    parser.add_argument('--graphdef', required=False, action='store_true',
                        help='Generate GraphDef models')
    parser.add_argument('--savedmodel', required=False, action='store_true',
                        help='Generate SavedModel models')
    parser.add_argument('--netdef', required=False, action='store_true',
                        help='Generate NetDef models')
    FLAGS, unparsed = parser.parse_known_args()

    if FLAGS.netdef:
        from caffe2.python import core as c2core
        from caffe2.python import model_helper as c2model_helper
    if FLAGS.graphdef or FLAGS.savedmodel:
        import tensorflow as tf
        from tensorflow.python.framework import graph_io, graph_util

    import test_util as tu

    # Create a batching model with zero-sized tensors
    create_models(FLAGS.models_dir, np.float32, [], False)
