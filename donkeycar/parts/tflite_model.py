'''

tflite_model.py

Represents the TFLite model.

'''

import numpy as np
import donkeycar as dk

import tensorflow as tf
from tflite_runtime import Interpreter
from tensorflow.python.framework import graph_util


class TfLiteModel(object):

  def __init__(self, model, keras_model, testing=False):
    # Store keras model for testing.
    self._keras_model = keras_model

    # Boolean testing flag.
    self._testing = testing

    # Initialize interpreter.
    self._interpreter = Interpreter(model_path=model)
    self._interpreter.allocate_tensors()


# Handles TFLite models converted from donkeycar.parts.keras.KerasCategorical
# models.
class TfLiteCategorical(TfLiteModel):

  def __init__(self, model, keras_model, testing=False):
    super(TfLiteCategorical, self).__init__(model, keras_model, testing)

    # Create references to tensors.
    self.image_input_tensor = self._interpreter.get_input_details()[0]['index']

    for tensor_details in self._interpreter.get_output_details():
      if 'angle_out' in tensor_details['name']:
        self.angle_tensor = tensor_details['index']
      elif 'throttle_out' in tensor_details['name']:
        self.throttle_tensor = tensor_details['index']
      else:
        raise RuntimeError(
            "Unexpected extra output in model %r" % tensor_details['name'])

  def run(self, image_array):
    # Gets keras values for testing.
    k_steering, k_throttle = None, None
    if self._testing:
      k_steering, k_throttle = self._keras_model.run(image_array)

    # Gets tflite values.
    image_array = image_array.astype(np.float32)[None]
    self._interpreter.set_tensor(self.image_input_tensor, image_array)
    self._interpreter.invoke()

    angle = self._interpreter.get_tensor(self.angle_tensor)
    throttle = self._interpreter.get_tensor(self.throttle_tensor)
    angle_unbinned = dk.utils.linear_unbin(angle)

    # Compares tflite and keras results.
    if self._testing:
      if k_steering != angle_unbinned or k_throttle != throttle[0][0]:
        print("steering: {} vs {}".format(k_steering, angle_unbinned))
        print("throttle: {} vs {}".format(k_throttle, throttle[0][0]))
      else:
        print("CORRECT!")
    else:
      print("steer: {} throttle: {}".format(angle_unbinned, throttle[0][0]))

    return angle_unbinned, throttle[0][0]



# Converts the model to a TFLite flatbuffer. Returns the filename of the file
# containing the TFLite flatbuffer.
def convert_keras_to_tflite(keras_model, model_name):
  output_tensors = keras_model.model.outputs
  output_tensor_names = [tensor.name.split(":")[0] for tensor in output_tensors]

  # Load TF GraphDef.
  sess = keras_backend.get_session()

  graph_def = sess.graph.as_graph_def()
  frozen_graph = graph_util.convert_variables_to_constants(
     sess, graph_def, output_tensor_names)

  # Convert to .pb.
  output_directory = "."
  filename = "{}.pb".format(model_name)
  tf.train.write_graph(frozen_graph, output_directory, filename,
                       as_text=False)

  # Convert to .pbtxt.
  filename = "{}.pbtxt".format(model_name)
  tf.train.write_graph(frozen_graph, output_directory, filename,
                       as_text=True)

  # Convert to .tflite.
  input_tensor = keras_model.model.inputs[0]
  input_tensor.get_shape = lambda : [1,120,160,3]
  filename = "{}.tflite".format(model_name)
  tflite_model = tf.contrib.lite.toco_convert(
      frozen_graph, [input_tensor], output_tensors)

  with open(filename, "wb") as fp:
    fp.write(tflite_model)
    fp.flush()
  return filename
