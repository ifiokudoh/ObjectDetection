# Model to use
# MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17' # BOXES
MODEL_NAME = 'ssd_mobilenet_v2_coco_2018_03_29' # BOXES
# MODEL_NAME = 'faster_rcnn_nas_coco_2018_01_28' # BOXES
# MODEL_NAME = 'mask_rcnn_inception_v2_coco_2018_01_28'  # MASKS

# standard packages
import glob
import os
import sys

# get models from url and untar
import six.moves.urllib as urllib
import tarfile

# import TensorFlow in compat v1 mode for object detection
import tensorflow.compat.v1 as tf

# extend path to tensorflow models
TF_MODELS_RESEARCH_PATH = '/home/ifiok/github/Tensorflow/models/research'
sys.path.append(TF_MODELS_RESEARCH_PATH)

# import packages from object_detection
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# import number and plotting packages
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from PIL import Image

# use Tkinter to visualize image plots
matplotlib.use('TkAgg')

# Models download addresses
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Models files
PATH_TO_FROZEN_INFERENCE_GRAPH = os.path.join(MODEL_NAME, 'frozen_inference_graph.pb')
PATH_TO_GRAPH_LABELS = os.path.join(TF_MODELS_RESEARCH_PATH, 'object_detection', 'data', 'mscoco_label_map.pbtxt')
MAX_NUM_CLASSES = 90

# Images to test
TEST_IMAGE_MASK = os.path.join(TF_MODELS_RESEARCH_PATH, 'object_detection', 'test_images', 'image*.jpg')
MATPLOT_IMAGE_DISPLAY_SIZE = [9.0, 6.0]


"""
Download model from internet and extract its frozen inference graph.
"""
if not os.path.exists(PATH_TO_FROZEN_INFERENCE_GRAPH):
    if not os.path.exists(MODEL_FILE):
        print("Download model file from: " + DOWNLOAD_BASE + MODEL_FILE)
        opener = urllib.request.URLopener()
        opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
    print("Extract frozen inference graph to: " + PATH_TO_FROZEN_INFERENCE_GRAPH)
    tar_file = tarfile.open(MODEL_FILE)
    for file in tar_file.getmembers():
        file_name = os.path.basename(file.name)
        if 'frozen_inference_graph.pb' in file_name:
            tar_file.extract(file, os.getcwd())


"""
Load model graph and labels
"""
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_INFERENCE_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_GRAPH_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=MAX_NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)


"""
Convert image data into a numPy array for processing
"""
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


"""
Code runs the inference for a single image and detects the objects.
Also make boxes and display the class and score of each particular object
"""
def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)

            # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
            real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)

            # The following processing is only for single image
            detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
            detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])

            if 'detection_masks' in tensor_dict:
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)

            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]

            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
            else:
                output_dict['detection_masks'] = None

            return output_dict


"""
Final loop will run the inference on all input images one by one.
The output per image is: detected objects with labels and score
based or similar to the training data.
"""
for image_path in glob.glob(TEST_IMAGE_MASK):
    print('Processing image: ' + image_path)
    image = Image.open(image_path)
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = load_image_into_numpy_array(image)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
    output_dict = run_inference_for_single_image(image_np, detection_graph)
    # Visualization of the results of a detection.
    # print("Before")
    # print(output_dict)
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks'),
        use_normalized_coordinates=True,
        line_thickness=8)
    # print("After")
    # print(output_dict)
    plt.figure(figsize=MATPLOT_IMAGE_DISPLAY_SIZE)
    plt.imshow(image_np)

plt.show()
