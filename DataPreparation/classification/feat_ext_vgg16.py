import numpy as np
import sys
import caffe
import os

model_file_dir = '/abs_path/VGG/models'
feat_dir = '/abs_path/ori_features/classification/vgg16'
input_list = 'file_list.npy'
dataset_dir = '/abs_path/imagenet/raw_data/validation'

blobs_out = ['conv1_2', 'pool1', 'conv2_2', 'pool2', 'conv3_3', 'pool3', 'conv4_3',
             'pool4', 'conv5_3', 'pool5', 'fc6', 'fc7', 'fc8', 'prob']

MODEL_DEPLOY_FILE = os.path.join(model_file_dir, 'VGG_ILSVRC_16_layers_deploy.prototxt')

MODEL_WEIGHT_FILE = os.path.join(model_file_dir, 'VGG_ILSVRC_16_layers.caffemodel')

MODEL_ORIGINAL_INPUT_SIZE = 256, 256
MODEL_INPUT_SIZE = 224, 224

MODEL_MEAN_VALUE = np.array([103.939, 116.779, 123.68])
# NOTE: MODEL_MEAN_VALUE.shape is (3,224,224) which is differ from vgg16's mean-value format

oversample = False

def ext_feat(net, inputs, blobs, oversample=False):
  input_ = np.zeros((len(inputs),
                    net.image_dims[0],
                    net.image_dims[1],
                    inputs[0].shape[2]),
                    dtype=np.float32)
  for ix, in_ in enumerate(inputs):
    input_[ix] = caffe.io.resize_image(in_, net.image_dims)
  if oversample:
    # Generate center, corner, and mirrored crops.
    input_ = caffe.io.oversample(input_, net.crop_dims)
  else:
    # Take center crop.
    center = np.array(net.image_dims) / 2.0
    crop = np.tile(center, (1, 2))[0] + np.concatenate([-net.crop_dims / 2.0, net.crop_dims / 2.0])
    input_ = input_[:, int(crop[0]):int(crop[2]), int(crop[1]):int(crop[3]), :]
  caffe_in = np.zeros(np.array(input_.shape)[[0, 3, 1, 2]], dtype=np.float32)
  for ix, in_ in enumerate(input_):
    caffe_in[ix] = net.transformer.preprocess(net.inputs[0], in_)
  # out = self.forward_all(**{self.inputs[0]: caffe_in})
  out = net.forward(**{net.inputs[0]: caffe_in, 'end': 'prob','blobs': blobs})
  feats = {}
  for blob in blobs:
    feats[blob] = []
  for blob in blobs:
    feat = out[blob]
    for i in xrange(caffe_in.shape[0]):
      feats[blob].append(feat[i])
  return feats

if __name__ == '__main__':
  caffe.set_device(0)
  caffe.set_mode_gpu()
  net = caffe.Classifier( \
    model_file=MODEL_DEPLOY_FILE,
    pretrained_file=MODEL_WEIGHT_FILE,
    image_dims=(MODEL_ORIGINAL_INPUT_SIZE[0], MODEL_ORIGINAL_INPUT_SIZE[1]),
    raw_scale=255., # scale befor mean subtraction
    input_scale=None, # scale after mean subtraction
    mean = MODEL_MEAN_VALUE,
    channel_swap = (2, 1, 0) )

  filelist = np.load(input_list)

  hit_count, hit5_count = 0, 0
  for n, fname in enumerate(filelist):
    im = caffe.io.load_image(os.path.join(dataset_dir, fname[0]))
    feats_dic = ext_feat(net, [im], blobs_out, oversample)

    save_path = os.path.join(feat_dir, fname[1])
    os.makedirs(save_path)
    for i in blobs_out:
      np.save(os.path.join(save_path, i+'.npy'), np.array(feats_dic[i]).astype(np.float32))
    print n
    sys.stdout.flush()
