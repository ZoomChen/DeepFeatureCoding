import numpy as np
import sys
import caffe
import os

model_file_dir = '/abs_path/deep-residual-networks/models'
feat_dir = '/abs_path/ori_features/classification/res152'
input_list = 'file_list.npy'
dataset_dir = '/abs_path/imagenet/raw_data/validation'

blobs_out = ['conv1', 'pool1', 'res2c', 'res3b7', 'res4b35', 'res5c', 'pool5','fc1000']

MODEL_DEPLOY_FILE = os.path.join(model_file_dir, 'ResNet-152-deploy.prototxt')

MODEL_WEIGHT_FILE = os.path.join(model_file_dir, 'ResNet-152-model.caffemodel')

MODEL_ORIGINAL_INPUT_SIZE = 256, 256
MODEL_INPUT_SIZE = 224, 224
MODEL_MEAN_FILE = os.path.join(model_file_dir, 'ResNet_mean.binaryproto')
blob = caffe.proto.caffe_pb2.BlobProto()
data = open(MODEL_MEAN_FILE, 'rb').read()
blob.ParseFromString(data)
MODEL_MEAN_VALUE = np.squeeze(np.array( caffe.io.blobproto_to_array(blob) ))
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

# 152 shape
# (1, 64, 56, 56)
# (1, 256, 56, 56)
# (1, 512, 28, 28)
# (1, 1024, 14, 14)
# (1, 2048, 7, 7)
# (1, 2048, 1, 1)
# (1, 1000)