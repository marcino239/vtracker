#!/usr/bin/python

import sys

caffe_root = '/home/user/work/caffe/'
sys.path.insert( 0, caffe_root + 'python' )
import caffe
import numpy as np
import pandas as pd

import serial
import time

MODEL_FILE = caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt'
PRETRAINED = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
LABELS_FILE = caffe_root + '/data/ilsvrc12/synset_words.txt'

import cv2
from video import create_capture

if __name__ == '__main__':

#	#open serial
#	ser = serial.Serial( '/dev/ttyUSB0', 115200 )

	# caffe run mode
	caffe.set_mode_gpu()
	
	net = caffe.Classifier(MODEL_FILE, PRETRAINED,
			mean=np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1),
			channel_swap=(2,1,0),
			raw_scale=255,
			image_dims=(256, 256))

	IMAGE_FILE = caffe_root + 'examples/images/cat.jpg'

	with open( LABELS_FILE ) as f:
		labels_df = pd.DataFrame([
			{
				'synset_id': l.strip().split(' ')[0],
				'name': ' '.join(l.strip().split(' ')[1:]).split(',')[0]
			}
			for l in f.readlines()
		])
	labels = labels_df.sort('synset_id')['name'].values

	input_image = caffe.io.load_image(IMAGE_FILE)
	prediction = net.predict([input_image])  # predict takes any number of images, and formats them for the Caffe net automatically
	print 'prediction shape:', prediction[0].shape
	print 'predicted class:', prediction[0].argmax()
	label = labels[ prediction[0].argmax() ]
	print( label )

	# load input and configure preprocessing
	im = caffe.io.load_image( IMAGE_FILE )
		
	transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
	transformer.set_mean('data', np.load( caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy' ).mean(1).mean(1))
	transformer.set_transpose('data', (2,0,1))
	transformer.set_channel_swap('data', (2,1,0))
	transformer.set_raw_scale('data', 255.0)

	tr_im = transformer.preprocess('data', im)
	
	# make classification map by forward and print prediction indices at each location
	out = net.forward_all(data=np.asarray([ tr_im ]))
	print out['prob'][0].argmax(axis=0)
	label = labels[ prediction[0].argmax() ]
	print( label )

	# set the cv2
	video_source = '1'
	video_capture = create_capture( video_source )

	pos_pan_increment = 3
	pos_tilt_increment = 3

	pos_pan = 70
	pos_tilt = 70
#	ser.write( '%d %d\n' % (pos_pan, pos_tilt ) )
	
	while True:
		ret, img = video_capture.read()
		cv2.imshow( 'capture ' + video_source, img )

		# move camera
		pos_pan = pos_pan + pos_pan_increment

		if pos_pan >= 110:
			pos_tilt = pos_tilt + pos_tilt_increment
			pos_pan = 70
		
		if pos_tilt >= 110:
			pos_tilt = 70

#		ser.write( '%d %d\n' % (pos_pan, pos_tilt ) )

		# get the prediction
		tr_im = transformer.preprocess('data', img)
		out = net.forward_all(data=np.asarray([ tr_im ]))
		label = out['prob'][0].argmax(axis=0)
		
#		print( '%d:%d - %s' % ( pos_pan, pos_tilt, str( labels[ label ] ) ) )
		print( labels[ label ] )

		ch = 0xFF & cv2.waitKey(1)
		if ch == 27:
			break
		if ch == ord(' '):
#			ser.write( '90 90\n' )
#			ser.flush()
#			ser.close()
			print( 'got space' )
	
	cv2.destroyAllWindows()
