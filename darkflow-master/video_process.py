import cv2
from darkflow.net.build import TFNet
import numpy as np
import time

option = {
	'model': 'cfg/yolo.cfg'
	'load': 'bin/yolov2.weights',
	'threshold': 0.15
	'gpu': 1.5
}

tfnet = TFNet(option)

capture = cv2.VideoCapture('SR.avi')
colors = [tuple(255 * np.random.rand(3) for i in range(5)]

while(capture.isOpened()):
	stime = time.time()
	ret, frame = capture.read
	results = tfnet.return_predict()
	if ret:
		for color, result in zip(colors, results):
			t1 = result['topleft']['x'], result['topleft']['y']
			br = result['bottomright']['x'], result['botomleft']['y']
			label = result['label']
			frame = cv2.rectangle(frame, t1, br, colors, 7) 
			frame = cv2.putText(frame, label, cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 2)
		cv2.imshow('frame', frame)
		print ('FPS {:.1f}'.format(1/ (time.time() - stime )))
		if cv2.waitkey(1) & 0xFF == ord('q'):
			break	

	else:
		capture.release()
		capture.destroyALLWindows()
		break
