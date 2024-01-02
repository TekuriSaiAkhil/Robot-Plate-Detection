import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np 

#Function returns the feature matched image
def Flanned_Matcher(main_image,sub_image):

	# Initiating the SIFT detector
	#sift = cv.SIFT_create(contrastThreshold = 0.01, edgeThreshold=10)
	orb = cv.ORB_create(fastThreshold=10, edgeThreshold=20) 

	#Find the keypoints and descriptors with SIFT.
	key_point1, descr1 = orb.detectAndCompute(main_image,None)
	key_point2, descr2 = orb.detectAndCompute(sub_image,None)

	descr1 = np.float32(descr1)
	descr2 = np.float32(descr2)


	# FLANN parameters.
	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks=50) 

	# FLANN based matcher with implementation of k nearest neighbour.
	flann = cv.FlannBasedMatcher(index_params,search_params)
	matches = flann.knnMatch(descr1,descr2,k=2)

	# selecting only good matches.
	matchesMask = [[0,0] for i in range(len(matches))]
	key_matches = 0
	# ratio test.
	for i,(m,n) in enumerate(matches):
		if( m.distance < 0.1*n.distance):
			matchesMask[i]=[1,0]
			key_matches = key_matches+1

	print(key_matches)
	draw_params = dict(matchColor = (0,255,0),
					singlePointColor = (255,0,0),
					matchesMask = matchesMask,flags = 0)
	
	# drawing nearest neighbours
	img = cv.drawMatchesKnn(main_image,
							key_point1,
							sub_image,
							key_point2,
							matches,
							None,
							**draw_params)
	return img

# reading two input images
main_image = cv.imread('original.png')
sub_image = cv.imread('sentry1.png')

#Passing two input images
output=Flanned_Matcher(main_image,sub_image)

# Save the image
#cv.imwrite('Match.jpg', output)

plt.imshow(output)
plt.show()
