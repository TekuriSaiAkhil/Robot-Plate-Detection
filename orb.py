import numpy as np 
import cv2 

	
# Read the query image as query_img 
# and train image This query image 
# is what you need to find in train image 
# Save it in the same directory 
# with the name image.jpg 
query_img = cv2.imread('original.png') 
train_img = cv2.imread('template3.png') 

# Convert it to grayscale 
query_img_bw = cv2.cvtColor(query_img,cv2.COLOR_BGR2GRAY) 
train_img_bw = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY) 

# Initialize the ORB detector algorithm 
orb = cv2.ORB_create(fastThreshold=10, edgeThreshold=20) 

# Now detect the keypoints and compute 
# the descriptors for the query image 
# and train image 
queryKeypoints, queryDescriptors = orb.detectAndCompute(query_img_bw,None) 
trainKeypoints, trainDescriptors = orb.detectAndCompute(train_img_bw,None) 

# Initialize the Matcher for matching 
# the keypoints and then match the 
# keypoints 
matcher = cv2.BFMatcher() 
matches = matcher.match(queryDescriptors,trainDescriptors) 

# draw the matches to the final image 
# containing both the images the drawMatches() 
# function takes both images and keypoints 
# and outputs the matched query image with 
# its train image 
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        key_matches = key_matches + 1
    
    #print('key_matches: ', key_matches)
print((key_matches, matches))

goodMatch = matches

if(len(goodMatch)>4):
    tp=[]
    qp=[]
    for m in goodMatch:
        tp.append(trainKeypoints[m.trainIdx].pt)
        qp.append(queryKeypoints[m.queryIdx].pt)
    tp,qp=np.float32((tp,qp))
    H,status=cv2.findHomography(tp,qp,cv2.RANSAC,3.0)
                    
    h,w=train_img_bw.shape
    trainBorder=np.float32([[[0,0],[0,h-1],[w-1,h-1],[w-1,0]]])
    if  H is not None:
        queryBorder=cv2.perspectiveTransform(trainBorder,H)
        final_img = cv2.polylines(query_img,[np.int32(queryBorder)],True,(0,255,0),2)
        cv2.imshow("Matches", final_img) 
        cv2.waitKey()

# Show the final image 

