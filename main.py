import cv2
import numpy as np
import time

def kp_des(templates):
     
    # get template
    kp_des_train = []
    for template in templates:
        
        img_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        # find the keypoints and descriptors with SIFT
        #img_template = template
        kp_template, des_template = detector.detectAndCompute(img_template,None)
        kp_des_train.append((kp_template, des_template))
    
    return kp_des_train

def find_matches(des_query, des_train, kp1, kp2):
    
    key_matches = 0
    
    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 10)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    if(len(kp1)>=2 and len(kp2)>=2) :
        
        matches = flann.knnMatch(des_query, des_train, k=2)
    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]
    
    # print(matches)
    # ratio test
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matchesMask[i]=[1,0]
            key_matches = key_matches + 1
    
    #print('key_matches: ', key_matches)
    return(key_matches, matches)


def temp_query_match(templates, rgb_frame, kp_des_train, kp_img_rgb_frame, des_img_rgb_frame):
    
    # run a loop through template images
    #templates
    queryBorders = []
    for i,template in enumerate(templates):


        trainImg = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        key_matches, matches = find_matches(des_img_rgb_frame, kp_des_train[i][1], kp_img_rgb_frame, kp_des_train[i][0])
        if matches == 0:
            continue

        if key_matches > MIN_MATCH_COUNT:
            goodMatch=[]
            for m,n in matches:
                if(m.distance<0.55*n.distance):
                    goodMatch.append(m)

            if(len(goodMatch)>MIN_MATCH_COUNT):
                    tp=[]
                    qp=[]
                    for m in goodMatch:
                        tp.append(kp_des_train[i][0][m.trainIdx].pt)
                        qp.append(kp_img_rgb_frame[m.queryIdx].pt)
                    tp,qp=np.float32((tp,qp))

                    H,status=cv2.findHomography(tp,qp,cv2.RANSAC,3.0)
                    
                    h,w=trainImg.shape
                    trainBorder=np.float32([[[0,0],[0,h-1],[w-1,h-1],[w-1,0]]])
                    if  H is not None:
                        queryBorder=cv2.perspectiveTransform(trainBorder,H)
                        queryBorders.append(np.int32(queryBorder))
                        #return cv2.polylines(rgb_frame,[np.int32(queryBorder)],True,(0,255,0),2)
                    
    return queryBorders #rgb_frame


def add_noise(image):
    row,col= image.shape
    mean = 0
    var = 0.1
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col))
    gauss = gauss.reshape(row,col)
    noisy = image + gauss
    return noisy



MIN_MATCH_COUNT = 4

detector=cv2.SIFT_create()


#templates = [sub_image]
template1 = cv2.imread('template1.png')
template2 = cv2.imread('template1_1.jpg')
template3 = cv2.imread('template1_2.png')
template4 = cv2.imread('template3.png')
template5 = cv2.imread('template3_1.jpg')
template6 = cv2.imread('sentry1.png')

templates = [template1, template2, template3, template4, template5, template6]


kp_des_train = kp_des(templates)

cap = cv2.VideoCapture("vid.mp4")

while cap.isOpened():

    _, rgb_frame = cap.read()

    img_rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
    #img_rgb_frame = rgb_frame
    kp_img_rgb_frame, des_img_rgb_frame = detector.detectAndCompute(img_rgb_frame,None)
    t0 = time.perf_counter()
    queryBorders = temp_query_match(templates, rgb_frame, kp_des_train, kp_img_rgb_frame, des_img_rgb_frame)
    output = cv2.polylines(rgb_frame,queryBorders,True,(0,255,0),2)
    t1 = time.perf_counter()
    print(f"time for frame = {t1 - t0:0.4f} seconds")
    cv2.imshow("frame", output)
    if cv2.waitKey(33) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()