import cv2
import numpy as np
import math

def create_homographic_matrix(src,dst):
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M, Minv 

def warp_add(img_org,img_bev, Minv):
    newwarp = cv2.warpPerspective(img_bev, Minv, (img_bev.shape[1], img_bev.shape[0])) 
    result = cv2.addWeighted(img_org, 1, newwarp, 0.3, 0)
    return result

def point_polygon(Bbox, polygon):
    x_mid = (Bbox[0]+Bbox[2])//2
    y_mid = Bbox[3]
    label = cv2.pointPolygonTest(polygon,(x_mid,y_mid),False)
    return label

def cal_distance(Bbox):
    h = 1.45  # Assume the height of camera = 1.45 m
    alpha = 0 # Assume the angle of camera = a = 0
    intrinsics_matrix = [480, 270,775.9, 776.9] # Assume the intrinsics_matrix of camera
    u0 = intrinsics_matrix[0]  
    v0 = intrinsics_matrix[1]  
    fx = intrinsics_matrix[2]
    fy = intrinsics_matrix[3]

    y = Bbox[3]
    x = (Bbox[0] + Bbox[2]) // 2
    Q_pie = [x - u0, y - v0]
    gamma_pie = math.atan(Q_pie[1] / fy) 
    beta_pie = alpha + gamma_pie
    if beta_pie == 0:
        beta_pie = 0.01
    O1Q = round(h / math.tan(beta_pie), 1)
    z_in_cam = (h / math.sin(beta_pie)) * math.cos(gamma_pie)
    x_in_cam = z_in_cam * (x - u0) / fx
    y_in_cam = z_in_cam * (y - v0) / fy
    distance = round(math.sqrt(O1Q ** 2 + x_in_cam ** 2), 2)
    return distance

def drawing(img,Bbox,distance):
        x_mid = round((Bbox[0]+Bbox[2])*0.5)
        y_mid = Bbox[3]
        distance = distance
        h, w = img.shape[:2]
        cv2.circle(img,(x_mid,y_mid),3,(0,0,255),3,8,0)
        cv2.line(img,(w//2,h-20),(x_mid,y_mid),(21, 21, 255),2)
        cv2.putText(img,str(distance)+'m',(x_mid,y_mid),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 255, 255),1, cv2.LINE_AA)

# Read Img
img_org = cv2.imread("33.jpg") #960*540*3
img_org2 = img_org.copy()
src = np.array([[470,410],[550,410],[670,515],[390,515]],np.float32)
dst = np.array([[420,0],[605,0],[605,515],[420,515]],np.float32) 

# BEV
h, w = img_org.shape[:2]
M, Minv = create_homographic_matrix(src,dst)
img_bev =  cv2.warpPerspective(img_org2, M,(w,h), flags=cv2.INTER_LINEAR)
cv2.imshow("img_bev",img_bev)

# ROI
pts =  np.array([[420,0],[605,0],[605,515],[420,515]], np.int32) 
cv2.fillPoly(img_bev, [pts], (0,255, 0))

# ICP
img_roi = warp_add(img_org,img_bev,Minv)
img_roi_c = img_roi.copy()

# Assume a BoundingBox
# Bbox = [450,350,550,420]
# distance = cal_distance(Bbox)
# drawing(img_roi_c,Bbox)
# cv2.imshow("img_distance",img_roi_c)  
# cv2.waitKey()

# Assume some Bboxes
# If Points in Polygon
Bboxes = range(10)
for i in Bboxes:
    Bbox = [150+10*i,350,530+50*i,350+15*i]
    label = point_polygon(Bbox,src)

    # distance all
    distance = cal_distance(Bbox)
    drawing(img_roi_c,Bbox,distance)
    if i == len(Bboxes)-1:
        cv2.imshow("img_distance",img_roi_c)  

    # distance in roi
    if label != 1:
        print("BoundingBox is NOT in the roi")
    else:
        distance = cal_distance(Bbox)
        drawing(img_roi,Bbox,distance)
        if i == len(Bboxes)-1:
            cv2.imshow("roi_distance",img_roi)
            cv2.waitKey() 
cv2.imwrite("save_img/all.png",img_roi_c)
cv2.imwrite("save_img/roi.png",img_roi)