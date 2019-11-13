import cv2
import numpy as np


def cv_show(name, image):
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#获取sift关键点
def detectAndCompute(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    (kps, features) = sift.detectAndCompute(image, None)
    kps = np.float32([kp.pt for kp in kps])  # 得到的点需要进一步转换才能使用
    return (kps, features)

#采用RANSAC方法 获取两个图片关键点的相似矩阵
def matchKeyPoints(kpsA, kpsB, featuresA, featuresB, ratio=0.75, reprojThresh=4.0):
    #  ratio是最近邻匹配的推荐阈值
    #  reprojThresh是随机取样一致性的推荐阈值
    matcher = cv2.BFMatcher()
    rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
    matches = []
    for m in rawMatches:
        if len(m) == 2 and m[0].distance < ratio * m[1].distance:
            matches.append((m[0].queryIdx, m[0].trainIdx))
    kpsA = np.float32([kpsA[m[0]] for m in matches])  # 使用np.float32转化列表
    kpsB = np.float32([kpsB[m[1]] for m in matches])
    (M, status) = cv2.findHomography(kpsA, kpsB, cv2.RANSAC, reprojThresh)
    return (M, matches, status)  # 并不是所有的点都有匹配解，它们的状态存在status中

#拼接图片
def stich(imgA, imgB, M):
    result = cv2.warpPerspective(imgA, M, (imgA.shape[1] + imgB.shape[1], imgA.shape[0]))
    result[0:imageA.shape[0], 0:imageB.shape[1]] = imageB
    cv_show('result', result)





# 读取图像
imageA = cv2.imread('./left.png')
cv_show("imageA", imageA)
imageB = cv2.imread('./right.png')
cv_show("imageB", imageB)
# 计算SIFT特征点和特征向量
(kpsA, featuresA) = detectAndCompute(imageA)
(kpsB, featuresB) = detectAndCompute(imageB)
# 基于最近邻和随机取样一致性得到一个单应性矩阵
(M, matches, status) = matchKeyPoints(kpsA, kpsB, featuresA, featuresB)
# 拼接
stich(imageA, imageB, M)

