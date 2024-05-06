import copy

import cv2
import numpy as np
import math
import time
import random
from Data.compute_avg_reproj_error import compute_avg_reproj_error


def apply_transform(x, y, T):
    homo = np.array([x, y, 1])
    transformed = np.dot(T, homo)
    transformed /= transformed[2]
    return transformed[0], transformed[1]


def compute_F_cv2(M):
    pts1 = [(pt[0], pt[1]) for pt in M]
    pts2 = [(pt[2], pt[3]) for pt in M]

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)

    F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)

    pts1 = pts1[mask.ravel()==1]
    pts2 = pts2[mask.ravel()==1]

    return F


def compute_F_raw(M):

    A = []

    for i in range(len(M)):
        x = M[i][0]
        y = M[i][1]
        x_ = M[i][2]
        y_ = M[i][3]

        Ai = [
            x * x_, x * y_, x, y * x_, y * y_, y, x_, y_, 1
        ]

        A.append(Ai)

    A = np.array(A)
    u, s, vh = np.linalg.svd(A)

    min_idx = np.argmin(s)
    F = vh[min_idx].reshape((3, 3))
    F /= F[2][2]

    return F


def compute_F_norm(M):
    global img_shape
    A = []
    for i in range(len(M)):
        x = M[i][0]
        y = M[i][1]
        x_ = M[i][2]
        y_ = M[i][3]

        T_trans = [
            [2/img_shape[1], 0, -1],
            [0, 2/img_shape[0], -1],
            [0, 0, 1]
        ]

        x, y = apply_transform(x, y, T_trans)
        x_, y_ = apply_transform(x_, y_, T_trans)

        Ai = [
            x * x_, x * y_, x, y * x_, y * y_, y, x_, y_, 1
        ]

        A.append(Ai)

    A = np.array(A)
    u, s, vh = np.linalg.svd(A)

    min_idx = np.argmin(s)
    F = vh[min_idx].reshape((3, 3))

    F_denorm = np.dot(np.dot(np.transpose(T_trans), F), T_trans)
    F_denorm /= F_denorm[2][2]
    return F_denorm


def compute_F_mine(M, th):
    global img_shape
    time_i = time.time()
    F_max = []
    inliers_max = -1
    inliers_max_list = []
    while True:
        indices = random.sample(range(len(M)), 8)

        # srcP_sampled = srcP[np.array(indices)]
        # destP_sampled = destP[np.array(indices)]
        M_sampled = M[np.array(indices)]

        # H_cur = compute_homography(srcP_sampled, destP_sampled)
        F_cur = compute_F_norm(M_sampled)

        inliers_cnt = 0
        inliers_list = []
        for i, match in enumerate(M):
            # dest = destP[i]
            homo_location = [match[0], match[1], 1]

            l = np.dot(F_cur, homo_location)

            dist = math.fabs(l[0] * match[2] + l[1] * match[3] + l[2]) / math.sqrt(l[0] ** 2 + l[1] ** 2)
            if math.fabs(dist) <= th:
                inliers_cnt += 1
                inliers_list.append(i)

        if inliers_cnt > inliers_max:
            F_max = F_cur
            inliers_max_list = inliers_list
            inliers_max = inliers_cnt

        if time.time() - time_i > 2.7:
            break

    # srcP_inliers = srcP[np.array(inliers_max_list)]
    # destP_inliers = destP[np.array(inliers_max_list)]
    M_inliers = M[np.array(inliers_max_list)]

    F_result = compute_F_norm(M_inliers)

    return F_result


def draw_epipolar(M, F, img1, img2):
    _, X1, _ = img1.shape
    _, X2, _ = img2.shape

    while True:
        img1_draw = copy.deepcopy(img1)
        img2_draw = copy.deepcopy(img2)

        pts1 = [(pt[0], pt[1]) for pt in M]
        pts2 = [(pt[2], pt[3]) for pt in M]

        pts1 = np.int32(pts1)
        pts2 = np.int32(pts2)

        indices = random.sample(range(len(M)), 3)

        pts1_sampled = pts1[np.array(indices)]
        pts2_sampled = pts2[np.array(indices)]

        p1 = pts2_sampled[0]
        p2 = pts2_sampled[1]
        p3 = pts2_sampled[2]
        q1 = pts1_sampled[0]
        q2 = pts1_sampled[1]
        q3 = pts1_sampled[2]

        l1 = np.dot([p1[0], p1[1], 1], F)
        x0 = 0
        y0 = int(-l1[2]/l1[1])
        x1 = X1
        y1 = int(-(l1[2]+l1[0]*X1)/l1[1])
        img1_draw = cv2.line(img1_draw, (x0, y0), (x1, y1), (0, 0, 255), 1)
        img2_draw = cv2.circle(img2_draw, p1, 5, (0, 0, 255), -1)

        l2 = np.dot([p2[0], p2[1], 1], F)
        x0 = 0
        y0 = int(-l2[2]/l2[1])
        x1 = X1
        y1 = int(-(l2[2]+l2[0]*X1)/l2[1])
        img1_draw = cv2.line(img1_draw, (x0, y0), (x1, y1), (0, 255, 0), 1)
        img2_draw = cv2.circle(img2_draw, p2, 5, (0, 255, 0), -1)

        l3 = np.dot([p3[0], p3[1], 1], F)
        x0 = 0
        y0 = int(-l3[2]/l3[1])
        x1 = X1
        y1 = int(-(l3[2]+l3[0]*X1)/l3[1])
        img1_draw = cv2.line(img1_draw, (x0, y0), (x1, y1), (255, 0, 0), 1)
        img2_draw = cv2.circle(img2_draw, p3, 5, (255, 0, 0), -1)

        m1 = np.dot([q1[0], q1[1], 1], np.transpose(F))
        x0 = 0
        y0 = int(-m1[2]/m1[1])
        x1 = X1
        y1 = int(-(m1[2]+m1[0]*X1)/m1[1])
        img2_draw = cv2.line(img2_draw, (x0, y0), (x1, y1), (0, 0, 255), 1)
        img1_draw = cv2.circle(img1_draw, q1, 5, (0, 0, 255), -1)

        m2 = np.dot([q2[0], q2[1], 1], np.transpose(F))
        x0 = 0
        y0 = int(-m2[2]/m2[1])
        x1 = X1
        y1 = int(-(m2[2]+m2[0]*X1)/m2[1])
        img2_draw = cv2.line(img2_draw, (x0, y0), (x1, y1), (0, 255, 0), 1)
        img1_draw = cv2.circle(img1_draw, q2, 5, (0, 255, 0), -1)

        m3 = np.dot([q3[0], q3[1], 1], np.transpose(F))
        x0 = 0
        y0 = int(-m3[2]/m3[1])
        x1 = X1
        y1 = int(-(m3[2]+m3[0]*X1)/m3[1])
        img2_draw = cv2.line(img2_draw, (x0, y0), (x1, y1), (255, 0, 0), 1)
        img1_draw = cv2.circle(img1_draw, q3, 5, (255, 0, 0), -1)

        concat = cv2.hconcat([img1_draw,img2_draw])

        cv2.imshow('image visualization', concat)
        action = cv2.waitKey()
        cv2.destroyAllWindows()

        if action == ord('q'):
            break



if __name__=="__main__":
    img1 = cv2.imread('./Data/temple1.png')
    img2 = cv2.imread('./Data/temple2.png')
    M = np.loadtxt('Data/temple_matches.txt')
    img_shape = img1.shape

    print("Average Reprojection Errors (temple1.png and temple2.png)")

    F = compute_F_raw(M)
    print("\tRaw = ", compute_avg_reproj_error(M, F))
    F = compute_F_norm(M)
    print("\tNorm = ", compute_avg_reproj_error(M, F))
    F = compute_F_mine(M, th=10)
    print("\tMine = ", compute_avg_reproj_error(M, F))

    draw_epipolar(M, F, img1, img2)

    img1 = cv2.imread('./Data/house1.jpg')
    img2 = cv2.imread('./Data/house2.jpg')
    M = np.loadtxt('Data/house_matches.txt')
    img_shape = img1.shape

    print("Average Reprojection Errors (house1.jpg and house2.jpg)")

    F = compute_F_raw(M)
    print("\tRaw = ", compute_avg_reproj_error(M, F))
    F = compute_F_norm(M)
    print("\tNorm = ", compute_avg_reproj_error(M, F))
    F = compute_F_mine(M, th=10)
    print("\tMine = ", compute_avg_reproj_error(M, F))

    draw_epipolar(M, F, img1, img2)

    img1 = cv2.imread('./Data/library1.jpg')
    img2 = cv2.imread('./Data/library2.jpg')
    M = np.loadtxt('Data/library_matches.txt')
    img_shape = img1.shape

    print("Average Reprojection Errors (library1.jpg and library2.jpg)")

    F = compute_F_raw(M)
    print("\tRaw = ", compute_avg_reproj_error(M, F))
    F = compute_F_norm(M)
    print("\tNorm = ", compute_avg_reproj_error(M, F))
    F = compute_F_mine(M, th=10)
    print("\tMine = ", compute_avg_reproj_error(M, F))

    draw_epipolar(M, F, img1, img2)
