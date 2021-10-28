# Writer : Aji Pamungkas
# github: @aji-ptn


import numpy as np
import cv2

# from main import Controller

STAGE_FIRST_FRAME = 0
STAGE_SECOND_FRAME = 1
STAGE_DEFAULT_FRAME = 2
kMinNumFeature = 2000

lk_params = dict(winSize=(21, 21),
                 # maxLevel = 3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


def featureTracking(image_ref, image_cur, px_ref):
    kp2, st, err = cv2.calcOpticalFlowPyrLK(image_ref, image_cur, px_ref, None, **lk_params)  # shape: [k,2] [k,1] [k,1]

    st = st.reshape(st.shape[0])
    kp1 = px_ref[st == 1]
    kp2 = kp2[st == 1]

    return kp1, kp2


class Intrinsic_parameters(object):
    def __init__(self, width, height, fx, fy, cx, cy,
                 k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.distortion = (abs(k1) > 0.0000001)
        self.d = [k1, k2, p1, p2, k3]


class VisualOdometry(object):
    def __init__(self, parameters, annotations):
        # self.Controller = Controller(self)
        # print(self.select_detector)
        self.select_detector = None
        self.frame_stage = 0
        self.cam = parameters
        self.new_frame = None
        self.last_frame = None
        self.scale_3D = []
        self.scale_2D = []
        self.cur_R = None
        self.cur_t = None
        self.px_ref = None
        self.px_cur = None
        self.focal = parameters.fx
        self.pp = (parameters.cx, parameters.cy)
        self.trueX, self.trueY, self.trueZ = 0, 0, 0
        self.detection()
        with open(annotations) as f:
            self.annotations = f.readlines()

    def detection(self):
        self.select_detector = 3
        if self.select_detector == 1:
            self.detector = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)
            self.text = "Fast tres 25"
        elif self.select_detector == 2:
            self.detector = cv2.ORB_create(3000)
            self.text = "ORB"
        elif self.select_detector == 3:
            self.detector = cv2.xfeatures2d.SURF_create()
            self.text = "SURF_create"
        elif self.select_detector == 4:
            self.detector = cv2.xfeatures2d.SIFT_create()
            self.text = "SIFT"
        elif self.select_detector == 6:
            self.detector = cv2.AKAZE_create()
            self.text = "AKAZE"
        elif self.select_detector == 5:
            self.detector = cv2.KAZE_create()
            self.text = "Kaze"

    def getAbsoluteScale(self, frame_id):
        if frame_id <= 2:
            x_prev = 0
            y_prev = 0
            z_prev = 0
        else:
            ss = self.annotations[frame_id - 1].strip().split()
            x_prev = float(ss[3])
            y_prev = float(ss[7])
            z_prev = float(ss[11])
        ss = self.annotations[frame_id].strip().split()
        x = float(ss[3])
        y = float(ss[7])
        z = float(ss[11])
        self.trueX, self.trueY, self.trueZ = x, y, z
        scale_3D = np.sqrt((x - x_prev) * (x - x_prev) + (y - y_prev) * (y - y_prev) + (z - z_prev) * (z - z_prev))
        scale_2D = np.sqrt((x - x_prev) * (x - x_prev) + (z - z_prev) * (z - z_prev))
        return x, y, z, scale_3D, scale_2D

    def processFirstFrame(self):  # first image process
        self.px_ref = self.detector.detect(self.new_frame)
        # print("feature detection = " + str(len(self.px_ref)))
        self.px_ref = np.array([x.pt for x in self.px_ref], dtype=np.float32)
        self.frame_stage = 1

    def processSecondFrame(self):  # second image process
        self.px_ref, self.px_cur = featureTracking(self.last_frame, self.new_frame, self.px_ref)
        E, mask = cv2.findEssentialMat(self.px_cur, self.px_ref, focal=self.focal, pp=self.pp, method=cv2.RANSAC,
                                       prob=0.999, threshold=1.0)
        _, self.cur_R, self.cur_t, mask = cv2.recoverPose(E, self.px_cur, self.px_ref, focal=self.focal,
                                                          pp=self.pp)
        self.frame_stage = 2
        self.px_ref = self.px_cur
        # print("feature detection = " + (str(len(self.px_cur))))

    def processFrame(self, frame_id):
        self.px_ref, self.px_cur = featureTracking(self.last_frame, self.new_frame, self.px_ref)
        E, mask = cv2.findEssentialMat(self.px_cur, self.px_ref, focal=self.focal, pp=self.pp, method=cv2.RANSAC,
                                       prob=0.999, threshold=1.0)
        # print(mask)
        _, R, t, mask = cv2.recoverPose(E, self.px_cur, self.px_ref, focal=self.focal, pp=self.pp)

        # self.cur_R = R
        # print(R)

        # self.cur_t = t
        # print(t)
        self.trueX, self.trueY, self.trueZ, self.scale_3D, self.scale_2D = self.getAbsoluteScale(frame_id)
        # self.absolute.append(self.absolute_scale)
        # print(self.scale_2D)

        if self.scale_2D > 0.1:
            self.cur_t = self.cur_t + self.scale_2D * self.cur_R.dot(t)
            # print(self.cur_t)
            self.cur_R = R.dot(self.cur_R)
            # print(self.cur_R)

        if self.px_ref.shape[0] < kMinNumFeature:
            self.px_cur = self.detector.detect(self.new_frame)
            self.px_cur = np.array([x.pt for x in self.px_cur], dtype=np.float32)
        self.px_ref = self.px_cur
        # print("feature detection = " + (str(len(self.px_cur))))

    def update(self, img, frame_id):
        assert (img.shape[0] == self.cam.height and img.shape[
            1] == self.cam.width), "Frame: provided image " \
                                   "has not the same size " \
                                   "as the camera model " \
                                   "or image is not " \
                                   "grayscale "
        self.new_frame = img
        if self.frame_stage >= 2:
            self.processFrame(frame_id)
        elif self.frame_stage == 1:
            self.processSecondFrame()
        elif self.frame_stage == 0:
            self.processFirstFrame()
        self.last_frame = self.new_frame
