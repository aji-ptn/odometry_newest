import numpy as np
import cv2
import Moildev
from visual_odometry import Intrinsic_parameters, VisualOdometry
import os
import time
import pandas as pd
import csv


class Controller():
    def __init__(self):
        # super().__init__(parent)
        # self.parent = parent
        self.type_dataset = 2  # 1. kitti, 2. fisheye, image only on png please change manual.
        self.detector = 1
        self.image_folder = None
        self.poses = None
        self.moildev = None
        self.img_id = None
        self.pose_id = 0
        self.first_img = 0
        self.parameters, self.visual_odometry, self.image = self.dataset(self.type_dataset)
        self.traj = np.zeros((800, 800, 3), dtype=np.uint8)
        # self.traj = np.zeros((500, 500, 3), dtype=np.uint8)
        self.start = None
        self.end = None
        self.predict_x = []
        self.predict_y = []
        self.predict_z = []
        self.truepose_x = []
        self.truepose_y = []
        self.truepose_z = []
        self.detection_point = []
        self.index = []
        self.number_img=[]

        self.columns = ["Z_predict", "Z_ori", "error"]
        # self.index = None

        self.main()

    def dataset(self, type_dataset):
        if type_dataset == 1:
            self.image_folder = '/home/aji/Documents/Dataset/data_odometry_gray/dataset/sequences/00/image_0/'
            # self.image_folder = '/home/aji/Documents/Dataset/data_odometry_gray/dataset/sequences/02/image_0/'
            self.poses = '/home/aji/Documents/Dataset/data_odometry_poses/dataset/poses/00.txt'
            # self.poses = '/home/aji/Documents/Dataset/data_odometry_poses/dataset/poses/02.txt'
            parameter = Intrinsic_parameters(1241.0, 376.0, 718.8560, 718.8560, 607.1928, 185.2157)
            self.visual_odometry = VisualOdometry(parameter, self.poses)
            self.img_id = 0
            self.first_img = 1

        elif type_dataset == 2:  # Fisheye image (Realsense Camera)
            self.image_folder = '/home/aji/Documents/Dataset/moildataset/D16-straightline/Left/'
            # self.image_folder = "/home/aji/Documents/Dataset/moildataset/round-dataset/Left/"

            # self.image_folder = '/home/aji/Documents/Dataset/data_moil_0/dataset_outdoor/intelL/'
            # self.image_folder = '/home/aji/Documents/Dataset/data_moil_0/dataset_outdoor/intelR/'

            # self.image_folder = '/home/aji/Documents/Dataset/data_moil_0/dataset_stright/intelL/'
            # self.image_folder = '/home/aji/Documents/Dataset/data_moil_0/dataset_stright/intelR/'

            # self.image_folder = '/home/aji/Documents/Dataset/data_moil_0/20_M_outdoor/intelL/'
            # self.image_folder = '/home/aji/Documents/Dataset/data_moil_0/20_M_outdoor/intelR/'


            self.poses = "/home/aji/Documents/Dataset/moildataset/D16-straightline/Poses/D16.txt"
            # self.poses = "/home/aji/Documents/Dataset/moildataset/round-dataset/pose/00.txt"

            # self.poses = '/home/aji/Documents/Dataset/data_moil_0/dataset_outdoor/poses.txt'

            # self.poses = '/home/aji/Documents/Dataset/data_moil_0/dataset_stright/poses.txt'

            # self.poses = "/home/aji/Documents/Dataset/data_moil_0/20_M_outdoor/poses_12.txt"


            parameter = Intrinsic_parameters(848, 800, 616.8560, 293.032, 427, 394)  # left
            # parameter = Intrinsic_parameters(848, 800, 616.8560, 293.032, 431, 397)  # right
            self.visual_odometry = VisualOdometry(parameter, self.poses)
            self.moildev = Moildev.Moildev("Intel-T265_L.json")
            self.img_id = 1
            # self.type_dataset = 2

        elif type_dataset == 3:  # WEB-Camera
            self.image_folder = '/home/aji/Documents/Dataset/data_moil_0/dataset_outdoor/webcam/'
            # self.image_folder = '/home/aji/Documents/Dataset/data_moil_0/dataset_stright/webcam/'

            # self.image_folder = '/home/aji/Documents/Dataset/data_moil_0/20_M_outdoor/webcam/'

            self.poses = "/home/aji/Documents/Dataset/data_moil_0/dataset_outdoor/poses.txt"
            # self.poses = "/home/aji/Documents/Dataset/data_moil_0/dataset_stright/poses.txt"

            # self.poses = "/home/aji/Documents/Dataset/data_moil_0/20_M_outdoor/poses_12.txt"

            self.parameter = "C.npz"
            parameter = Intrinsic_parameters(640.0, 480.0, 652.527,  651.7243, 316.6353, 240.90377)
            # parameter = Intrinsic_parameters(619.0, 463.0, 652.527, 651.7243, 316.6353, 240.90377)
            self.visual_odometry = VisualOdometry(parameter, self.poses)
            self.img_id = 1

        elif type_dataset == 4:  # pi-camera
            self.image_folder = '/home/aji/Documents/Dataset/data_moil_0/dataset_outdoor/pi/'
            # self.image_folder = '/home/aji/Documents/Dataset/data_moil_0/dataset_stright/pi/'

            self.poses = '/home/aji/Documents/Dataset/data_moil_0/dataset_outdoor/poses.txt'
            # self.poses = '/home/aji/Documents/Dataset/data_moil_0/dataset_stright/poses.txt'
            parameter = Intrinsic_parameters(2592.0, 1944.0, 652.527, 651.7243, 1298, 966)
            self.visual_odometry = VisualOdometry(parameter, self.poses)
            self.moildev = Moildev.Moildev("entaniya.json")
            self.img_id = 1
        else:
            print("please select the correct dataset")
        return parameter, self.visual_odometry, self.image_folder

    def main(self):
        absolute_scale=[]
        self.count_file = len(
            [name for name in os.listdir(self.image)]) - self.first_img  # if your data start by 0 (-1) else 0
        # print("totals image is " + str(self.count_file))
        self.start = time.time()
        while self.img_id <= self.count_file:
            print("===============================================================")
            if self.type_dataset == 1:  # Kitti dataset
                self.number = str(self.img_id).zfill(6)
                self.img_name = self.image + self.number + '.png'
                print(self.img_name)
                self.img = cv2.imread(self.img_name, 0)
                self.data_text = "Kitti dataset"

            elif self.type_dataset == 2:  # fisheye img Realsense
                self.number = str(self.img_id)
                self.img_name = (self.image + str(self.img_id) + '.png')
                print(self.img_name)
                self.img = cv2.imread(self.img_name, 0)
                self.img = self.create_anypoint(self.img)
                self.data_text = "Fish-Eye dataset"

            elif self.type_dataset == 4:  # fisheye img PI
                self.img_name = (self.image + str(self.img_id) + '.png')
                print(self.img_name)
                self.img = cv2.imread(self.img_name, 0)
                self.img = self.create_anypoint(self.img)
                self.data_text = "Fish-Eye dataset"

            elif self.type_dataset == 3:  # webcam
                self.number = str(self.img_id)
                self.img_name = (self.image + str(self.img_id) + '.png')
                print(self.img_name)
                self.img = self.undistortion_webcam(self.img_name, self.parameter)
                self.data_text = "Logitec WEB-Cam dataset"

            # update frame by frame
            self.visual_odometry.update(self.img, self.pose_id)
            cur_t = self.visual_odometry.cur_t
            true_x, true_y, true_z = self.visual_odometry.trueX, self.visual_odometry.trueY, self.visual_odometry.trueZ
            if self.img_id > 2:
                x, y, z = cur_t[0], cur_t[1], cur_t[2]
            else:
                x, y, z = true_x, true_y, true_z
            draw_x, draw_y, draw_z = (x, y, z)
            print("visual_odometry = " + str(draw_x), str(draw_y), str(draw_z))
            print("Original Poses = " + str(true_x), str(true_y), str(true_z))
            self.drawing(draw_x, draw_y, draw_z, true_x, true_y, true_z)

            # print("=====================================================================================")
            deviation_z = np.subtract(self.truepose_z, self.predict_z)

            error = (abs(deviation_z)/self.truepose_z) * 100
            absolute_scale.append(self.visual_odometry.absolute_scale)
            self.number_img.append(self.number + ".png")
            self.index.append(self.img_id-1)
            dict = {'name': self.index, 'image': self.number_img, 'original pose': self.truepose_z, 'odometry':
                self.predict_z, "deviation (original - odometry)": deviation_z, 'deviation': abs(deviation_z),
                    'scale': absolute_scale, 'error': error}
            df = pd.DataFrame(dict)
            # saving the dataframe
            df.to_csv('20_m/666 ' + self.data_text + self.visual_odometry.text + '.csv', index=False)
            # print("=====================================================================================")

        self.end = time.time()
        self.time_final = self.time_calculate(self.end, self.start)
        cv2.putText(self.traj, ("time: " + self.time_final + " seconds"), (20, 750), cv2.FONT_HERSHEY_PLAIN, 1,
                    (255, 255, 255), 1, 8)
        if self.type_dataset == 1:
            cv2.imwrite('kitti' + self.visual_odometry.text + ".png", self.traj)
        if self.type_dataset == 2:
            cv2.imwrite('realsense ' + self.visual_odometry.text + ".png", self.traj)
        elif self.type_dataset == 3:
            cv2.imwrite('20_m/logitec' + self.visual_odometry.text + ".png", self.traj)
        elif self.type_dataset == 4:
            cv2.imwrite('picamera/ ' + self.visual_odometry.text + " Turn-Left.png", self.traj)
        else:
            cv2.imwrite('result/ ' + self.visual_odometry.text + " Turn-Left.png", self.traj)
        # cv2.waitKey(0)

    # def err

    def calculate_MSE_error(self, predict_x, predict_y, predict_z, pose_1, pose_2, pose_3):
        self.predict_x.append(float(predict_x))
        self.predict_y.append(float(predict_y))
        self.predict_z.append(float(predict_z))

        self.truepose_x.append(pose_1)
        self.truepose_y.append(pose_2)
        self.truepose_z.append(pose_3)

        # print(self.predict_z)
        # print(self.truepose_z)

        MSE_x = np.square(np.subtract(self.truepose_x, self.predict_x)).mean()
        MSE_y = np.square(np.subtract(self.truepose_y, self.predict_y)).mean()
        MSE_z = np.square(np.subtract(self.truepose_z, self.predict_z)).mean()
        self.text_error = "MSE_Error: X-coor = %2fm Y-coor =%2fm Z-coor = %2fm " % (
            MSE_x, MSE_y, MSE_z)

    def time_calculate(self, start, end):
        time_calculate = start - end
        if time_calculate >= 60:
            minute = time_calculate // 60
            second = time_calculate % 60
            time_final = (str(int(minute)) + " : " + str(int(second)))
        else:
            minute = round(time_calculate, 2)
            time_final = (str(minute))
        return time_final

    def drawing(self, draw_x, draw_y, draw_z, true_x, true_y, true_z):
        """
        for draw trajectory in two dimension
        Args:
            draw_x: prediction pose x
            draw_y: prediction pose y
            draw_z: prediction pose z
            true_x: original pose x
            true_y: original pose y
            true_z: original pose z

        Returns:
        """
        if self.type_dataset == 1:
            # print("lalla")
            draw_y = draw_y * -1
            cv2.circle(self.traj, (int(draw_x) + 290, int(draw_z) + 150), 1, (0, 0, 255), 1)  # odometry
            cv2.circle(self.traj, (int(true_x) + 290, int(true_z) + 150), 1, (255, 255, 255), 1)  # original
            # Direction
            cv2.arrowedLine(self.traj, (700, 50), (700, 100), (0, 255, 0), 2)
            cv2.arrowedLine(self.traj, (700, 50), (750, 50), (0, 255, 0), 2)
            cv2.putText(self.traj, "Z", (680, 75), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1, 8)
            cv2.putText(self.traj, "X", (725, 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1, 8)

        elif self.type_dataset == 2:
            scale = 1  # scale your image trajectory
            draw_x, draw_y, draw_z = draw_x, draw_y * -1, draw_z * -1
            cv2.circle(self.traj, (int(true_x * scale) + 290, int(true_z * scale) + 290), 1, (255, 255, 255),
                       1)  # original
            cv2.circle(self.traj, (int(draw_x * scale) + 290, int(draw_z * scale) + 290), 1, (0, 0, 255), 1)  # odometry

            # Direction
            cv2.arrowedLine(self.traj, (750, 50), (750, 100), (0, 255, 0), 2)
            cv2.arrowedLine(self.traj, (750, 50), (700, 50), (0, 255, 0), 2)
            cv2.putText(self.traj, "Z", (760, 75), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1, 8)
            cv2.putText(self.traj, "X", (725, 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1, 8)

        elif self.type_dataset == 3:
            scale = 1
            draw_x, draw_y, draw_z = draw_x, draw_y * -1, draw_z * 1  # Z = (Sift (-1), Fast (1)) (Surf(
            cv2.circle(self.traj, (int(draw_z * scale) + 290, int(draw_x * scale) + 290), 1, (0, 0, 255), 1)  # odometry
            cv2.circle(self.traj, (int(true_x * scale) + 290, int(true_z * scale) + 290), 1, (255, 255, 255),
                       1)  # original
            # Direction
            cv2.arrowedLine(self.traj, (750, 50), (750, 100), (0, 255, 0), 2)
            cv2.arrowedLine(self.traj, (750, 50), (700, 50), (0, 255, 0), 2)
            cv2.putText(self.traj, "Z", (760, 75), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1, 8)
            cv2.putText(self.traj, "X", (725, 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1, 8)

        a = (len(self.visual_odometry.px_ref))
        self.detection_point.append(a)
        # self.detection_point = self.detection_point + a
        self.mean_detection = np.mean(self.detection_point)

        # print(round(self.mean_detection))
        if self.type_dataset == 3:
            self.calculate_MSE_error(draw_z, draw_y, draw_x, true_x, true_y, true_z)
        else:
            self.calculate_MSE_error(draw_x, draw_y, draw_z, true_x, true_y, true_z)
        cv2.rectangle(self.traj, (0, 0), (650, 60), (0, 0, 0), -1)
        text_draw = "Result Coordinates: x=%2fm y=%2fm z=%2fm" % (draw_x, draw_y, draw_z)
        text_real = "Origin Coordinates: x=%2fm y=%2fm z=%2fm" % (true_x, true_y, true_z)
        cv2.putText(self.traj, "Coordinate", (680, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1, 8)
        cv2.putText(self.traj, "Visual Odometry (" + self.data_text + ")", (200, 20), cv2.FONT_HERSHEY_PLAIN, 1,
                    (0, 255, 0), 1, 8)
        cv2.putText(self.traj, text_draw, (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, 8)
        cv2.putText(self.traj, text_real, (20, 60), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)

        cv2.rectangle(self.traj, (0, 700), (800, 800), (0, 0, 0), -1)
        cv2.putText(self.traj,
                    ("method = " + self.visual_odometry.text + ", Mean point detection = " + str(self.mean_detection)),
                    (10, 790), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)
        cv2.putText(self.traj, self.text_error, (10, 770), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255),
                    1, 8)
        if self.type_dataset == 4:
            self.img = cv2.resize(self.img, (640, 480), interpolation=cv2.INTER_AREA)
        else:
            pass
        cv2.imshow('Image', self.img)
        cv2.imshow('trajectory', self.traj)
        cv2.waitKey(1)
        self.img_id += 1
        self.pose_id += 1

    def create_anypoint(self, image):
        mapsX, mapsY = self.moildev.getAnypointMaps(0, 0, 3.8, mode=2)
        image = cv2.remap(image, mapsX, mapsY, cv2.INTER_CUBIC, cv2.BORDER_CONSTANT)
        return image

    def undistortion_webcam(self, img, param):
        parameters = np.load(param)
        mtx = parameters["mtx"]
        dist = parameters["dist"]
        img = cv2.imread(img, 0)
        h, w = img.shape[:2]
        print(img.shape)
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

        # undistort
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

        # crop the image
        x, y, w, h = roi
        dst = dst[y:y + h, x:x + w]
        print(dst.shape)
        # cv2.imwrite('calibresult.png', dst)
        return dst


if __name__ == '__main__':
    Controller()
