import numpy as np
import cv2
import Moildev
from visual_odometry import PinholeCamera, VisualOdometry
import os
import time


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
        self.start = None
        self.end = None
        self.predict_x = []
        self.predict_y = []
        self.predict_z = []
        self.truepose_x = []
        self.truepose_y = []
        self.truepose_z = []

        self.main()

    def dataset(self, type_dataset):
        if type_dataset == 1:
            self.image_folder = '/home/aji/Documents/Dataset/data_odometry_gray/dataset/sequences/00/image_0/'
            self.poses = '/home/aji/Documents/Dataset/data_odometry_poses/dataset/poses/00.txt'
            parameter = PinholeCamera(1241.0, 376.0, 718.8560, 718.8560, 607.1928, 185.2157)
            self.visual_odometry = VisualOdometry(parameter, self.poses)
            self.img_id = 0
            self.first_img = 1

        if type_dataset == 3:
            self.image_folder = '/home/aji/Documents/My/Github/visual-odometry/Code/processed_data/frames/'
            self.poses = '/home/aji/Documents/Dataset/data_odometry_poses/dataset/poses/00.txt'
            parameter = PinholeCamera(1280.0, 800.0, 964.828979, 964.828979, 643.788025, 484.407990)
            self.visual_odometry = VisualOdometry(parameter, self.poses)
            self.img_id = 1

        elif type_dataset == 2:
            self.image_folder = '/home/aji/Documents/Dataset/moildataset/D16-straightline/Left/'
            # self.image_folder = "/home/aji/Documents/Dataset/moildataset/round-dataset/Left/"
            self.poses = "/home/aji/Documents/Dataset/moildataset/D16-straightline/Poses/D16.txt"
            # self.poses = "/home/aji/Documents/Dataset/moildataset/round-dataset/pose/00.txt"
            parameter = PinholeCamera(848, 800, 616.8560, 293.032, 427, 394)
            self.visual_odometry = VisualOdometry(parameter, self.poses)
            self.moildev = Moildev.Moildev("Intel-T265_L.json")
            self.img_id = 1
            # self.type_dataset = 2
        else:
            print("please select the correct dataset")
        return parameter, self.visual_odometry, self.image_folder

    def main(self):
        self.start = time.time()
        self.count_file = len([name for name in os.listdir(self.image)]) - self.first_img  # if your data start by 0 (-1) else 0
        print("totals image is " + str(self.count_file))
        while self.img_id <= self.count_file:
            if self.type_dataset == 1:  # Kitti dataset
                self.img_name = self.image + str(self.img_id).zfill(6) + '.png'
                print(self.img_name)
                self.img = cv2.imread(self.img_name, 0)

            elif self.type_dataset == 2:  # fisheye img
                self.img_name = (self.image + str(self.img_id) + '.png')
                print(self.img_name)
                self.img = cv2.imread(self.img_name, 0)
                self.img = self.create_anypoint(self.img)

            elif self.type_dataset == 3:
                self.img = cv2.imread(self.image + str(self.img_id).zfill(6) + '.png', 0)

            # update frame by frame
            self.visual_odometry.update(self.img, self.pose_id)
            translation = self.visual_odometry.translation
            true_x, true_y, true_z = self.visual_odometry.trueX, self.visual_odometry.trueY, self.visual_odometry.trueZ
            if self.img_id > 2:
                x, y, z = translation[0], translation[1], translation[2]
            else:
                x, y, z = true_x, true_y, true_z
            draw_x, draw_y, draw_z = (x, y, z)

            self.predict_x.append(draw_x)


            # print(draw_x, draw_y, draw_z)
            print(self.predict_x)
            # print(true_x, true_y, true_z)
            self.drawing(draw_x, draw_y, draw_z, true_x, true_y, true_z)

        self.end = time.time()
        self.time_final = self.time_calculate(self.end, self.start)
        cv2.putText(self.traj, self.time_final, (20, 790), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)
        cv2.imwrite('result ' + self.visual_odometry.text + ".png", self.traj)

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
            cv2.circle(self.traj, (int(draw_x) + 290, int(draw_z) + 150), 1, (0, 0, 255), 1)  # odometry
            cv2.circle(self.traj, (int(true_x) + 290, int(true_z) + 150), 1, (255, 255, 255), 1)  # original

        elif self.type_dataset == 2:
            scale = 5  # scale your image trajectory
            draw_x, draw_y, draw_z = draw_x, draw_y * -1, draw_z * -1
            cv2.circle(self.traj, (int(draw_x * scale) + 290, int(draw_z * scale) + 290), 1, (0, 0, 255), 1)  # odometry
            cv2.circle(self.traj, (int(true_x * scale) + 290, int(true_z * scale) + 290), 1, (255, 255, 255),
                       1)  # original

        elif self.type_dataset == 3:
            draw_x, draw_y, draw_z = draw_x, draw_y, draw_z * -1
            cv2.circle(self.traj, (int(draw_x) + 290, int(draw_z) + 290), 1, (0, 0, 255), 1)  # odometry
            # cv2.circle(self.traj, (int(true_x) + 290, int(true_z) + 290), 1, (255, 255, 255), 1)  # original

        cv2.rectangle(self.traj, (0, 0), (600, 60), (0, 0, 0), -1)
        text_draw = "Result Coordinates: x=%2fm y=%2fm z=%2fm" % (draw_x, draw_y, draw_z)
        text_real = "Origin Coordinates: x=%2fm y=%2fm z=%2fm" % (true_x, true_y, true_z)
        # if self.img_id >= 4:
        #     calculate_x = (draw_x - true_x) / abs(true_x)
        #     print(draw_x, true_x, (draw_x - true_x), calculate_x)
        cv2.putText(self.traj, "Visual Odometry", (200, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1, 8)
        cv2.putText(self.traj, text_draw, (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, 8)
        cv2.putText(self.traj, text_real, (20, 55), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)
        cv2.rectangle(self.traj, (0, 750), (800, 800), (0, 0, 255), -1)

        cv2.putText(self.traj, ("method = " + self.visual_odometry.text), (10, 790), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)

        # print(draw_x, draw_y, draw_z)

        cv2.imshow('Image', self.img)
        cv2.imshow('trajectory', self.traj)
        cv2.waitKey(1)
        self.img_id += 1
        self.pose_id += 1

    def create_anypoint(self, image):
        mapsX, mapsY = self.moildev.getAnypointMaps(0, 0, 3.8, mode=2)
        image = cv2.remap(image, mapsX, mapsY, cv2.INTER_CUBIC, cv2.BORDER_CONSTANT)
        return image

    # def calculate_error(self, img_cound, true_x, true_y, true_z, draw_x, draw_y, draw_z):
    #     try:
    #         errorx =
    #     except ZeroDivisionError:
    #         print("error")
    #         pass
    #     return errorx, errory, errorz

# Controller()

if __name__ == '__main__':
    Controller()
