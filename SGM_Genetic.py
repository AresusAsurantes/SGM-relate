from sko.GA import GA
import numpy as np
import cv2 as cv
import pandas as pd
import matplotlib.pyplot as plt

class SGM_Genetic:
    def __init__(self, left, right, GT, size_pop=50, metric="mse"):

        self.count = 1
        self.generation = 1
        self.left = left
        self.right = right
        self.GT = GT.astype(np.float32)
        self.parameter = [0, 160, 10, 150, 3, 50]
        self.Y = []
        self.size_pop = size_pop
        self.metric = metric

        if metric == "mse" or metric == "MSE":
            fun = self.sgmMSE
        elif metric == "bpp" or metric == "BPP":
            fun = self.sgmBPP

        # Genetic Algorithm
        # [minDisparity, numDisparity, p1, p2, blocksize, speckleWindowSize]
        self.ga = GA(func=fun, n_dim=6, size_pop=self.size_pop, prob_mut=0.01,
                     lb=[0, 160, 10, 150, 3, 50], ub=[50, 480, 100, 500, 11, 200], precision=1e-7)

    def run(self, iter=200):
        self.parameter, self.Y = self.ga.run(iter)
        return self.parameter

    def showResult(self):
        mindisp, numdisp, p1, p2, blocksize, windowsize = self.parameter.astype(int)

        print(" minDisparity:", mindisp, "\n",
              "numDisparity:", numdisp, "\n",
              "P1:", p1, "\n",
              "P2:", p2, "\n",
              "blocksize:", blocksize, "\n",
              "speckleWindowSize", windowsize)

        # visualize process
        Y_history = pd.DataFrame(self.ga.all_history_Y)
        fig, ax = plt.subplots(2, 1)
        ax[0].plot(Y_history.index, Y_history.values, '.', color='red')
        Y_history.min(axis=1).cummin().plot(kind='line')
        plt.show()

    def sgm(self, parameter):

        # set parameter
        mindisp, numdisp, p1, p2, blocksize, windowsize = parameter.astype(int)

        # create matcher
        left_matcher = cv.StereoSGBM_create(minDisparity=mindisp, numDisparities=numdisp,
                                            P1=p1, P2=p2,
                                            blockSize=blocksize, speckleWindowSize=windowsize)
        right_matcher = cv.ximgproc.createRightMatcher(left_matcher)

        # create filter
        wls_filter = cv.ximgproc.createDisparityWLSFilter(left_matcher)
        wls_filter.setSigmaColor(1.5)
        wls_filter.setLambda(8000)

        # use filter get disparity
        left_disp = left_matcher.compute(self.left, self.right)
        right_disp = right_matcher.compute(self.right, self.left)
        disparity = wls_filter.filter(disparity_map_left=left_disp, left_view=self.left,
                                           filtered_disparity_map=None,
                                           disparity_map_right=right_disp, right_view=self.right)
        disparity = cv.ximgproc.getDisparityVis(disparity).astype(np.float32)
        roi = wls_filter.getROI()

        return [disparity, roi]

    def sgmMSE(self, parameter):

        if (self.count > self.size_pop):
            self.count = 1
            self.generation += 1

        disparity, roi = self.sgm(parameter)

        err = cv.ximgproc.computeMSE(self.GT, disparity, roi)
        print("第", self.generation, "代，个体编号：", self.count, " 基因：", err)

        self.count += 1

        return err

    def sgmBPP(self, parameter):

        if (self.count > self.size_pop):
            self.count = 1
            self.generation += 1

        disparity, roi = self.sgm(parameter)

        err = cv.ximgproc.computeBadPixelPercent(self.GT, disparity, roi)
        print("第", self.generation, "代，个体编号：", self.count, " 基因：", err, "%")

        self.count += 1

        return err

    def compute(self):
        self.disparity, roi = self.sgm(self.parameter)
        return self.disparity

if __name__ == '__main__':

    left = cv.imread("/Users/aresus/Downloads/stereo_train_001/camera_5/171206_034625454_Camera_5.jpg", cv.IMREAD_GRAYSCALE)
    right = cv.imread("/Users/aresus/Downloads/stereo_train_001/camera_6/171206_034625454_Camera_6.jpg", cv.IMREAD_GRAYSCALE)
    ground_truth = cv.imread("/Users/aresus/Downloads/stereo_train_001/disparity/171206_034625454_Camera_5.png", cv.IMREAD_GRAYSCALE)
    ground_truth = ground_truth.astype(np.float32)

    sgm_gen = SGM_Genetic(left, right, ground_truth, 4)

    sgm_gen.run(1)

    # sgm_gen.showResult()

    disparity = sgm_gen.compute()

    cv.imwrite("/Users/aresus/Downloads/res.png",disparity)

    print("success")