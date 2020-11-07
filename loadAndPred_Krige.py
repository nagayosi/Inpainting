import numpy as np
import pdb
import os
import glob
import pickle
import time
import cv2
import sys
import cupy as cp

import pyKriging
from pyKriging.krige import kriging  
from pyKriging.samplingplan import samplingplan

from PIL import Image
from test_pyKriging import CuPyPredicter

from argparse import ArgumentParser


def cmap(x,sta=[222,222,222],end=[255,0,0]): #x:gray-image([w,h]) , sta,end:[B,G,R]
    vec = np.array(end) - np.array(sta)
    res = []
    for i in range(x.shape[0]):
        tmp = []
        for j in range(x.shape[1]):
            tmp.append(np.array(sta)+x[i,j]*vec)
        res.append(tmp)
    res = np.array(res).astype("uint8")
    res[sea_rgb==0] = 255
    return res

def parse_args():
    parser = ArgumentParser(description="学習済みのパラメータでテストをし、真値との比較や分析結果の保存を行います")
    parser.add_argument('exi_list',type=lambda x:list(map(str, x.split(','))),help="実験番号(0-0,0-1,...,6-4,7-0,7-1まで)のリスト（例）0-1,4-2,2-3")
    return  parser.parse_args()


class CuPyPredicter:
    def __init__(self,model):
        # Numpy -> Cupy
        #pdb.set_trace()
        self.n = cp.asarray(model.n)
        self.mu = cp.asarray(model.mu)
        self.theta = cp.asarray(model.theta)
        self.pl = cp.asarray(model.pl)
        self.U = cp.asarray(model.U)
        self.one = cp.asarray(model.one)
        self.y = cp.asarray(model.y) # [n]
        self.X = cp.asarray(model.X) # [n,2]
        self.normRange = cp.asarray(model.normRange)
        self.ynormRange = cp.asarray(model.ynormRange)
        z = self.y-self.one.dot(self.mu)
        aaa = cp.linalg.solve(self.U.T, z)
        self.bbb = cp.linalg.solve(self.U, aaa)

    def predict_normalized(self,x): # x:shape=[2]
        Psi=cp.exp(-cp.sum(self.theta*cp.power((cp.abs(self.X-x[cp.newaxis,:])),self.pl),axis=1))
        ccc = Psi.T.dot(self.bbb)
        fff = self.mu + ccc
        return fff 
    
    def predict_row(self,x):# x.shape=[N,2]
        #pdb.set_trace()
        x = self.normX(cp.asarray(x))
        dist = cp.tile(self.X,[x.shape[0], 1]) - cp.reshape(cp.tile(x,[1,self.X.shape[0]]),[self.X.shape[0]*x.shape[0], 2])
        Psi = cp.reshape(cp.exp(-cp.sum(self.theta*cp.power(cp.abs(dist),self.pl),axis=1)),[x.shape[0],self.X.shape[0]]) # 次元方向に和
        ccc = Psi.dot(self.bbb)
        fff = ccc + self.mu
        return cp.asnumpy(self.inversenormy(fff))

    def predict(self,X): # X:shape=[2] 出力はNumpy
        X = self.normX(cp.asarray(X))
        return cp.asnumpy(self.inversenormy(self.predict_normalized(X)))

    def normX(self, X):
        '''
        :param X: An array of points (self.k long) in physical world units
        :return X: An array normed to our model range of [0,1] for each dimension
        '''
        return cp.array( (X - self.normRange[0,0]) / float(self.normRange[0,1] - self.normRange[0,0]) )

    def inversenormy(self, y):
        '''
        :param y: A normalized array of model units in the range of [0,1]
        :return: An array of observed values in real-world units
        '''
        return (y * (self.ynormRange[1] - self.ynormRange[0])) + self.ynormRange[0]


if __name__=="__main__":

    args = parse_args()
    exi_list = args.exi_list

    # GPU上での処理が終わるのを待ってから次の行を実行
    cp.cuda.Stream.null.synchronize()

    # data dimension
    dim = 2
    samplePath = "data"+os.sep+"test_sample"
    dataPath = "data"+os.sep+"SeismicCoefficient"
    pred_num = 5

    exi = [int(ex.split('-')[0]) for ex in args.exi_list] # 実験番号(0~7)
    exj = [int(ex.split('-')[1]) for ex in args.exi_list] # データ番号(0~4)

    testData = pickle.load(open("data/raw_testData.pickle","rb"))
    testImg = testData["amp"]
    # 指定されたデータに関して実験
    testInd = [i*pred_num + j for i,j in zip(exi,exj)]
    #testImg = np.array([testImg[ite] for ite in testInd])
    #testImg[testImg>8] = 8
    #testImg = testImg/8
    testNames = testData["name"]
    testNames = [testNames[ite].split('/')[-1] for ite in testInd]

    maskImg = np.array(Image.open(os.path.join(dataPath,"KrigingMask.png")))
    seaImg = np.array(Image.open(os.path.join(dataPath,"KrigingSea.png")))
    sea_rgb = np.tile(seaImg[:,:,np.newaxis],[1,1,3]) # important
    # inds = np.where(maskImg*seaImg>0)
    # inds = [inds[0][np.newaxis],inds[1][np.newaxis]]

    # 補間したい点を集める
    target_mask = seaImg
    tPoint = np.where(target_mask>0) # 補間したい点
    tPoint = np.concatenate([tPoint[0][np.newaxis],tPoint[1][np.newaxis]]).T.astype("float32") # shape=[M,2]
    pnum = int(np.sum(target_mask/255)) # 1枚の画像内にある予測しないといけない点の数
    
    for iteration,exp_id in enumerate(exi_list):
        print("ite{}".format(iteration))
        k = pickle.load(open("krigingmodel{}".format(exp_id),"rb"))
        Prediction = CuPyPredicter(k)

        def predict(xs):
            batch = 1000
            res = np.zeros_like(seaImg).astype("float32")
            rangemax = 0
            dnum = xs.shape[0]
            for ite in range( (dnum//batch) +1):
                rangemax = batch*(ite+1)
                if rangemax > dnum:
                    rangemax = dnum

                #pdb.set_trace()
                x = xs[batch*ite:rangemax]

                pred_row = Prediction.predict_row(x)
                for px,py in zip(x,pred_row):
                    res[int(px[0]),int(px[1])] = py
                print("\r predict point:{} ".format(rangemax),end="")
            return res

        #pdb.set_trace()
        img = predict(tPoint)
        #pdb.set_trace()
        img[img>1] = 1
        img[img<0] = 0
        print("{}".format(testNames[iteration]))
        cv2.imwrite("./data/test_sample/{}".format(testNames[iteration]),img*255)
