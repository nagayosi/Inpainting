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


def rangeError(pre,tru,domain=[-1.0,0.0],opt="MA"): # pred:予測値, true:真値 , domain:変域(domain[0]< x <=domain[1])
    # normalyse
    domain = [domain[0]/8,domain[1]/8]

    inds = np.where(np.logical_and(tru>domain[0],tru<=domain[1]))

    if inds[0].shape[0]==0:
        return np.NaN

    error_ = tru[inds[0],inds[1]]-pre[inds[0],inds[1]]
    if opt=="MA": # MAE
        error_ = np.mean(np.abs(error_))
    elif opt=="MS": # MSE
        error_ = np.mean(error_**2)
    elif opt=="A":
        error_ = np.abs(error_)

    return error_


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

    def predict_normalized(self,x): # x:shape=[2]
        Psi=cp.exp(-cp.sum(self.theta*cp.power((cp.abs(self.X-x[cp.newaxis,:])),self.pl)))
        z = self.y-self.one.dot(self.mu)
        a = cp.linalg.solve(self.U.T, z)
        b = cp.linalg.solve(self.U, a)
        c = Psi.T.dot(b)

        f=self.mu + c
        return f

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



if __name__ == "__main__":
    
    # GPU上での処理が終わるのを待ってから次の行を実行
    cp.cuda.Stream.null.synchronize()

    #pdb.set_trace()
    # data dimension
    dim = 2
    samplePath = "data"+os.sep+"test_sample"
    dataPath = "data"+os.sep+"SeismicCoefficient"
    
    pred_num = 10
    exi = int(sys.argv[1]) # 実験番号

    testData = pickle.load(open("data/raw_testData.pickle","rb"))
    testImg = testData["amp"]
    # 一部を実験
    if pred_num*(exi+1)>testImg.shape[0]:
        testImg = testImg[pred_num*exi:]
    else:
        testImg = testImg[pred_num*exi:pred_num*(exi+1)]
    testImg[testImg>8] = 8
    testImg = testImg/8
    testNames = testData["name"]
    maskImg = np.array(Image.open(os.path.join(dataPath,"KrigingMask.png")))
    seaImg = np.array(Image.open(os.path.join(dataPath,"KrigingSea.png")))
    inds = np.where(maskImg*seaImg>0)
    inds = [inds[0][np.newaxis],inds[1][np.newaxis]]

    X = np.concatenate(inds,axis=0).T.astype("float32") #入力点 shape=[N,2]
    
    # 補間したい点を集める
    target_mask = seaImg
    tPoint = np.where(target_mask>0) # 補間したい点
    tPoint = np.concatenate([tPoint[0][np.newaxis],tPoint[1][np.newaxis]]).T.astype("float32") # shape=[M,2]
    pnum = int(np.sum(target_mask/255)) # 1枚の画像内にある予測しないといけない点の数
    
    startFlag = True

    pred_imgs= []

    # 範囲ごとのMAEを保存するリスト
    maes, mae0, mae05_2, maes_sep = [],[],[],[]
    for ite,img in enumerate(testImg):
        print("{} of {} images".format(ite+1,testImg.shape[0]))
        y = img[maskImg*seaImg>0]
        
        #pdb.set_trace()
        stime = time.time()
        k = kriging(X, y, name='InpaintingKriging')
        k.train()
        train_time = time.time() - stime
        print("{}sec for training".format(train_time),end=" ")

        pdb.set_trace()
        Prediction = CuPyPredicter(k)

        def predict(xs):
            res = np.zeros_like(img)
            for iterate,p in enumerate(xs):
                print("\r predict ite:{}-point:{} ".format(ite+1,iterate),end="")
                i = int(p[0])
                j = int(p[1])
                res[i,j] = Prediction.predict([p[0],p[1]])
            return res

        # predict
        stime = time.time()
        pred = predict(tPoint)
        print("{}sec for predict".format(time.time()-stime))
        pred_imgs.append(pred)

        # 補間結果をプロットして保存
        tmp = pred*np.ones_like(pred) # 参照渡しを防ぐ
        tmp[tmp>1] = 1
        tmp[tmp<0] = 0
        testname = testNames[ite].split("/")[-1]
        cv2.imwrite(os.path.join(samplePath,testname),tmp*255)
    
        # 各震度値ごとのMAE(テキスト) 
        #pdb.set_trace()
        mae = np.sum(np.abs(img-pred))/pnum
        e0 = rangeError(pred,img,domain=[-1.0,0.0])
        e05_2 = rangeError(pred,img,domain=[0.5,2.0])
        sep_errs = [rangeError(pred,img,domain=[i*0.8,(i+1)*0.8]) for i in range(10)]

        maes.append(mae)
        mae0.append(e0)
        mae05_2.append(e05_2)
        maes_sep.append(sep_errs)

    #pdb.set_trace()
    summary_data = {
        "pred":np.array(pred_imgs),
        "name":testNames,
        "MAE":np.mean(np.array(maes)),
        "MAE0":np.mean(np.array(mae0)),
        "MAE-sep0.8":np.array(maes_sep),
        "MAE0.5~2":np.mean(np.array(mae05_2))
    }

    print("MAE={0:.8f}, MAE0.5~2={1:.8f}".format(summary_data["MAE"],summary_data["MAE0.5~2"]))

    pkl_path = os.path.join("data","kriging_result{}.pickle".format(exi))
    with open(pkl_path,"wb") as f:
        pickle.dump(summary_data,f)




