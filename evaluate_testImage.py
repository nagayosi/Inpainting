import numpy as np
import pdb
import os
import glob
import pickle
import time
import cv2
import sys
import cupy as cp
from PIL import Image
from argparse import ArgumentParser

import matplotlib.pyplot as plt

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

def calcPCV1(x):
    x = cp.array(cp.where(x>pcv_thre))
    if 0 in x.shape:
        return np.array([[0,0]]).T , np.array([[0,0],[0,0]])
    #center = np.array([[256,256]]).T
    center = cp.mean(x,axis=1)[:,cp.newaxis]
    xCe = x - center
    Cov = cp.cov(xCe,bias=1)
    if True in cp.isnan(Cov):
        print("nan")
        pdb.set_trace()
    elif True in cp.isinf(Cov):
        print("inf")
        pdb.set_trace()
    V,D = cp.linalg.eig(Cov)
    vec = D[:,[cp.argmax(V)]]
    line = cp.concatenate([vec*-256,vec*256],axis=1) + center
    return cp.asnumpy(center),cp.asnumpy(line)

def nonhole(x,hole,opt=""):
    #pdb.set_trace()
    shape = x.shape
    flatt = np.reshape(x,(np.product(shape)))
    holes = np.reshape(hole,(np.product(shape)))
    tmp = []
    x,y = [],[]
    for pix,hole,ite in zip(flatt,holes,[i for i in range(flatt.shape[0])]):
        if np.sum(hole) < 1e-10:
            continue
        tmp.append(pix)
        if opt=="xy":
            x.append(ite//shape[1])
            y.append(ite%shape[0])

    if opt=="xy":
        return np.array(tmp),np.array(x),np.array(y)

    return np.array(tmp)

def clip(x,sta=-0.1,end=0.1):
    x[x<sta] = sta
    x[x>end] = end
    dist = end-sta
    res = (x-sta)/dist
    return res

def amp_mse(a,b):
    res = 8*(a-b)
    return np.sum(res**2)/dnum

if __name__ == "__main__":
    targetPath = "./data/test_sample"
    sourcePath = "./data/raw_testData.pickle"
    pcv_thre = 0.2
    pcvPath = "./data/pcv-thre{}".format(pcv_thre)
    comparePath = "./data/comparison"
    
    for DIR in [pcvPath,comparePath]:
        if not os.path.isdir(DIR):
            os.makedirs(DIR)

    source = pickle.load(open(sourcePath,"rb"))
    sAmps = source['amp']
    sImgs = sAmps/8
    sNames = [name.split(os.sep)[-1] for name in source['name']]

    tFiles = sorted(glob.glob(targetPath+os.sep+"*.png"))
    tNames = [name.split(os.sep)[-1] for name in tFiles]
    if sNames!=tNames: #データに不備がないか確認
        #pdb.set_trace()
        tNames = sNames
        newtFiles = []
        for t in tNames:
            for tf in tFiles:
                if t in tf:
                    newtFiles.append(tf)
                    break
        tFiles = newtFiles
        
    tImgs = np.array([cv2.imread(img,0) for img in tFiles])/255

    seaImg = np.array(Image.open("./data/SeismicCoefficient/KrigingSea.png"))
    sea_rgb = np.tile(seaImg[:,:,np.newaxis],[1,1,3])
    maskImg = np.array(Image.open("./data/SeismicCoefficient/KrigingMask.png"))
    mask_rgb = np.tile(maskImg[:,:,np.newaxis],[1,1,3])
    dnum = np.where(seaImg>0)[0].shape[0]

    mses = []
    errors, shift_errors  = [],[]
    maes, mses,shift_maes = [],[],[]
    centers, lines = [], []
    # 範囲ごとのMAE
    mae0, mae05_2, maes_sep = [],[],[]
    cm_bwr = plt.get_cmap("bwr")

    for ite,timg,simg,name in zip(range(len(tNames)),tImgs,sImgs,tNames):
        print("ite:{}".format(ite+1))
        err = timg-simg
        mse_grand = amp_mse(timg,simg)
        mses.append(mse_grand)

        # ==================================================================
        pdb.set_trace()
        x1 = cmap(maskImg*simg)
        x1[mask_rgb==0] = 255
        xs = [x1,cmap(timg),cmap(simg)]
        titles = ["masked","pred(MSE={0:.4f})".format(mse_grand),"original"]

        _, axes = plt.subplots(3, 3, figsize=(14, 15))
        for i,x in enumerate(xs):
            axes[0,i].imshow(x,vmin=0,vmax=255)
            axes[0,i].set_title(titles[i])

        # hisogram
        bins = 20
        hs = []
        hs.append(nonhole(simg,maskImg*seaImg))
        hs.append(nonhole(timg,seaImg))
        hs.append(nonhole(simg,seaImg))
        tmp = np.concatenate(hs,axis=0)
        maxs = np.max(tmp)

        for i,h in enumerate(hs):
            axes[1,i].hist(h,bins=bins,range=(0,maxs))
        
        
        # 各震度値ごとのMAE(テキスト) 
        e0 = rangeError(timg,simg,domain=[-1.0,0.0])
        e05_2 = rangeError(timg,simg,domain=[0.5,2.0])
        sep_errs = [rangeError(timg,simg,domain=[i*0.8,(i+1)*0.8]) for i in range(10)]
        mae0.append(e0)
        mae05_2.append(e05_2)
        maes_sep.append(sep_errs)
        
        
        axes[2,0].text(0.2,0.05,"mae0:{0:.4f}".format(e0))
        axes[2,0].text(0.2,0.125,"mae0.5_2:{0:.4f}".format(e05_2))
        for i,er in enumerate(sep_errs):
            axes[2,0].text(0.2,0.2+i*0.075,"mae{0:.1f}_{1:.1f}:{2:.4f}".format(0.1*i,0.1*(i+1),er))

        axes[2,-1].plot(np.array([(i+1)*0.1 for i in range(10)]),sep_errs)

        # AE map
        err = cm_bwr(clip(err,-0.1,0.1))[:,:,:3]
        axes[2,1].imshow(err*sea_rgb,vmin=0,vmax=1.0)

        plt.savefig(os.path.join(comparePath,name))
        plt.close()
        # ==================================================================

    print("MSE={}".format(np.mean(mses)))
