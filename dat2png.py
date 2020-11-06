# -*- coding: utf-8 -*-
import pandas as pd
import glob
import pdb
import numpy as np
#import seaborn as sns
import pickle
import matplotlib.pylab as plt
import cv2
import sys
import os
import random

from PIL import Image

def scale(x,s=[0,7],t=[0,1]):
    d_t = t[0]-t[1]
    d_s = s[0]-s[1]
    ratio = (d_t/d_s)
    d_s_t = t[0] - s[0]*ratio
    # s range -> t range
    res = x*ratio+d_s_ta
    return res

def cmap(x,sta=[255,0,56],end=[0,0,255]): #x:scalar , sta,end:[B,G,R]
    vec = np.array(end) - np.array(sta)
    res = sta+scale(x)*vec
    return res

def ll2ij(point,starts,dist): # point=[lon,lat] , starts=[lon_st,lat_st] , dist=[lon_dis,lat_dis]
    i = (point[0]-starts[0])/dist[0]
    j = (point[1]-starts[1])/dist[1]
    return [i,j]
    
# datファイルの一覧を取得
path = "data/SeismicCoefficient/"
files = sorted(glob.glob(path+"2013_Sv05s_LL/*.dat"))

rad = float(sys.argv[1])  # pixel
isLog = "log" in sys.argv[2] # scaling by log

df = pd.read_csv(files[0],names=["lon", "lat", "amp", "sid"])
lons_df = np.unique(df['lon'].values)
lats_df = np.unique(df['lat'].values)[::-1]
p_df = np.array([df['lon'].values,df['lat'].values]).transpose() # [M,2] 全データ

#maps = pickle.load(open(path+"maps_amp.pickle"))
excel = pd.read_excel(path+"site_schema_20200322A.xlsx")

# 観測点
# exist points
p_ex = np.array([excel['lon'].values,excel['lat'].values]) 

# interval of simulated points
dis_lon = lons_df[1]-lons_df[0]
dis_lat = lats_df[0]-lats_df[1]

# 観測値のxy軸の端
# start and end points of simulation
st = np.array([np.min(lons_df),np.min(lats_df)])
en = [np.max(lons_df),np.max(lats_df)]

# cut points over simulation range
keep_ind = []
keep_lons = np.concatenate([np.where(p_ex[0]>=st[0]-dis_lon/2)[0],np.where(p_ex[0]<=en[0]+dis_lon/2)[0]])
uni,counts_lon = np.unique(keep_lons,return_counts=True)
keep_ind.append(uni[counts_lon>=2])

keep_lats = np.concatenate([np.where(p_ex[1]>=st[1]-dis_lat/2)[0],np.where(p_ex[1]<=en[1]+dis_lat/2)[0]])
uni,counts_lat = np.unique(keep_lats,return_counts=True)
keep_ind.append(uni[counts_lat>=2])
uni,counts = np.unique(np.concatenate(keep_ind),return_counts=True)
keep_ind = uni[counts>=2]

#pdb.set_trace()
p_ex = p_ex[:,keep_ind].transpose() #[N,2] 観測所

#======================================================================
#pdb.set_trace()
if rad > 0:
    # scaling [lon_s~lon_e,lat_s~lat_e] -> [0~507,0~517]
    xy_df = (p_df - np.tile(st[np.newaxis], (p_df.shape[0],1))) / np.array([dis_lon,dis_lat])
    xy_ex = (p_ex - np.tile(st[np.newaxis], (p_ex.shape[0],1))) / np.array([dis_lon,dis_lat])

    # 観測点から半径rad の円に中心が含まれるピクセルを観測ピクセルとして採用
    n = xy_ex.shape[0]
    m = xy_df.shape[0]

    # calculate distances
    dists = np.tile(xy_df,(n,1)) - np.reshape(np.tile(xy_ex,(1,m)),[n*m,2])
    dists = np.linalg.norm(dists, ord=2, axis=1)
    p_ex = p_df[np.unique( np.where(dists<=rad)[0] % m )]

# points lon,lat ->  indexs i,j
#pdb.set_trace()
inds_ex = []
for p in p_ex:
    i = np.abs(lons_df-p[0]).argmin()
    j = np.abs(lats_df-p[1]).argmin()
    inds_ex.append([i,j])
inds_ex = np.array(inds_ex).transpose()

#======================================================================
# make mask image and sea image
mask = np.zeros([len(lats_df),len(lons_df)])
sea = np.zeros([len(lats_df),len(lons_df)])
count = 0
for ind in range(len(df)):
    # 緯度と経度の座標取得
    lonInd = int(np.where(df['lon'][ind]==lons_df)[0]) #経度
    latInd = int(np.where(df['lat'][ind]==lats_df)[0]) #緯度

    if latInd in inds_ex[1][np.where(inds_ex[0]==lonInd)[0]]: # Exist
        count += 1
        mask[latInd,lonInd] = 255
    sea[latInd,lonInd] = 1

"""
# クリギングは画像サイズ518x508で行う
krigeMask = (mask*sea).astype("uint8")
cv2.imwrite(path+"KrigingMask.png", krigeMask)
cv2.imwrite(path+"KrigingSea.png", (sea*255).astype("uint8"))
"""

mask = mask.astype("uint8")
mask = cv2.resize(mask,(512,512))
_, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

tmp_sea = cv2.threshold(cv2.resize(sea*255,(512,512)), 127, 255, cv2.THRESH_BINARY)[1]

mask[tmp_sea==0] = 255

#pdb.set_trace()

# datasetPath
d_path = os.path.join(path,"dataSet-r{}".format(rad))
if isLog:
    d_path += "-log"

TRAIN_DIR = d_path+os.sep+"train"+os.sep+"train_amp"+os.sep
VALID_DIR = d_path+os.sep+"valid"+os.sep+"valid_amp"+os.sep
TEST_DIR = d_path+os.sep+"test"+os.sep+"test_amp"+os.sep

TRAIN_MASK = d_path+os.sep+"train_mask"+os.sep
VALID_MASK = d_path+os.sep+"valid_mask"+os.sep
TEST_MASK = d_path+os.sep+"test_mask"+os.sep

TRAIN_SEA = d_path+os.sep+"train_sea"+os.sep
VALID_SEA = d_path+os.sep+"valid_sea"+os.sep
TEST_SEA = d_path+os.sep+"test_sea"+os.sep

for DIR in [d_path,TRAIN_DIR,VALID_DIR,TEST_DIR,TRAIN_MASK,VALID_MASK,TEST_MASK,TRAIN_SEA,VALID_SEA,TEST_SEA]:
    if not os.path.isdir(DIR):
        os.makedirs(DIR)

# save sea and mask
cv2.imwrite(path+"sea.png",tmp_sea)
cv2.imwrite(path + 'mask_r{}.png'.format(rad),mask)
pdb.set_trace()

# make amp-map images
maps = []
seas = []
names = []
mask_names = []
sea_names = []

# data config file
#pdb.set_trace()
dcf_path = path + "DataConfig.txt" 
datanum = len(files)
validnum = datanum//10
testnum = (datanum-validnum)//9
trainnum = datanum-validnum-testnum

if not os.path.isfile(dcf_path):
    print("make new-dataset")
    random.shuffle(files)
    with open(dcf_path, mode='w') as f:
        for ite,name in enumerate(files):
            if ite < validnum: #valid
                f.write(name+" valid\n")
            elif validnum <= ite and ite < validnum+testnum: #test
                f.write(name+" test\n")
            else: # train
                f.write(name+" train\n")
    f.close()

# データ分割情報をロード
#pdb.set_trace()
with open(dcf_path) as f:
    l_strip = [s.strip() for s in f.readlines()] # remove '\n'
    files = [st.split(" ")[0] for st in l_strip]
    dtypes = [st.split(" ")[1] for st in l_strip]
f.close()

print("train:{}, valid:{}, test:{}".format(trainnum,validnum,testnum))
# datファイルのループ
for ite,file,dtype in zip(range(len(files)),files,dtypes):
    print(" \r ite:{}".format(ite),end="")

    # datファイルの読み込み
    df = pd.read_csv(file,names=["lon", "lat", "amp", "sid"])

    # 経度と緯度の値の格子
    lons = np.unique(df['lon'].values)
    lats = np.unique(df['lat'].values)[::-1]

    # マップ画像の初期化
    map = np.zeros([len(lats),len(lons)])
    sea = np.zeros([len(lats),len(lons)])
    #pdb.set_trace()
    #count = 0
    """
    # マップ画像の座標のループ（もっと賢いやり方がないのか？）
    for ind in range(len(df)):
        #pdb.set_trace()
        # 緯度と経度の座標取得
        lonInd = int(np.where(df['lon'][ind]==lons)[0])
        latInd = int(np.where(df['lat'][ind]==lats)[0])

        map[latInd,lonInd] = df['amp'][ind]
        sea[latInd,lonInd] = 1

    """
    fname = (file.split('.')[0]).split('/')[-1]
    maps.append(map)
    seas.append(sea)
    

    if dtype=="valid": #valid
        DIR = VALID_DIR
        MASK = VALID_MASK
        SEA = VALID_SEA
    elif dtype=="test": #test
        DIR = TEST_DIR
        MASK = TEST_MASK
        SEA = TEST_SEA
    elif dtype=="train": # train
        DIR = TRAIN_DIR
        MASK = TRAIN_MASK
        SEA = TRAIN_SEA
    else:
        print("Unknown value")
        pdb.set_trace()

    names.append(DIR+fname+".png")
    mask_names.append(MASK+fname+".png")
    sea_names.append(SEA+fname+".png")
"""
#pdb.set_trace()
maps = np.array(maps)
seas = np.array(seas)

pdb.set_trace()

# scaling
scale_max = 8.0
#scale_max = np.max(maps) # max=37.2756
maps[maps>scale_max] = scale_max

if isLog:
    maps = np.log2((maps/scale_max)+1)*255
    print("log-scaling")
else:
    maps = (maps/scale_max)*255
    print("normal scaling")
"""

for map,sea,name,m_name,s_name in zip(maps,seas,names,mask_names,sea_names):
    #map = map.astype("uint8")
    #map = cv2.resize(map,(512,512))
    #cv2.imwrite(name,map)
    #sea = sea.astype("uint8")
    #sea = cv2.resize(sea,(512,512))
    #cv2.imwrite(s_name,sea)
    mask = np.array(Image.open("./data/SeismicCoefficient/mask_r0.0.png"))
    cv2.imwrite(m_name,mask)
#pdb.set_trace()



