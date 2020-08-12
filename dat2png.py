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

def scale(x,s=[0,7],t=[0,1]):
    d_t = t[0]-t[1]
    d_s = s[0]-s[1]
    ratio = (d_t/d_s)
    d_s_t = t[0] - s[0]*ratio
    # s range -> t range
    res = x*ratio+d_s_t

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

df = pd.read_csv(files[0],names=["lon", "lat", "amp", "sid"])
lons_df = np.unique(df['lon'].values)
lats_df = np.unique(df['lat'].values)[::-1]

#maps = pickle.load(open(path+"maps_amp.pickle"))

excel = pd.read_excel(path+"site_schema_20200322A.xlsx")

# exist points
p_ex = np.array([excel['lon'].values,excel['lat'].values]) 

# interval of simulated points
dis_lon = lons_df[1]-lons_df[0]
dis_lat = lats_df[0]-lats_df[1]

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

keep_ind = np.unique(np.concatenate(keep_ind))

p_ex = p_ex[:,keep_ind].transpose() #[N,2]


p_df = np.array([df['lon'].values,df['lat'].values]).transpose() # [M,2]

rad = float(sys.argv[1])  # pixel

#pdb.set_trace()
# scaling [lon_s~lon_e,lat_s~lat_e] -> [0~507,0~517]
xy_df = (p_df - np.tile(st[np.newaxis], (p_df.shape[0],1)))/np.array([dis_lon,dis_lat])
xy_ex = (p_ex - np.tile(st[np.newaxis], (p_ex.shape[0],1)))/np.array([dis_lon,dis_lat])

n = xy_ex.shape[0]
m = xy_df.shape[0]

# calculate distances
dists = np.tile(xy_df,(n,1)) - np.reshape(np.tile(xy_ex,(1,m)),[n*m,2])
dists = np.linalg.norm(dists, ord=2, axis=1)

p_ex = p_df[np.unique( np.where(dists<=rad)[0] % m )]


# points lon,lat ->  indexs i,j
inds_ex = []
for p in p_ex:
    i = np.abs(lons_df-p[0]).argmin()
    j = np.abs(lats_df-p[1]).argmin()
    inds_ex.append([i,j])
#pdb.set_trace()
inds_ex = np.array(inds_ex).transpose()

"""
# expand inds_ex (L1)
new_inds = []
rad = int(sys.argv[1])

for p in inds_ex:
    for rx in range(rad+1):
        for ry in range(rad+1):
            if rx+ry <= rad:
                new_inds.append(np.array([p[0]+rx,p[1]+ry]))
                new_inds.append(np.array([p[0]-rx,p[1]+ry]))
                new_inds.append(np.array([p[0]-rx,p[1]-ry]))
                new_inds.append(np.array([p[0]+rx,p[1]-ry]))
"""
#inds_ex = np.unique(new_inds,axis=0).transpose()


# make mask image
mask = np.ones([len(lats_df),len(lons_df),3])*255

count = 0
for ind in range(len(df)):
    #pdb.set_trace()
    # 緯度と経度の座標取得
    lonInd = int(np.where(df['lon'][ind]==lons_df)[0])
    latInd = int(np.where(df['lat'][ind]==lats_df)[0])

    if not (latInd in inds_ex[1][np.where(inds_ex[0]==lonInd)[0]]): # not Exist
        count += 1
        mask[latInd,lonInd] = np.array([0,0,0])
pdb.set_trace()
mask = mask.astype("uint8")
cv2.imwrite(path+"mask_L2-rad{}.png".format(rad),mask)

SAVE_DIR = path + "amp_map"+os.sep
if not os.path.isdir(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# make amp-map images
maps = []
names = []
# datファイルのループ
for file in files:
    print("loading {}".format(file))
    #pdb.set_trace()

    # datファイルの読み込み
    df = pd.read_csv(file,names=["lon", "lat", "amp", "sid"])

    # 経度と緯度の値の格子
    lons = np.unique(df['lon'].values)
    lats = np.unique(df['lat'].values)[::-1]

    # マップ画像の初期化
    map = np.ones([len(lats),len(lons),3])*255
    #pdb.set_trace()
    #count = 0
    
    # マップ画像の座標のループ（もっと賢いやり方がないのか？）
    for ind in range(len(df)):
        #pdb.set_trace()
        # 緯度と経度の座標取得
        lonInd = int(np.where(df['lon'][ind]==lons)[0])
        latInd = int(np.where(df['lat'][ind]==lats)[0])
        """
        if latInd in inds_ex[1][np.where(inds_ex[0]==lonInd)[0]]: # Exist
            # 座標のピクセルに振幅を設定
            map[latInd,lonInd] = cmap(df['amp'][ind])
        else:
            count += 1
            map[latInd,lonInd] = np.array([0,0,0])
        """
        map[latInd,lonInd] = cmap(df['amp'][ind])


    #print("count:{}".format(count))
    map = map.astype("uint8")
    maps.append(map)
    names.append((file.split('.')[0]).split('/')[-1])

    
    # ヒートマップでマップ画像を表示
    #xticklabels=[round(lons[i],2) if i%100==0 else '' for i in range(len(lons))]
    #yticklabels=[round(lats[i],2) if i%100==0 else '' for i in range(len(lats))]
    #sns.heatmap(map,xticklabels=xticklabels,yticklabels=yticklabels,vmin=0,vmax=7)
    
    #pdb.set_trace()
    fname = (file.split('.')[0]).split('/')[-1]
    #plt.savefig(f"{fname}.png")
    #plt.close()
    #pdb.set_trace()
    cv2.imwrite(SAVE_DIR+fname+".png",map)

#pickle.dump([np.array(maps),np.array(masks),names],open(SAVE_DIR+"maps_amp_L1-rad{}.pickle".format(rad),"wb"))
#pickle.dump(names,open(SAVE_DIR+"names_L1-rad{}.pickle".format(rad),"wb"))


