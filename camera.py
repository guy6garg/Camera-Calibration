# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.linalg as linalg
import math as math
import os
from PIL import Image

# Functions for Normalizing
def normalise2d(img):
   # non homogenous image i.e (xi,yi ) pairs
    avg_distance=1/len(img) *np.sum(np.sqrt((np.sum(abs(img-np.mean(img,axis=0))**2,axis=1))))
    s=np.sqrt(2)/avg_distance
    m=np.mean(img,0)
    tr=np.array([[s,0,-s*m[0]],[0,s,-s*m[1]],[0,0,1]])
    tr=tr.astype(float)
    normalized_points=[]
    for i in range(len(img)):
        temp=img[i]
        temp=np.expand_dims(temp,axis=1)
        conc=np.expand_dims(np.array([1]),axis=1)
        temp=np.concatenate((temp,conc),axis=0)
        transformed=np.dot(tr,temp)
        normalized_points.append([float(transformed[0]),float(transformed[1]),1])
    return tr,np.array(normalized_points)
def normalise3d(img):
    avg_distance=1/len(img) *np.sum(np.sqrt((np.sum(abs(img-np.mean(img,axis=0))**2,axis=1))))
    s=np.sqrt(3)/avg_distance
    m=np.mean(img,0)
    #print(m)
    tr=np.array([[s,0,0,-s*m[0]],[0,s,0,-s*m[1]],[0,0,s,-s*m[2]],[0,0,0,1]])
    normalized_points=[]
    for i in range(len(img)):
        temp=img[i]
        temp=np.expand_dims(temp,axis=1)
        conc=np.expand_dims(np.array([1]),axis=1)
        temp=np.concatenate((temp,conc),axis=0)
        transformed=np.dot(tr,temp)
        normalized_points.append([float(transformed[0]),float(transformed[1]),float(transformed[2]),1])
    return tr,np.array(normalized_points)

# Arrays containing parameters for multiple images and points
alpha_arr=[]
beta_arr=[]
x0_arr=[]
y0_arr=[]
R_arr=[]
t_arr=[]
theta_arr=[]
rmse_arr=[]
K_arr=[]
actual_points_arr=[]
predicted_points_arr=[]

# the 3d points used here are found using a website whose link is as follows
# https://www.mobilefish.com/services/record_mouse_coordinates/record_mouse_coordinates.php

for k in range(10):
    alpha_mean=0
    beta_mean=0
    x0_mean=0
    y0_mean=0
    rmse_mean=0
    theta_mean=0
    K_mean=np.array([[0,0,0],[0,0,0],[0,0,0]])
    R_mean=np.array([[0,0,0],[0,0,0],[0,0,0]])
    t_mean=np.array([0,0,0])
    K_mean=K_mean.astype('float64')
    R_mean=R_mean.astype('float64')
    t_mean=t_mean.astype('float64')
    for j in range(3):
        # i use 3 pairs of 6 points to get the K,r,t matrix for a single image
        if(k==0):
            points_2d=[[179,345],[250,344],[318,308],[286,376],[357,407],[219,444],[388,269],[431,376],[452,463],[451,343],[465,301],[481,251],[478,439],[496,402],[514,409],[213,548],[253,562],[204,580],[294,531],[369,528],[334,542],[389,572]]
            points_3d=[[1,8,5],[3,8,5],[5,8,6],[4,8,4],[6,8,3],[2,8,2],[7,8,7],[8,1,4],[8,3,2],[8,3,5],[8,4,6],[8,5,7],[8,5,3],[8,6,4],[8,7,4],[2,5,0],[3,4,0],[2,3,0],[4,6,0],[6,6,0],[5,5,0],[6,3,0]]
            points_2d=np.array(points_2d)
            points_3d=np.array(points_3d)
            random=np.random.choice(20,6,replace=False)
            #for z in range(len(points_2d)):
             #   points_2d[z][1]=813-points_2d[z][1]
        elif(k==1):
            points_2d=[[129,388],[197,376],[251,332],[227,405],[280,425],[171,490],[302,295],[346,391],[398,461],[400,367],[431,337],[464,301],[458,442],[496,416],[532,422],[250,596],[309,593],[309,627],[281,556],[327,532],[330,553],[410,562]]
            points_3d=[[1,8,5],[3,8,5],[5,8,6],[4,8,4],[6,8,3],[2,8,2],[7,8,7],[8,1,4],[8,3,2],[8,3,5],[8,4,6],[8,5,7],[8,5,3],[8,6,4],[8,7,4],[2,5,0],[3,4,0],[2,3,0],[4,6,0],[6,6,0],[5,5,0],[6,3,0]]
            points_2d=np.array(points_2d)
            points_3d=np.array(points_3d)
            random=np.random.choice(20,6,replace=False)
            #for z in range(len(points_2d)):
             #   points_2d[z][1]=813-points_2d[z][1]
        elif(k==2):
            points_2d=[[93,463],[136,439],[169,382],[152,468],[188,483],[120,583],[196,338],[240,437],[304,501],[307,407],[341,375],[376,343],[371,473],[408,445],[443,446],[257,680],[308,657],[347,688],[233,627],[251,589],[281,609],[356,597]]
            points_3d=[[1,8,5],[3,8,5],[5,8,6],[4,8,4],[6,8,3],[2,8,2],[7,8,7],[8,1,4],[8,3,2],[8,3,5],[8,4,6],[8,5,7],[8,5,3],[8,6,4],[8,7,4],[2,5,0],[3,4,0],[2,3,0],[4,6,0],[6,6,0],[5,5,0],[6,3,0]]
            points_2d=np.array(points_2d)
            points_3d=np.array(points_3d)
            random=np.random.choice(20,6,replace=False)
            #for z in range(len(points_2d)):
             #   points_2d[z][1]=813-points_2d[z][1]
        elif(k==3):
            points_2d=[[91,440],[148,416],[191,363],[174,442],[218,458],[133,543],[227,320],[278,415],[337,477],[337,386],[369,353],[403,319],[401,452],[439,423],[474,426],[251,643],[306,626],[331,659],[250,591],[280,558],[298,580],[378,572]]
            points_3d=[[1,8,5],[3,8,5],[5,8,6],[4,8,4],[6,8,3],[2,8,2],[7,8,7],[8,1,4],[8,3,2],[8,3,5],[8,4,6],[8,5,7],[8,5,3],[8,6,4],[8,7,4],[2,5,0],[3,4,0],[2,3,0],[4,6,0],[6,6,0],[5,5,0],[6,3,0]]
            points_2d=np.array(points_2d)
            points_3d=np.array(points_3d)
            random=np.random.choice(20,6,replace=False)
            #for z in range(len(points_2d)):
             #   points_2d[z][1]=813-points_2d[z][1]
        elif(k==4):
            points_2d=[[137,386],[198,370],[248,326],[226,398],[276,417],[177,488],[292,286],[336,379],[393,449],[393,355],[426,325],[461,290],[450,428],[492,401],[529,405],[268,589],[324,583],[332,618],[288,586],[326,520],[336,541],[416,546]]
            points_3d=[[1,8,5],[3,8,5],[5,8,6],[4,8,4],[6,8,3],[2,8,2],[7,8,7],[8,1,4],[8,3,2],[8,3,5],[8,4,6],[8,5,7],[8,5,3],[8,6,4],[8,7,4],[2,5,0],[3,4,0],[2,3,0],[4,6,0],[6,6,0],[5,5,0],[6,3,0]]
            points_2d=np.array(points_2d)
            points_3d=np.array(points_3d)
            random=np.random.choice(20,6,replace=False)
            #for z in range(len(points_2d)):
             #   points_2d[z][1]=813-points_2d[z][1]
        elif(k==5):
            points_2d=[[245,309],[531,366],[216,411],[310,542],[329,368],[439,412]]
            points_3d=[[3,8,5],[8,6,4],[2,8,2],[2,3,0],[6,8,3],[8,3,2]]
            points_2d=np.array(points_2d)
            points_3d=np.array(points_3d)
            #for z in range(len(points_2d)):
             #   points_2d[z][1]=813-points_2d[z][1]
            random=np.array([0,1,2,3,4,5])
        elif(k==6):
            points_2d=[[476,417],[310,543],[455,293],[264,322],[404,360],[325,522]]
            points_3d=[[8,6,4],[5,5,0],[8,5,7],[5,8,6],[8,3,5],[6,6,0]]
            points_2d=np.array(points_2d)
            points_3d=np.array(points_3d)
            #for z in range(len(points_2d)):
             #   points_2d[z][1]=813-points_2d[z][1]
            random=np.array([0,1,2,3,4,5])
        elif(k==7):
            points_2d=[[204,345],[407,340],[436,308],[409,531],[249,556],[179,454]]
            points_3d=[[3,8,5],[8,3,5],[8,4,6],[6,3,0],[2,5,0],[2,8,2]]
            points_2d=np.array(points_2d)
            points_3d=np.array(points_3d)
            #for z in range(len(points_2d)):
              #  points_2d[z][1]=813-points_2d[z][1]
            random=np.array([0,1,2,3,4,5])
        elif(k==8):
            points_2d=[[490,396],[418,422],[190,409],[357,445],[325,620],[146,502]]
            points_3d=[[8,7,4],[8,5,3],[4,8,4],[8,3,2],[2,3,0],[2,8,2]]
            points_2d=np.array(points_2d)
            points_3d=np.array(points_3d)
            #for z in range(len(points_2d)):
             #   points_2d[z][1]=813-points_2d[z][1]
            random=np.array([0,1,2,3,4,5])
        elif(k==9):
            points_2d=[[204,345],[407,340],[436,308],[409,531],[249,556],[179,454]]
            points_3d=[[8,5,3],[5,8,6],[6,8,3],[4,6,0],[3,8,5],[2,3,0]]
            random=np.array([0,1,2,3,4,5])
            points_2d=np.array(points_2d)
            points_3d=np.array(points_3d)
            #for z in range(len(points_2d)):
             #   points_2d[z][1]=813-points_2d[z][1]
        
            
        
        img1=[points_2d[k] for k in random]
        img2=[points_3d[k] for k in random]
        img1=np.array(img1)
        img2=np.array(img2)
        
        T2d,normalized_points_2d=normalise2d(img1)
        T3d,normalized_points_3d=normalise3d(img2)

        P=[]
        for i in range (len(normalized_points_3d)):
            temp=normalized_points_3d[i]
            temp=np.expand_dims(temp,axis=1)
            conc=np.expand_dims(np.array([0,0,0,0]),axis=1)
            t=np.concatenate((temp,conc),axis=0)
            t=np.concatenate((t,-normalized_points_2d[i][0]*temp),axis=0)
            P.append([float(t[0]),float(t[1]),float(t[2]),float(t[3]),float(t[4]),float(t[5]),float(t[6]),float(t[7]),float(t[8]),float(t[9]),float(t[10]),float(t[11])])
            row2=np.concatenate((conc,temp),axis=0)
            row2=np.concatenate((row2,-normalized_points_2d[i][1]*temp),axis=0)
            P.append([float(row2[0]),float(row2[1]),float(row2[2]),float(row2[3]),float(row2[4]),float(row2[5]),float(row2[6]),float(row2[7]),float(row2[8]),float(row2[9]),float(row2[10]),float(row2[11])])
        P1=np.array(P)
        u,sh,v=np.linalg.svd(P1,full_matrices=True)
        m=v[v.shape[0]-1,:]
        P_hat=np.reshape(m,(3,4))
        # denormalizing
        P_denormalize=np.dot(np.linalg.inv(T2d),np.dot(P_hat,T3d))
        # P denormalize is our projection matrix M=(A b)
        #A=P_denormalize[:,0:3]
        #b=P_denormalize[:,3]
        #rho=float(1/np.linalg.norm(A[2]))
        #print(rho)
        #r3=rho*A[2]
        #print(r3)
        #x0=np.dot(A[0],A[2])*rho**2
        #print(x0)
        #y0=np.dot(A[1],A[2])*rho**2
        #print(y0)
        #cos_theta= -1*np.dot((np.cross(A[0],A[2])),(np.cross(A[1],A[2])))/(np.linalg.norm(np.cross(A[0],A[2]))*np.linalg.norm(np.cross(A[1],A[2])))
        #alpha=float(np.linalg.norm(np.cross(A[0],A[2]))*(np.sin(np.arccos(cos_theta))))*rho**2
        #beta=np.linalg.norm(np.cross(A[1],A[2]))*np.sin(np.arccos(cos_theta))*rho**2
        #r1=rho**2*np.sin(np.arccos(cos_theta))*np.cross(A[1],A[2])/beta
        #r2=np.cross(r3,r1)
        #K=np.array([[alpha,-alpha/np.tan(np.arccos(cos_theta)),x0],[0,beta/np.sin(np.arccos(cos_theta)),y0],[0,0,1]])
        #t=rho*np.dot(np.linalg.inv(K),b)
        #R=[r1,r2,r3]
        
        K_R=P_denormalize[:,0:3]
        K,R=linalg.rq(K_R)
        t=np.dot(np.linalg.inv(K),P_denormalize[:,3])
        K_normalize=K/K[2,2]
        #print(K_normalize)
        alpha=K_normalize[0,0]
        cot_theta= -K_normalize[0,1]/K_normalize[0,0]
        tan_theta=1/cot_theta
        theta=abs(np.arctan(tan_theta))*180/3.14 # returns in radians
        print("theta="+ str(theta)) #degree
        sin_theta=np.sin(theta) # takes radians as input
        beta=-K_normalize[1,1]*sin_theta
        x0=K_normalize[0,2]
        y0=K_normalize[1,2]
        print(("alpha= " +str(alpha)))
        print(("beta= " +str(beta)))
        print("x0= "+ str(x0))
        print("yo= "+str(y0))
        #print("R"+ str(R))
        #print("t"+ str(t))

        rmse=0
        predicted_points=[]
        actual_points=[]
        for i in range(len(img2)):
            point_3d=np.array([img2[i,0],img2[i,1],img2[i,2],1])
            point_2d=np.dot(P_denormalize,point_3d)
            predicted_points.append([point_2d[0]/point_2d[2], point_2d[1]/point_2d[2]])
            actual_points.append([img1[i,0],img1[i,1]])
            rmse+= (img1[i,0]-(point_2d[0]/point_2d[2]))**2+(img1[i,1]-(point_2d[1]/point_2d[2]))**2

        rmse=math.sqrt(rmse)
        print("k= "+ str(k)+ " j= "+ str(j))
        print("RMSE= " + str(rmse))
        #print("beta= " + str(beta))
        #print("alpha= " + str(alpha))
        #print("x0= " + str(x0))
        #print("y0= " + str(y0))
        #print("theta= " +str(np.arccos(cos_theta)*180/3.141))
        #print(K)
        #print(R)
        #print(t)
        
        K_mean+=K_normalize
        alpha_mean+=abs(alpha)
        beta_mean+=abs(beta)
        t_mean+=t
        R_mean+=R
        rmse_mean+=rmse
        x0_mean+=abs(x0)
        y0_mean+=abs(y0)
        theta_mean+=abs(theta)
    if(rmse_mean<100):
        # this is done so as to reduce the errors due to some kind of distortion or non focused image
        actual_points_arr.append([str(k),actual_points])
        predicted_points_arr.append([str(k),predicted_points])
        
        K_arr.append(K_mean/3)
        alpha_arr.append(alpha_mean/3)
        beta_arr.append(beta_mean/3)
        x0_arr.append(x0_mean/3)
        y0_arr.append(y0_mean/3)
        R_arr.append(R_mean/3)
        t_arr.append(t_mean/3)
        theta_arr.append(theta_mean/3)
        rmse_arr.append(rmse_mean/3)
        
# Final parameters after iterating through the above loop
x0_final=sum(x0_arr)/len(x0_arr)
y0_final=sum(y0_arr)/len(y0_arr)
alpha_final=sum(alpha_arr)/len(alpha_arr)
theta_final=sum(theta_arr)/len(theta_arr)
beta_final=sum(beta_arr)/len(beta_arr)
rmse_final=sum(rmse_arr)/len(rmse_arr)
K_final=sum(K_arr)/len(K_arr)
R_final=R_arr
t_final=t_arr

# final intrinsic parameters
# the extrinsic parameters are in R_final,t_final containing extrinsic parameters for each image
print("x0= "+ str(x0_final))
print("y0= "+ str(y0_final))
print("alpha= "+ str(alpha_final))
print("theta= " + str(theta_final))
print("beta= "+ str(beta_final))
print("rmse= " + str(rmse_final))
