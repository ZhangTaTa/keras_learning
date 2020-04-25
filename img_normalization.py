import os
from PIL import Image
import numpy as np


def img_prog(folderName, imgName, thumbSize):        # 图片所在文件夹名字，图片名字，缩略尺寸((int,int))
    path = "./img/%s/%s" %(folderName, imgName)
    try:
        im = Image.open(path)
        height = im.size[0]
        width = im.size[1]
        size = min(height, width)
        centre = [height/2, width/2]
        imCrop = im.crop(
            (int(centre[0] - size/2), int(centre[1] - size/2), int(centre[0] + size/2), int(centre[1] + size/2)))
        imCrop.thumbnail(thumbSize)      # 缩略图
        imCrop.save("./img_thumb/%s" % imgName)
        data = np.asarray(imCrop)
        data = data.reshape((1, thumbSize[0]*thumbSize[1]*3))
    except:
        if imgName != "783.jpg":
            print("error size: " + str(data.shape))
        data = []
    return data     # 返回处理后的图片data


# 标准化图片
thumbSize = (100, 100)      # 标准化后图片尺寸
imgDataMatrix = []
imgNameMatrix = []
# 获取图片所在文件夹
folNameList = os.listdir("./img")
folNameList.remove("record.txt")
# 按文件夹顺序标准化图片
for i in range(0, len(folNameList)):
    imgNameList = os.listdir("./img/%s" % folNameList[i])
    for j in range(0, len(imgNameList)):
        imgData = img_prog(folNameList[i], imgNameList[j], thumbSize)
        if len(imgData) != 0:
            imgDataMatrix.append(imgData)
            imgNameMatrix.append(i)
        else:
            print("error item:%s %s" % (folNameList[i], imgNameList[j]))
# 储存所有标准化图片数据
imgDataMatrix = np.array(imgDataMatrix)
imgDataMatrix = imgDataMatrix.reshape(imgDataMatrix.shape[0], imgDataMatrix.shape[2])
imgNameMatrix = np.array(imgNameMatrix)
imgNameMatrix = imgNameMatrix.reshape((len(imgNameMatrix), 1))
np.save("imgDataMatrix.npy", imgDataMatrix)
np.save("imgNameMatrix.npy", imgNameMatrix)
