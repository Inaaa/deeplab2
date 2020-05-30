import os
path = "/mrtstorage/users/chli/cityscapes/slope_data" #文件夹目录
files= os.listdir(path) #得到文件夹下的所有文件名称
s = []
for filename in files: #遍历文件夹

        if filename.startswith('.'):
            continue
        name = filename.split('.')[0]
        vorname = name.split('_')[0]
	    #id = name.split('_')
