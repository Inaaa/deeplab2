import cv2

# 将视频文件按照每秒一张图进行抓取图片

# path = file[:file.index('.')]

vc = cv2.VideoCapture('/home/chli/cc_code2/deeplab/Driving the Trans-Canada Highway in Newfoundland.mp4')  # 读入视频文件
f = 1
i = 1
if vc.isOpened():  # 判断是否正常打开
    rval, frame = vc.read()
else:
    rval = False

timeF = 25  # 视频帧计数间隔频率

while rval:  # 循环读取视频帧
    rval, frame = vc.read()
    if f % timeF == 0:  # 每隔timeF帧进行存储操作
        cv2.imwrite('/home/chli/cc_code2/deeplab/images/' + str(i) + '.jpg', frame)  # 存储为图像
        i = i + 1
        print("i：" + str(i) + ',frame:' + str(f))
    f = f + 1
    cv2.waitKey(1)

vc.release()