from distutils.command.build import build
import cv2
from tracker import *
import time
from skimage.io import imread
import glob
import numpy as np
import json
from utils1.utils import build_lut_cv2, volume_render_color
# Create tracker object

# f = open('cell_coords.json')
# f = open('cell_coords_trackres1.json')
# f = open('cell_coords_track_res1.json')


# f=open('./logs/cell_coords_0512_2GA_track_res_pytorch.json')
f=open('cell_coords_0512_2GA_track_res_3dee3.json')
video_path = '/data/nanolive/1hr_fixation_2GA/1hr_fixation_2GA_enhanced.avi'
tiff_raw_path = '/data/nanolive/Zihan/1hr_fixation/2%GA/*.tiff'
# seg_res_path = '/data/nanolive/Zihan/1hr_fixation/2%GA/seg_res_pytorch/*.tiff'
track_res_path = '/data/nanolive/Zihan/1hr_fixation/2%GA/track_res_3dee3/*.tiff'
#---------------------------------------------------
# f=open('cell_coords_0512_2GA_track_res1.json')
# video_path = '/data/nanolive/1hr_fixation_2GA/1hr_fixation_2GA.avi'
# tiff_raw_path = '/data/nanolive/Zihan/1hr_fixation/2%GA/*.tiff'
# seg_res_path = '/data/nanolive/Zihan/1hr_fixation/2%GA/seg_res/*.tiff'

#----------------------------------
# f=open('cell_coords_low_thred.json')
# # f=open('cell_coords_high_thred.json')
# video_path ='p20uMZnCl_uptakebuffer_every20in50min_RI.avi'
# tiff_raw_path = '/data/nanolive/Zihan/02172022/20uMZnCl_uptakebuffer_every20in50min_2nd/*.tiff'
# seg_res_path = '/data/nanolive/Zihan/02172022/20uMZnCl_uptakebuffer_every20in50min_2nd/seg_res2/*.tiff'

tiff_raw_files = sorted(glob.glob(tiff_raw_path))
track_res_files = sorted(glob.glob(track_res_path))
cell_coord = json.load(f)


cap = cv2.VideoCapture(video_path)
#video resolution 1024*1024
#tiff raw 512*512
track_result = cv2.VideoWriter('track_res.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         5, (1024,1024))
seg_result = cv2.VideoWriter('seg_res.avi', 
                         cv2.VideoWriter_fourcc('M','J','P','G'),
                         5, (512,512),True)
seg_result_3d = cv2.VideoWriter('seg_res_3d.avi', 
                         cv2.VideoWriter_fourcc('M','J','P','G'),
                         5, (512,512),True)
raw_video = cv2.VideoWriter('raw.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         5, (512,512),False)
count =0
while True:
    ret, frame = cap.read()
    # cv2.imwrite("frame%d.jpg" % count, frame)     # save frame as JPEG file
    # break
    tiff_img = imread(tiff_raw_files[count]).transpose((1,2,0))
    mask = imread(track_res_files[count]).transpose((1,2,0))
    mask_proj =np.max(mask,axis=-1)
    raw_proj = np.max(tiff_img,axis=-1)
    # Extract Region of interest
    # roi = frame[1: 1024,1: 1025]
    # roi = frame

    time.sleep(0.5)

    detections =cell_coord[str(count)][0] # it was packed insided a list, so add [0]

    for box_id in detections.keys():
        # min_x,max_x,min_y,max_y,_,_ =detections[str(box_id)]
        min_y,max_y,min_x,max_x,_,_ =detections[str(box_id)]

            
        # cv2.putText(mask, str(box_id), (min_x, min_y - 15), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 2)
        # cv2.rectangle(mask, (min_x, min_y), (max_x, max_y), (255, 255, 255), 3)
        offset = 182
        factor = 1.285
        min_x =int(offset-8 +1.02*factor *min_x)
        max_x =int(offset +factor *max_x)
        min_y =int(offset +factor *min_y)
        max_y =int(offset +factor *max_y)

        cv2.putText(frame, str(box_id), (min_x+25, min_y + 35), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (255, 0, 0), 3)
    # break

    cv2.imshow("Volume rendering Frame", frame)
#   convert to color
    mask_proj_color = cv2.cvtColor(mask_proj, cv2.COLOR_GRAY2BGR);
    im_color = cv2.LUT(mask_proj_color.astype(np.uint8),build_lut_cv2())
    mask_3d =volume_render_color(mask,a=0.02)
    #-------------------------------
    im3 =cv2.imshow('Segmented Cells',im_color)
    im4 =cv2.imshow('Segmented Cells color',mask_3d)
    cv2.imshow("Raw image (tiff)", raw_proj)
    # cv2.imshow("segmented cells", volume_render_color(im_color))
    track_result.write(frame)
    seg_result.write(im_color.astype('uint8'))
    seg_result_3d.write((mask_3d*255).astype('uint8'))
    raw_video.write((raw_proj/255*3).astype('uint8'))
    count +=1
    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()