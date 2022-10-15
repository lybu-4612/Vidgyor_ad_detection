For Vidgyor 
The attachments in mail
1) final code to find ads in a given video and their duration  (ad_detection.py)
2) image fetched from one of the videos for masking (img_19.jpg)
3) trained Keras model for detection( trained over 2700 images for each class, used mobile net transfer learning (tv_ad_224.h5)





# tv_ad_detection
Sample Input Video Files:  (Please check these out)
https://mkvidupload.s3.amazonaws.com/livetvrecordings/11-31-06.mp4
https://mkvidupload.s3.amazonaws.com/livetvrecordings/11-41-06.mp4
https://mkvidupload.s3.amazonaws.com/livetvrecordings/14-11-03.mp4
https://mkvidupload.s3.amazonaws.com/livetvrecordings/15-40-59.mp4
More recordings: http://newrecording.vidgyor.com/vod/recordings/transtv/


---> extract and masked the logo with original frame.
---> write the images in 224X224 size.
---> collected 3100 images in each category.
---> trained using transfer learnig of mobile net model.

---> final output.
---> if given a video, it will give time interval and duration of ads.
---> also shows given frame is ad or not

