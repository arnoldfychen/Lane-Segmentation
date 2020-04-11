import os
import pandas as pd
from sklearn.utils import shuffle


#================================================
# make train & validation lists
#================================================
label_list = []
image_list = []

#image_dir = '/home/gujingxiao/projects/PaddlePaddle/Image_Data/'
#label_dir = '/home/gujingxiao/projects/PaddlePaddle/Gray_Label/'

image_dir = '/root/private/LaneDataSet/TrainSet/Image_Data/'
label_dir = '/root/private/LaneDataSet/TrainSet/Gray_Label/'

for s1 in os.listdir(image_dir): # in my environment on HCTech's server, i.e., s1 in ['ColorImage_road02', 'ColorImage_road03', 'ColorImage_road04']
    roadx = s1.split("_")[1]     # roadx in ['road02','road03','road04']
    #image_sub_dir1 = os.path.join(image_dir, s1)
    #label_sub_dir1 = os.path.join(label_dir, 'Label_' + str.lower(s1), 'Label')
    
    image_sub_dir1 = os.path.join(image_dir, s1,'ColorImage')
    label_sub_dir1 = os.path.join(label_dir, 'Label_' + str.lower(roadx), 'Label')
    # print(image_sub_dir1, label_sub_dir1)

    for s2 in os.listdir(image_sub_dir1):  # s2 in ['Record001','Record002','Record003','Record004','Record005','Record006','Record007']
        image_sub_dir2 = os.path.join(image_sub_dir1, s2)
        label_sub_dir2 = os.path.join(label_sub_dir1, s2)
        # print(image_sub_dir2, label_sub_dir2)

        for s3 in os.listdir(image_sub_dir2):  #s3 in ['Camera 5','Camera 6']
            image_sub_dir3 = os.path.join(image_sub_dir2, s3)
            label_sub_dir3 = os.path.join(label_sub_dir2, s3)
            # print(image_sub_dir3, label_sub_dir3)
            #e.g., image_sub_dir3 = '/root/private/LaneDataSet/TrainSe/Image_Data/ColorImage_road02/ColorImage/Record001/Camera 5'
            #e.g., label_sub_dir3 = '/root/private/LaneDataSet/TrainSet/Gray_Label/Label_road02/Label/Record001/Camera 5'

            for s4 in os.listdir(image_sub_dir3):   #s4 in *.jpg
                s44 = s4.replace('.jpg','_bin.png') #s44 in *_bin.png
                image_sub_dir4 = os.path.join(image_sub_dir3, s4)
                label_sub_dir4 = os.path.join(label_sub_dir3, s44)
                #if not os.path.exists(image_sub_dir4):
                #    print(image_sub_dir4)
                if not os.path.exists(label_sub_dir4):
                    print(label_sub_dir4," doesn'nt exist!!!")
                # print(image_sub_dir4, label_sub_dir4)
                image_list.append(image_sub_dir4)
                label_list.append(label_sub_dir4)
                
print(len(image_list), len(label_list))

save = pd.DataFrame({'image':image_list, 'label':label_list})
save_shuffle = shuffle(save)
save_shuffle.to_csv('../data_list/train.csv', index=False)