import os
import numpy as np
import cv2

import tensorflow as tf
from utils.image_process import expand_resize_data

# Create a bilineraNet to resize predictions to full size
def BilinearNet(tf.keras.Model):
    def __init__(self):
        super(BilinearNet,self)._init__()
        
    def call(self,predictions, submission_size, crop_offset):
        logit = tf.compat.v1.image.resize_bilinear(predictions,(submission_size[0], submission_size[1] - crop_offset))
        return logit

# Main
if __name__ == "__main__":
    print('Start Making Ensemble Submissions!')
    test_dir = '/root/private/LaneDataSet/TestSet/Image_Data/ColorImage/'
    sub_dir = './test_submission/'
    IMG_SIZE = [1536, 512]
    SUBMISSION_SIZE = [3384, 1710]
    crop_offset = 690
    # Ignore Class 4
    label_num = 8
    test_list = os.listdir(test_dir)

    # Three Folders which save npy files corresponding to all test images
    # ensemble index 1 0.61234
    model_lists = ['/npy_save/deeplabv3p/',
                   '/npy_save/unet_base/',
                   '/npy_save/unet_simple/']

    # Build Model & Initialize Program
    bilinearNet = BilinearNet()
    
    for index in range(len(test_list)):
        test_name = test_list[index]
        print(index, test_name)

        # Load three diffirent npys and then do average
        model_logits1 = np.load(model_lists[0] + test_name.replace('.jpg', '.npy'))
        model_logits2 = np.load(model_lists[1] + test_name.replace('.jpg', '.npy'))
        model_logits3 = np.load(model_lists[2] + test_name.replace('.jpg', '.npy'))
        avg_model_logits = (model_logits1 + model_logits2 + model_logits3) / 3.0
        logits_input = np.expand_dims(np.array(avg_model_logits), axis=0)
        result = bilinearNet(logits_input,SUBMISSION_SIZE, crop_offset)
        prediction = np.argmax(results[0][0], axis=0)
        # Convert prediction to submission image
        submission_mask = expand_resize_data(prediction, SUBMISSION_SIZE, crop_offset)
        # Save submission png
        cv2.imwrite(os.path.join(sub_dir, test_name.replace('.jpg', '.png')), submission_mask)