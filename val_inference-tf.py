import cv2
import sys
import os
import time
import pandas as pd
import numpy as np
import tensorflow as tf

from utils.process_labels import decode_color_labels
from utils.image_process import crop_resize_data, expand_resize_data
from utils.data_feeder import get_feeder_data, val_image_gen

from models.unet_base-tf import UnetBase
from models.unet_simple-tf import UnetSimple
from models.deeplabv3p-tf import Deeplabv3p

def mean_iou(pred, label, num_classes):
    pred = tf.math.argmax(pred, axis=1)
    pred = tf.cast(pred, tf.int32)
    label = tf.cast(label, tf.int32)
    mean_iou, update_op = tf.compat.v1.metrics.mean_iou(label,pred, num_classes)
    return miou

def create_loss(predict, label, num_classes):
    predict = tf.transpose(predict, perm=[0, 2, 3, 1])
    predict = tf.reshape(predict, shape=[-1, num_classes])
    predict = tf.nn.softmax(predict)
    label = tf.reshape(label, shape=[-1, 1])
    bce_loss = tf.nn.sigmoid_cross_entropy_with_logits(label,predict)
    loss = bce_loss
    miou = mean_iou(predict, label, num_classes)
    return tf.math.reduce_mean(loss), miou

def create_network(network='unet_simple'):
    if network == 'unet_base':
        model = UnetBase()
    elif network == 'unet_simple':
        model = UnetSimple()
    elif network == 'deeplabv3p':
        model = Deeplabv3p()
    else:
        raise Exception('Not support this model:', network)
    print('The program will run', network)
    return model


# The main method
def main():
    IMG_SIZE =[1536, 512]
    SUBMISSION_SIZE = [3384, 1710]
    save_test_logits = False
    num_classes = 8
    batch_size = 4
    log_iters = 100
    network = 'unet_simple'

    # Define paths for each model
    if network == 'deeplabv3p':
        model_path = "./model_weights/paddle_deeplabv3p_8_end_060223"
        npy_dir = '/npy_save/deeplabv3p/'
    elif network == 'unet_base':
        model_path = "./model_weights/paddle_unet_base_10_end_059909"
        npy_dir = '/npy_save/unet_base/'
    elif network == 'unet_simple':
        model_path = "./model_weights/paddle_unet_simple_12_end_060577"
        npy_dir = '/npy_save/unet_simple/'

    program_choice = 2 # 1 - Validtion; 2 - Test
    show_label = False
    crop_offset = 690
    data_dir = './data_list/val.csv'
    test_dir = '/root/private/LaneDataSet/TestSet/Image_Data/ColorImage/'
    sub_dir = './test_submission/'

    # Get data list and split it into train and validation set.
    val_list = pd.read_csv(data_dir)

    iter_id = 0
    total_loss = 0.0
    total_miou = 0.0
    prev_time = time.time()
    # Validation
    if program_choice == 1:
        val_reader = val_image_gen(val_list, batch_size=batch_size, image_size=IMG_SIZE, crop_offset=crop_offset)
        model = create_network(network=network)
        model.load_weights(model_path)
        print("loaded model from: %s" % model_path)

        print('Start Validation!')
        for iteration in range(int(len(val_list) / batch_size)):
            val_data,val_label = next(val_reader)
            results = model.evaluate(val_data,val_label)
            if iter_id % log_iters == 0:
                print('Finished Processing %d Images.' %(iter_id * batch_size))
            iter_id += 1
            total_loss += np.mean(results[0])
            total_miou += np.mean(results[1])
            # label to mask
            if show_label == True:
                label_image = val_label[0]
                color_label_mask = decode_color_labels(label_image)
                color_label_mask = np.transpose(color_label_mask, (1, 2, 0))
                cv2.imshow('gt_label', cv2.resize(color_label_mask, (IMG_SIZE[0], IMG_SIZE[1])))

                prediction = np.argmax(results[2][0], axis=0)
                color_pred_mask = decode_color_labels(prediction)
                color_pred_mask = np.transpose(color_pred_mask, (1, 2, 0))
                cv2.imshow('pred_label', cv2.resize(color_pred_mask, (IMG_SIZE[0], IMG_SIZE[1])))
                cv2.waitKey(0)

        end_time = time.time()
        print("validation loss: %.3f, mean iou: %.3f, time cost: %.3f s"
            % (total_loss / iter_id, total_miou / iter_id, end_time - prev_time))
    # Test
    elif program_choice == 2:
        model = create_network(network=network)
        model.load_weights(model_path)
        print("loaded model from: %s" % model_path)

        print('Start Making Submissions!')
        test_list = os.listdir(test_dir)
        for test_name in test_list:
            test_ori_image = cv2.imread(os.path.join(test_dir, test_name))
            test_image = crop_resize_data(test_ori_image, label=None, image_size=IMG_SIZE, offset=crop_offset)
            out_image = np.expand_dims(np.array(test_image), axis=0)
            out_image = out_image[:, :, :, ::-1].transpose(0, 3, 1, 2).astype(np.float32) / (255.0 / 2) - 1
            results_1 = model.evaluate(out_image,None)
            
            if iter_id % 20 == 0:
                print('Finished Processing %d Images.' %(iter_id))
            iter_id += 1
            prediction = np.argmax(results_1[0][0], axis=0)

            # Save npy files
            if save_test_logits == True:
                np.save(npy_dir + test_name.replace('.jpg', '.npy'), results_1[0][0])

            # Save Submission PNG
            submission_mask = expand_resize_data(prediction, SUBMISSION_SIZE, crop_offset)
            cv2.imwrite(os.path.join(sub_dir, test_name.replace('.jpg', '.png')), submission_mask)

            # Show Label
            if show_label == True:
                cv2.imshow('test_image', cv2.resize(test_ori_image,(IMG_SIZE[0], IMG_SIZE[1])))
                cv2.imshow('pred_label', cv2.resize(submission_mask,(IMG_SIZE[0], IMG_SIZE[1])))
                cv2.waitKey(0)
        sys.stdout.flush()

# Main
if __name__ == "__main__":
    main()
