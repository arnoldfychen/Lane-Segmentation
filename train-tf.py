import time
import pandas as pd
import numpy as np

import tensorflow as tf

from utils.data_feeder import get_feeder_data, train_image_gen
from models.unet_base-tf import UnetBase
from models.unet_simple-tf import UnetSimple

from models.deeplabv3p-tf import Deeplabv3p

# Compute Mean Iou
def mean_iou(pred, label, num_classes=8):
    pred = tf.math.argmax(pred, axis=1)
    pred = tf.cast(pred, tf.int32)
    label = tf.cast(label, tf.int32)
    mean_iou, update_op = tf.compat.v1.metrics.mean_iou(label,pred, num_classes)
    return mean_iou
    
# Get Loss Function
def dice_loss_func(y_true, y_pred, smooth=1):    
    mean_loss = 0
    for i in range(y_pred.shape(-1)):
        intersection = tf.reduce_sum(y_true[:,:,:,i] * y_pred[:,:,:,i], axis=[1,2,3])
        union = tf.reduce_sum(y_true[:,:,:,i], axis=[1,2,3]) + tf.reduce_sum(y_pred[:,:,:,i], axis=[1,2,3])
        mean_loss += (2. * intersection + smooth) / (union + smooth)    
    return 1 - tf.reduce_mean(mean_loss, axis=0)

def create_loss(predict, label, num_classes):
    predict = tf.transpose(predict, perm=[0, 2, 3, 1])
    predict = tf.reshape(predict, shape=[-1, num_classes])
    predict = tf.nn.softmax(predict)
    label = tf.reshape(label, shape=[-1, 1])
    # BCE with DICE
    bce_loss = tf.nn.sigmoid_cross_entropy_with_logits(label,predict)
    dice_loss = dice_loss_func(label,predict)
    loss = bce_loss + dice_loss
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
    add_num = 1
    num_classes = 8
    batch_size = 3
    log_iters = 100
    base_lr = 0.0006
    save_model_iters = 2000
    use_pretrained = False
    network = 'deeplabv3p'
    save_model_path = "./model_weights/paddle_" + network + "_"
    model_path = "./model_weights/paddle_" + network + "_12_end"
    epoches = 2
    crop_offset = 690
    data_dir = './data_list/train.csv'

    # Get data list and split it into train and validation set.
    train_list = pd.read_csv(data_dir)
    
    iter_id = 0
    total_loss = 0.0
    total_miou = 0.0
    prev_time = time.time()
    # Train
    print('Train Data Size:', len(train_list))
    train_reader = train_image_gen(train_list, batch_size, IMG_SIZE, crop_offset)
    # Create model and define optimizer
    model = create_network(network=network)
    optimizer=tf.optimizers.Adam(learning_rate=base_lr)

    # Training
    for epoch in range(epoches):
        print('Start Training Epoch: %d'%(epoch + 1))
        train_length = len(train_list)
        for iteration in range(int(train_length / batch_size)):
            train_data,label_data = next(train_reader)
            with tf.GradientTape() as t:
                predict = model(train_data,num_classes)
                current_loss, miou = create_loss(predict, label_data, num_classes)
                grads = t.gradient(current_loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads,model.trainable_variables))
                
                iter_id += 1
                total_loss += np.mean(current_loss)
                total_miou += np.mean(miou)

            if iter_id % log_iters == 0: # Print log
                end_time = time.time()
                print(
                "Iter - %d: train loss: %.3f, mean iou: %.3f, time cost: %.3f s"
                % (iter_id, total_loss / log_iters, total_miou / log_iters, end_time - prev_time))
                total_loss = 0.0
                total_miou = 0.0
                prev_time = time.time()

            if iter_id % save_model_iters == 0: # save model
                dir_name = save_model_path + str(epoch +'_'+ add_num) + '_' + str(iter_id)
                tf.keras.models.save_model(model, filepath=dir_name)
                print("Saved checkpoint: %s" % (dir_name))
        iter_id = 0
        dir_name = save_model_path + str(epoch +'_'+ add_num) + '_end'
        tf.keras.models.save_model(model, filepath=dir_name)
        print("Saved checkpoint: %s" % (dir_name))

# Main
if __name__ == "__main__":
    main()
