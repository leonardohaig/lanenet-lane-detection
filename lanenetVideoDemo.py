#!/usr/bin/env python3
#coding=utf-8

import argparse
import os.path as ops
import time

import cv2
import tensorflow as tf

from config import global_config
from lanenet_model import lanenet
from lanenet_model import lanenet_postprocess

CFG = global_config.cfg


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, help='The video path',default='/media/liheng/0F521CCC0F521CCC/7.29/ADAS_usb4mm-20190729-171456.avi')
    parser.add_argument('--weights_path', type=str, help='The model weights path',default='/home/liheng/Downloads/Compressed/New_Tusimple_Lanenet_Model_Weights/new/tusimple_lanenet_vgg_2018-10-19-13-33-56.ckpt-200000.data-00000-of-00001')

    return parser.parse_args()

def test_lanenet(video_path, weights_path):
    """

    :param video_path:
    :param weights_path:
    :return:
    """
    assert ops.exists(video_path), '{:s} not exist'.format(video_path)
    assert ops.exists(weights_path), '{:s} not exist'.format(weights_path)


    input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')

    net = lanenet.LaneNet(phase='test', net_flag='vgg')
    binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='lanenet_model')

    postprocessor = lanenet_postprocess.LaneNetPostProcessor()

    saver = tf.train.Saver()

    # Set sess configuration
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config)

    with sess.as_default():

        saver.restore(sess=sess, save_path=weights_path)
        nFrameIdx = 0
        videoCapture = cv2.VideoCapture(video_path)
        while True:
            ret,image = videoCapture.read()

            if not ret:break
            nFrameIdx += 1

            image_vis = image
            image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)
            image = image / 127.5 - 1.0


            t_start = time.time()
            binary_seg_image, instance_seg_image = sess.run(
                [binary_seg_ret, instance_seg_ret],
                feed_dict={input_tensor: [image]}
            )
            t_cost = time.time() - t_start
            print('Single imgae inference cost time: {:.5f}s'.format(t_cost))

            postprocess_result = postprocessor.postprocess(
                binary_seg_result=binary_seg_image[0],
                instance_seg_result=instance_seg_image[0],
                source_image=image_vis
            )

            resImage = postprocess_result['source_image']
            if resImage is None:
                continue
            else:
                cv2.imshow("DetectRes", resImage)
                cv2.waitKey(1)

    sess.close()

    return


if __name__ == '__main__':
    """
    test code
    """
    # init args
    args = init_args()

    test_lanenet(args.video_path, args.weights_path)
