import cv2
import numpy as np
import math


def read_square_image(file, cam, boxsize, type):
    # from file
    if type == 'IMAGE':
        oriImg = cv2.imread(file)
    # from webcam
    elif type == 'WEBCAM':
        _, oriImg = cam.read()

    scale = boxsize / (oriImg.shape[0] * 1.0)
    imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)

    output_img = np.ones((boxsize, boxsize, 3)) * 128

    if imageToTest.shape[1] < boxsize:
        offset = imageToTest.shape[1] % 2
        output_img[:, int(boxsize/2-math.ceil(imageToTest.shape[1]/2)):int(boxsize/2+math.ceil(imageToTest.shape[1]/2)+offset), :] = imageToTest
    else:
        output_img = imageToTest[:, int(imageToTest.shape[1]/2-boxsize/2):int(imageToTest.shape[1]/2+boxsize/2), :]
    return output_img

def resize_pad_img(img, scale, output_size):
    resized_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    pad_h = (output_size - resized_img.shape[0]) // 2
    pad_w = (output_size - resized_img.shape[1]) // 2
    pad_h_offset = (output_size - resized_img.shape[0]) % 2
    pad_w_offset = (output_size - resized_img.shape[1]) % 2
    resized_pad_img = np.pad(resized_img, ((pad_w, pad_w+pad_w_offset), (pad_h, pad_h+pad_h_offset), (0, 0)),
                             mode='constant', constant_values=128)

    return resized_pad_img


def draw_predicted_heatmap(heatmap, input_size):
    heatmap_resized = cv2.resize(heatmap, (input_size, input_size))

    output_img = None
    tmp_concat_img = None
    h_count = 0
    for joint_num in range(heatmap_resized.shape[2]):
        if h_count < 4:
            tmp_concat_img = np.concatenate((tmp_concat_img, heatmap_resized[:, :, joint_num]), axis=1) \
                if tmp_concat_img is not None else heatmap_resized[:, :, joint_num]
            h_count += 1
        else:
            output_img = np.concatenate((output_img, tmp_concat_img), axis=0) if output_img is not None else tmp_concat_img
            tmp_concat_img = None
            h_count = 0
    # last row img
    if h_count != 0:
        while h_count < 4:
            tmp_concat_img = np.concatenate((tmp_concat_img, np.zeros(shape=(input_size, input_size), dtype=np.float32)), axis=1)
            h_count += 1
        output_img = np.concatenate((output_img, tmp_concat_img), axis=0)

    # adjust heatmap color
    output_img = output_img.astype(np.uint8)
    output_img = cv2.applyColorMap(output_img, cv2.COLORMAP_JET)
    return output_img

def extract_2d_joint_from_heatmap(heatmap, input_size, joints_2d):
    heatmap_resized = cv2.resize(heatmap, (input_size, input_size))

    for joint_num in range(heatmap_resized.shape[2]):
        joint_coord = np.unravel_index(np.argmax(heatmap_resized[:, :, joint_num]), (input_size, input_size))
        joints_2d[joint_num, :] = joint_coord

    return


def extract_3d_joints_from_heatmap(joints_2d, x_hm, y_hm, z_hm, input_size, joints_3d):

    for joint_num in range(x_hm.shape[2]):
        coord_2d_y = joints_2d[joint_num][0]
        coord_2d_x = joints_2d[joint_num][1]

        joint_x = x_hm[max(int(coord_2d_x/8), 1), max(int(coord_2d_y/8), 1), joint_num] * 10
        joint_y = y_hm[max(int(coord_2d_x/8), 1), max(int(coord_2d_y/8), 1), joint_num] * 10
        joint_z = z_hm[max(int(coord_2d_x/8), 1), max(int(coord_2d_y/8), 1), joint_num] * 10
        joints_3d[joint_num, 0] = joint_x
        joints_3d[joint_num, 1] = joint_y
        joints_3d[joint_num, 2] = joint_z
    joints_3d -= joints_3d[14, :]

    return

