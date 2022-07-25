import SimpleITK as sitk
import numpy as np
import os
from tqdm import trange


def get_listdir(path):  # 获取目录下所有gz格式文件的地址，返回地址list
    tmp_list = []
    for file in os.listdir(path):
        if os.path.splitext(file)[1] == '.gz':
            file_path = os.path.join(path, file)
            tmp_list.append(file_path)
    return tmp_list


def dice_3d(mask_path, pred_path, label):
    mask_sitk_img = sitk.ReadImage(mask_path)
    mask_img_arr = sitk.GetArrayFromImage(mask_sitk_img)
    pred_sitk_img = sitk.ReadImage(pred_path)
    pred_img_arr = sitk.GetArrayFromImage(pred_sitk_img)
    pred_img_arr = pred_img_arr.astype(np.uint16)
    mask_img_arr[mask_img_arr != label] = 0
    mask_img_arr[mask_img_arr == label] = 1
    pred_img_arr[pred_img_arr != label] = 0
    pred_img_arr[pred_img_arr == label] = 1

    denominator = np.sum(mask_img_arr) + np.sum(pred_img_arr)
    numerator = 2 * np.sum(mask_img_arr * pred_img_arr)
    dice = numerator / denominator
    # print(dice)
    return dice
if __name__ == '__main__':
    mask = ""
    pred = ""

    mask_files = get_listdir(mask)
    mask_files.sort()
    
    pred_files = get_listdir(pred)
    pred_files.sort()
    print(mask_files)
    print(pred_files)
    dice_all = []
    for i in trange(len(mask_files)):
        dice_single = dice_3d(mask_files[i],pred_files[i],1)
        dice_all.append(dice_single)

    print(f'The average Dice of test image is {np.mean(np.array(dice_all)) }')
