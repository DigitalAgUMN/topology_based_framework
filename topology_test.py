import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import os
from utility_functions import get_hist2d, get_target, get_coordinate
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from smoother import whittaker_smooth

def copy_image(dir, save_dir, img_index, sensor):
    import shutil
    if not os.path.exists(os.path.join(save_dir, 'img')):
        os.makedirs(os.path.join(save_dir, 'img'))
    if sensor == 's2':
        for i in range(0, 60):
            if os.path.exists(os.path.join(dir, str(i)) + r'\\'+img_index+'.tif'):
                shutil.copyfile(os.path.join(dir, str(i)) + r'\\'+img_index+'.tif', os.path.join(save_dir, 'img', img_index+'_'+str(i)+'.tif'))
    if sensor == 'landsat':
        for i in range(0, 8):
            if os.path.exists(os.path.join(dir, str(i)+'_ls8') + r'\\' + img_index + '.tif'):
                shutil.copyfile(os.path.join(dir, str(i)+'_ls8') + r'\\' + img_index + '.tif',
                                os.path.join(save_dir, img_index + '_' + str(i) + '.tif'))
    if os.path.exists(os.path.join(dir, 'target') + r'\\'+img_index+'.tif'):
        shutil.copyfile(os.path.join(dir, 'target') + r'\\'+img_index+'.tif', os.path.join(save_dir, 'img', img_index+'.tif'))

def heat_map(dir, folder_id, img_index, band=['R', 'G']):
    band_name = {
        "Red":0,
        "Green":1,
        "Blue":2,
        "RDEG1": 0,
        "RDEG2": 4,
        "RDEG3": 5,
        "NIR":1,
        "RDEG4": 7,
        "SWIR1": 2,
        "SWIR2": 9,
    }
    if not os.path.exists(os.path.join(dir, folder_id)):
        os.makedirs(os.path.join(dir, folder_id))
    target = io.imread(os.path.join(dir, 'img', img_index +'.tif'))
    for img_name in os.listdir(os.path.join(dir, 'img')):
        if len(img_name.split('_')) != 3:
            continue
        img = io.imread(os.path.join(os.path.join(dir, 'img'), img_name)) / 10000
        img = img[:,:, [band_name[band[0]], band_name[band[1]]]]
        # make sure the shape is (channel, height, weight)
        # each image is a square
        if img.shape[0] == img.shape[1]:
            img = img.transpose(2, 0, 1)[np.newaxis, :, :, :]
        """
        convert the image to 3 classes:
        0 for background
        1 for corn
        2 for soybean
        """
        code_class = [1, 3, 4]
        bins_range = [0.3, 0.6]
        arr_class = np.zeros_like(target)
        for i_cur, code_cur in enumerate(code_class):
            arr_class[target == code_cur] = i_cur + 1
        list_img, list_xedges, list_yedges = get_hist2d(img, bins=129, arr_class=arr_class, bins_range=bins_range)
        list_img2, list_xedges2, list_yedges2 = get_hist2d(img, bins=129, bins_range=bins_range)
        # mask_img, mask_xedges, mask_yedges = get_hist2d(img, arr_class=(mask != 0).astype(np.int8), bins_range=bins_range)
        x_coor = get_coordinate(list_img2)
        y_coor = get_coordinate(list_img2, x_coor=False)
        data_target = get_target \
                (
                list_img,
                percentile_pure=100,
                separability=np.ones_like(np.array(code_class)),
                threshold_separability=0.9,
                crop='all'
            )

        if not os.path.exists(os.path.join(dir, folder_id+r'\channels_tif')):
            os.makedirs(os.path.join(dir, folder_id+r'\channels_tif'))
        # io.imsave(os.path.join(os.path.join(dir, folder_id+r'\channels_tif'), img_name),
        #           list_img2)
        if not os.path.exists(os.path.join(dir, folder_id + r'\channels_jpg')):
            os.makedirs(os.path.join(dir, folder_id + r'\channels_jpg'))
        savefig(list_img2.squeeze().astype(np.int16),
                os.path.join(dir, folder_id+r'\channels_jpg'),
                img_name.replace('tif', 'jpg'), bins_range)

        alpha_mask = np.max(list_img.squeeze(), axis=0)
        alpha = ((alpha_mask - alpha_mask.min()) / (alpha_mask.max() - alpha_mask.min()))
        # alpha[np.logical_and(alpha > 0.05, alpha < 0.1)] = alpha[np.logical_and(alpha > 0.05, alpha < 0.1)] * 1.

        fig2 = plt.figure(2, figsize=[3,3])
        cmap3 = ListedColormap(np.array([
                  [255, 255, 255],
                  [165, 112, 0],
                  [168, 0, 226],
                  [255, 211, 0],
                  [112, 38, 0],
                  [209, 255, 0],
                  [226, 0, 124],
                  [255, 255, 255]])[[0]+ code_class] / 255.0)


        plt.imshow(data_target.squeeze().sum(axis=0).astype(np.int16), cmap=cmap3, alpha=alpha*2)
        plt.grid(True)
        plt.xticks([0, 31, 63, 95, 127], np.around(np.linspace(0, bins_range[0], 5), 2))
        plt.yticks([0, 31, 63, 95, 127], np.around(np.linspace(bins_range[1], 0, 5), 2))
        ax = plt.gca()
        plt.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            left=False,
            right=False,
            # labelbottom=False,
            # labelleft=False
        )
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        for label in labels:
            label.set_fontname('Arial')
            # label.set_style('italic')
            label.set_fontsize(12)
            label.set_weight('bold')
        if not os.path.exists(os.path.join(dir, folder_id + r'\channels_target')):
            os.makedirs(os.path.join(dir, folder_id + r'\channels_target'))
        plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
        plt.savefig(os.path.join(os.path.join(dir, folder_id + r'\channels_target'), img_name.replace('.tif', '.jpg')))
        # plt.show()
        plt.cla()

def time_series(dir, folder_id, img_index, year=2019, crop_index=None, band=['R']):
    band_name = {
        "Red": 0,
        "Green": 1,
        "Blue": 2,
        "RDEG1": 0,
        "RDEG2": 4,
        "RDEG3": 5,
        "NIR": 1,
        "RDEG4": 7,
        "SWIR1": 2,
        "SWIR2": 9,
    }

    crop_name_2019 = {
        0: 'background',
        1: 'Winter wheat',
        2: 'Beets',
        5: 'Rapeseed',
        4: 'Sunflower',
        6: 'Winter Barley',
        3: 'Corn'
    }

    crop_name_2018 = {
        0: 'background',
        1: 'Winter wheat',
        2: 'Beets',
        5: 'Rapeseed',
        4: 'Sunflower',
        6: 'Winter Barley',
        3: 'Corn'
    }
    ts = 24
    if not os.path.exists(os.path.join(dir, folder_id)):
        os.makedirs(os.path.join(dir, folder_id))
    target = io.imread(root_dir+'\\visualization' + r'\\' + img_index + '.tif')
    crop_number = len(crop_index)
    crop_mean = np.zeros([crop_number, ts, 2])
    crop_std = np.zeros([crop_number, ts, 2])


    for img_name in os.listdir(os.path.join(dir, 'img')):
        for c in range(crop_number):
            i = int(img_name.split('.')[0].split('_')[-1])
            img = io.imread(os.path.join(os.path.join(dir, 'img'), img_name)) / 10000
            valid_ratio = (img != 0).mean()
            b1 = img[:, :, [band_name[band[0]]]].squeeze()
            b2 = img[:, :, [band_name[band[1]]]].squeeze()
            if valid_ratio > 0.5:
                crop_mean[c, i, 0] = b1[target == crop_index[c]].mean()
                crop_mean[c, i, 1] = b2[target == crop_index[c]].mean()
                crop_std[c, i, 0] = b1[target == crop_index[c]].std()
                crop_std[c, i, 1] = b2[target == crop_index[c]].std()
            else:
                crop_mean[c, i, 0] = float('nan')
                crop_mean[c, i, 1] = float('nan')
                crop_std[c, i, 0] = float('nan')
                crop_std[c, i, 1] = float('nan')


    new_crop_mean = np.zeros([crop_number, (ts-1) * 5, 2])
    new_crop_std = np.zeros([crop_number, (ts-1) * 5, 2])

    for i in range(0, 2):
        for c in range(crop_number):
            new_crop_mean[c, :, i] = whittaker_smooth(interpolate(smooth(crop_mean[c, :, i])), 2, d=1)
            new_crop_std[c, :, i] = whittaker_smooth(interpolate(smooth(crop_std[c, :, i])), 2, d=1)


    color = [np.array([255, 211, 0]) / 255.0,
             np.array([38, 112, 0]) / 255.0,
             np.array([87, 245, 0]) / 255.0,
             np.array([201, 211, 0]) / 255.0,
             np.array([38, 14, 0]) / 255.0,
             np.array([0, 0, 0]) / 255.0
             ]


    # fig = plt.figure(1, figsize=[3,3])
    font1 = {
        'family': 'Arial',
        'weight': 'bold',
        'size': 10
        # 'style': 'italic'
    }
    std_times = 0.7
    crop_name = crop_name_2018
    for c in range(crop_number):
        plt.plot(np.arange(0, ts), crop_mean[c, :, 0], c=color[c], label=crop_name[crop_index[c]]+'_'+band[0])
        plt.plot(np.arange(0, ts), crop_mean[c, :, 1], c=color[c], ls='--', label=crop_name[crop_index[c]]+'_'+band[1])
        plt.fill_between(np.arange(0, ts), crop_mean[c, :, 0] - crop_std[c, :, 0] * std_times, crop_mean[c, :, 0] + crop_std[c, :, 0] * std_times,
                         alpha=0.3, color=color[c])
        plt.fill_between(np.arange(0, ts), crop_mean[c, :, 1] - crop_std[c, :, 1] * std_times, crop_mean[c, :, 1] + crop_std[c, :, 1] * std_times,
                         alpha=0.3, color=color[c])


    ax = plt.gca()
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    for label in labels:
        label.set_fontname('Arial')
        # label.set_style('italic')
        label.set_fontsize(10)
        label.set_weight('bold')
    plt.tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        left=False,
        right=False,
        labelbottom=True,
        labelleft=True
    )
    ax.legend(loc=1, ##facecolor='none', edgecolor='none',
              prop={
        'family': 'Arial',
        'weight': 'bold',
        'size': 10,
        'style': 'normal'
    })
    # plt.xticks([0, 25, 50, 75, 100], ['151', '176', '201', '226', '251'])
    # plt.xlabel('DOY', font1)
    # plt.ylabel('Reflectance', font1)
    plt.ylim(0, 0.8)
    # plt.xlim(-10,120)
    # plt.yticks([0, 0.2, 0.4, 0.6, 0.8], [0, 0.2, 0.4, 0.6, 0.8]) ## NIR+SWIR1 / RDEG3/4+SWIR1/2
    plt.yticks([0, 0.15, 0.3, 0.45, 0.6], [0, 0.15, 0.3, 0.45, 0.6]) ## RDEG1+SWIR1 / Red +SWIR1
    # plt.yticks([0, 0.075, 0.15, 0.225, 0.3], [0, 0.075, 0.15, 0.225, 0.3])  ## Red+Blue

    plt.grid(True, ls='--')
    plt.subplots_adjust(left=0.1, right=0.99, top=0.99, bottom=0.1)
    plt.show()
    # plt.savefig(os.path.join(os.path.join(dir, folder_id), img_name.replace('.tif', '.jpg')))


def savefig(channels, save_dir, img_name, bins):
    gist_ncar_r = cm.get_cmap('gist_ncar_r', 256)
    cmap = gist_ncar_r(np.arange(0, 256))
    cmap[:1, :] = np.array([1, 1, 1, 1])
    cmap = ListedColormap(cmap)
    # plt.imshow(channels[0, :, :], cmap=cmap)
    plt.imshow(channels[:, :], cmap=cmap)
    plt.grid(True)
    plt.xticks([0, 31, 63, 95, 127], np.around(np.linspace(0, bins[0], 5), 2))
    plt.yticks([0, 31, 63, 95, 127], np.around(np.linspace(bins[1], 0, 5), 2))
    ax = plt.gca()
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    plt.tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        left=False,
        right=False,
        labelbottom=True,
        labelleft=False
    )
    for label in labels:
        label.set_fontname('Arial')
        # label.set_style('italic')
        label.set_fontsize(16)
        label.set_weight('bold')
    plt.savefig(os.path.join(save_dir, img_name))
    plt.cla()

def check_crop_ratio():
    dir = r'F:\DigitalAG\liheng\IW\topology_test\segmentation\target'
    item = os.listdir(dir)
    for i in item:
        if i.endswith('tif'):
            img = io.imread(os.path.join(dir, i))
            print (i)
            print ("the ratio for corn is {}".format((img==1).sum() / 512**2))
            print ("the ratio for soybean is {}".format((img == 5).sum() / 512 ** 2))
            print ('-------------------------------------')

def check_img_num(dir, mode='count'):
    img_list = os.listdir(os.path.join(dir, 'target'))
    if mode == 'count':
        for i in img_list:
            count = 0
            for f in range(0, 36):
                img = io.imread(os.path.join(dir, str(f), i))
                if (img != 0).mean() > 0.5:
                    count += 1
            if count >= 10:
                ratios = get_class_ratio(
                            io.imread(os.path.join(os.path.join(dir, 'target'), i)), [1, 2, 3, 4, 5, 6])
                print('file: {}, count: {}, winter wheat%: {}, beet%: {}, corn%:{}, sunflowers%:{}, rapeseed%:{}, barley%:{}'.format(
                                i,
                                count,
                                ratios[0] * 100,
                                ratios[1] * 100,
                                ratios[2] * 100,
                                ratios[3] * 100,
                                ratios[4] * 100,
                                ratios[5] * 100,
                                ))
    if mode == 'ratio':
        for i in img_list:
            ratios = get_class_ratio(
                io.imread(os.path.join(os.path.join(dir, 'target'), i)), [1, 2, 3, 4, 5, 6])
            print(
                'file: {}, winter wheat%: {}, beet%: {}, corn%:{}, sunflowers%:{}, rapeseed%:{}, barley%:{}'.format(
                    i,
                    ratios[0] * 100,
                    ratios[1] * 100,
                    ratios[2] * 100,
                    ratios[3] * 100,
                    ratios[4] * 100,
                    ratios[5] * 100,
                ))

def smooth(array):
    target = array.copy()
    for i in range(2, len(array)-2):
        if target[i] == 0:
            array[i] = target[i-2 : i+3].mean()
    return array

def interpolate(array):
    new_array = np.zeros((len(array) - 1) * 5)
    for i in range(0, len(new_array)):
        if i % 5 == 0:
            new_array[i] = array[i // 5]
        else:
            try:
                new_array[i] = (array[(i // 5) + 1] - array[i // 5]) / 5.0 * (i % 5) + array[i // 5]
            except:
                print (i)
    return new_array


def get_class_ratio(arr, idx_class):
    '''
    :param arr:
    :param idx_class:
    :return:
    '''
    h, w = arr.shape
    ratios = []
    for idx in idx_class:
        count = (arr == idx).sum()
        ratios.append(count / (h * w))
    return ratios

global root_dir
root_dir = r'F:\DigitalAG\liheng\EU\2018\wt_sfl_cn'
if __name__ == '__main__':
    for img in os.listdir(os.path.join(root_dir, 'segmentation', 'target')):
        save_dir = os.path.join(root_dir, 'visualization', img.split('.')[0])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        copy_image(root_dir+'\\segmentation', save_dir, img.split('.')[0], 's2')
        heat_map(save_dir, 'RDEG1_SWIR1', img.split('.')[0], ['RDEG1', 'SWIR1'])
    # time_series(root_dir+'\\visualization', 'RDEG1_NIR', '4259_1105',  crop_index=[0,3,4], band=['NIR', 'SWIR1'], year=2019)
    # check_img_num(r'F:\DigitalAG\liheng\EU\2019_early\wt_bl_rp\segmentation', 'count')
