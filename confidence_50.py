import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import os
from utility_functions import get_hist2d, get_target, get_coordinate
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def copy_image(dir, save_dir, img_index, sensor):
    import shutil
    if not os.path.exists(save_dir):
        os.makedirs((save_dir))
    if sensor == 's2':
        for i in range(0, 24):
            if os.path.exists(os.path.join(dir, str(i)) + r'\\'+img_index+'.tif'):
                shutil.copyfile(os.path.join(dir, str(i)) + r'\\'+img_index+'.tif', os.path.join(save_dir, img_index+'_'+str(i)+'.tif'))
    if sensor == 'landsat':
        for i in range(0, 8):
            if os.path.exists(os.path.join(dir, str(i)+'_ls8') + r'\\' + img_index + '.tif'):
                shutil.copyfile(os.path.join(dir, str(i)+'_ls8') + r'\\' + img_index + '.tif',
                                os.path.join(save_dir, img_index + '_' + str(i) + '.tif'))
    if os.path.exists(os.path.join(dir, 'target') + r'\\'+img_index+'.tif'):
        shutil.copyfile(os.path.join(dir, 'target') + r'\\'+img_index+'.tif', os.path.join(save_dir, img_index+'.tif'))

def patch_time_series(dir, folder_id):
    if not os.path.exists(os.path.join(dir, folder_id)):
        os.makedirs(os.path.join(dir, folder_id))
    target = io.imread(root_dir + r'\2.tif')
    for img_name in os.listdir(os.path.join(dir, 'img')):
        new_img_name = index_date_coversion(img_name.split('.')[0].split('_')[0])
        img = io.imread(os.path.join(os.path.join(dir, 'img'), img_name)) / 10000
        img = img[:,:]
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
        code_class = [0, 1]
        bins_range = [0.3, 0.6]
        arr_class = np.zeros_like(target)
        for i_cur, code_cur in enumerate(code_class):
            arr_class[target == code_cur] = i_cur + 1
        list_img, list_xedges, list_yedges = get_hist2d(img, bins=257, arr_class=arr_class, bins_range=bins_range)
        list_img2, list_xedges2, list_yedges2 = get_hist2d(img, bins=257, bins_range=bins_range)
        # mask_img, mask_xedges, mask_yedges = get_hist2d(img, arr_class=(mask != 0).astype(np.int8), bins_range=bins_range)
        x_coor = get_coordinate(list_img2)
        y_coor = get_coordinate(list_img2, x_coor=False)
        data_target = get_target \
                (
                list_img,
                percentile_pure=50,
                separability=np.array([1,1,1]),
                threshold_separability=0.9,
                crop='all'
            )

        if not os.path.exists(os.path.join(dir, folder_id+r'\channels_tif')):
            os.makedirs(os.path.join(dir, folder_id+r'\channels_tif'))
        # io.imsave(os.path.join(os.path.join(dir, folder_id+r'\channels_tif'), img_name),
        #           list_img2)
        if not os.path.exists(os.path.join(dir, folder_id + r'\channels_jpg')):
            os.makedirs(os.path.join(dir, folder_id + r'\channels_jpg'))
        # savefig(list_img2.squeeze().astype(np.int16),
        #         os.path.join(dir, folder_id+r'\channels_jpg'),
        #         img_name.replace('tif', 'jpg'), bins_range)

        alpha_mask = np.max(list_img.squeeze(), axis=0)
        alpha = ((alpha_mask - alpha_mask.min()) / (alpha_mask.max() - alpha_mask.min()))
        # alpha[np.logical_and(alpha > 0.05, alpha < 0.1)] = alpha[np.logical_and(alpha > 0.05, alpha < 0.1)] * 1.5
        cmap = ListedColormap(np.array([[0, 0, 0],
                                        [0, 169, 230]
                                        [255, 211, 0],
                                        # [38, 112, 0],
                                        [255, 255, 255]
        ]) / 255.0)

        fig2 = plt.figure(2)
        ax = fig2.add_subplot(111, projection='3d')
        X = np.linspace(0, 257, 256)
        Y = np.linspace(0, 257, 256)
        X, Y = np.meshgrid(X, Y)
        mask = data_target.squeeze().sum(axis=0)
        mask[mask!=0] = 1
        corn = list_img.squeeze()[1, :, :]
        corn /= corn.max()
        # corn *= mask
        corn[corn <= 0] = float('nan')
        soybean = list_img.squeeze()[2, :, :]
        soybean /= soybean.max()
        # soybean *= mask
        soybean[soybean <= 0] = float('nan')
        # ax.plot_surface(X, Y, corn, color=np.array([255, 211, 0])/255.0, alpha=0.1, edgecolor='None')
        # ax.plot_surface(X, Y, soybean, color=np.array([38, 112, 0])/255.0, alpha=0.1, edgecolor='None')
        ax.plot_surface(X, Y, corn, color=np.array([255, 211, 0])/255.0, alpha=0.6, edgecolor='None')
        ax.plot_surface(X, Y, soybean, color=np.array([38, 112, 0])/255.0, alpha=0.6, edgecolor='None')
        # ax.xaxis._axinfo["grid"].update({"linewidth": 0})
        # plt.xticks([0, 31, 63, 95, 127], np.around(np.linspace(0, bins_range[0], 5), 2))
        # plt.yticks([0, 31, 63, 95, 127], np.around(np.linspace(bins_range[1], 0, 5),2))
        plt.xticks([0, 63, 127, 191, 255], np.around(np.linspace(0, bins_range[1], 5), 2))
        plt.yticks([0, 63, 127, 191, 255], np.around(np.linspace(bins_range[0], 0,  5),2))
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        ax.zaxis.set_tick_params(labelsize=10)
        ax.set_zticks([0, 0.2,0.4,0.6,0.8,1.0])
        for label in labels:
            label.set_fontname('Arial')
            # label.set_style('italic')
            label.set_fontsize(12)
            # label.set_weight('bold')
        if not os.path.exists(os.path.join(dir, folder_id + r'\channels_target')):
            os.makedirs(os.path.join(dir, folder_id + r'\channels_target'))
        # plt.savefig(os.path.join(os.path.join(dir, folder_id+r'\channels_target'), img_name.replace('.tif', '.jpg')))
        # plt.savefig(os.path.join(os.path.join(dir, 'time_series_hist'), img_name.replace('.tif', '_bin11.png')))
        plt.show()
        plt.cla()

def patch_time_series_2d(dir, folder_id, img_index):
    if not os.path.exists(os.path.join(dir, folder_id)):
        os.makedirs(os.path.join(dir, folder_id))
    target = io.imread(root_dir + r'\\' + img_index +'.tif')
    for img_name in os.listdir(os.path.join(dir, 'img')):
        new_img_name = index_date_coversion(img_name.split('.')[0].split('_')[2])
        img = io.imread(os.path.join(os.path.join(dir, 'img'), img_name)) / 10000
        img = img[:,:]
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
        code_class = [1, 2]
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
                separability=np.array([1,1,1]),
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
                new_img_name.replace('tif', 'jpg'), bins_range)

        alpha_mask = np.max(list_img.squeeze(), axis=0)
        alpha = ((alpha_mask - alpha_mask.min()) / (alpha_mask.max() - alpha_mask.min()))
        # alpha[np.logical_and(alpha > 0.05, alpha < 0.1)] = alpha[np.logical_and(alpha > 0.05, alpha < 0.1)] * 1.5


        fig2 = plt.figure(2)
        cmap2 = ListedColormap(np.array([[255, 255, 255],
                                        [255, 211, 0],
                                        [38, 112, 0],
                                        [0,0,0]
        ]) / 255.0)
        idx_max = np.argmax(list_img, axis=1).squeeze()
        diff = list_img.squeeze()[1, :, :] - list_img.squeeze()[2, :, :]
        mask2 = np.abs(diff) < 100
        mask = data_target.squeeze().sum(axis=0) * (idx_max!=0)
        mask[mask!=0] = 1
        corn = list_img.squeeze()[1, :, :]
        corn[corn != 0] = 1
        soybean = list_img.squeeze()[2, :, :]
        soybean[soybean != 0] = 1
        overlap = ((corn * soybean * mask * mask2)!=0).astype(np.int)
        img = data_target.squeeze().sum(axis=0)
        img[overlap == 1] = 3
        plt.imshow(img, alpha=1, cmap=cmap2)
        plt.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False
        )
        plt.xticks([0, 31, 63, 95, 127], np.around(np.linspace(0, bins_range[0], 5), 3))
        plt.yticks([0, 31, 63, 95, 127], np.around(np.linspace(bins_range[1], 0, 5),2))
        ax = plt.gca()
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        for label in labels:
            label.set_fontname('Arial')
            # label.set_style('italic')
            label.set_fontsize(12)
            # label.set_weight('bold')
        if not os.path.exists(os.path.join(dir, folder_id + r'\channels_target')):
            os.makedirs(os.path.join(dir, folder_id + r'\channels_target'))
        # plt.savefig(os.path.join(os.path.join(dir, folder_id+r'\channels_target'), img_name.replace('.tif', '.jpg')))
        # plt.savefig(os.path.join(os.path.join(dir, 'time_series_hist'), img_name.replace('.tif', '_bin11.png')))
        plt.grid()
        # plt.show()
        plt.cla()

        fig3 = plt.figure(3, figsize=[4,4])
        cmap3 = ListedColormap(np.array([[0,0,0],
                                         # [0, 169, 230],
                                        [255, 211, 0],
                                        [38, 112, 0],
                                        [255, 255, 255]

        ]) / 255.0)
        plt.imshow(data_target.squeeze().sum(axis=0).astype(np.int16), cmap=cmap3, alpha=alpha*2)
        plt.grid(True)
        # plt.xticks([0, 31, 63, 95, 127], np.around(np.linspace(0, bins_range[0], 5), 3))
        plt.xticks([0, 31, 63, 95, 127], ['0', '0.075', '0.15', '0.225', '0.3'])
        # plt.yticks([0, 31, 63, 95, 127], np.around(np.linspace(bins_range[1], 0, 5), 2))
        plt.yticks([0, 31, 63, 95, 127], ['0.6', '0.45', '0.3', '0.15', '0'])
        ax = plt.gca()
        plt.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False
        )
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        for label in labels:
            label.set_fontname('Arial')
            # label.set_style('italic')
            label.set_fontsize(16)
            label.set_weight('bold')
        if not os.path.exists(os.path.join(dir, folder_id + r'\channels_target')):
            os.makedirs(os.path.join(dir, folder_id + r'\channels_target'))
        plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
        plt.savefig(os.path.join(os.path.join(dir, folder_id + r'\channels_target'), new_img_name.replace('.tif', '.jpg')))
        # plt.show()
        plt.cla()

def index_date_coversion(index):
    conversion_dict = {
        '0': '0601', '1': '0606', '2': '0611', '3': '0616', '4': '0621', '5': '0626',
        '6': '0701', '7': '0706', '8': '0711', '9': '0716', '10': '0721', '11': '0726',
        '12': '0801', '13': '0806', '14': '0811', '15': '0816', '16': '0821', '17': '0826',
        '18': '0901', '19': '0906', '20': '0911', '21': '0916', '22': '0921', '23': '0926'
    }
    return conversion_dict[index]+'.tif'

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
        # labelbottom=False,
        # labelleft=False
    )
    for label in labels:
        label.set_fontname('Arial')
        # label.set_style('italic')
        label.set_fontsize(16)
        label.set_weight('bold')
    plt.savefig(os.path.join(save_dir, img_name))
    plt.cla()

def check_crop_ratio():
    # dir = r"F:\DigitalAG\liheng\IW\sentinel-2\NIR\2019\target"
    dir = r"F:\DigitalAG\liheng\Northeast\2019_new\segmentation\target"
    item = os.listdir(dir)
    for i in item:
        if i.endswith('tif'):
            img = io.imread(os.path.join(dir, i))
            if (img==2).sum() / 512**2 >= 0.4:
                print (i)
                print ("the ratio for corn is {}".format((img==1).sum() / 512 ** 2))
                print ("the ratio for soybean is {}".format((img == 2).sum() / 512 ** 2))
                print ('-------------------------------------')


global root_dir
root_dir = r'F:\DigitalAG\liheng\Northeast\visualization_NE'
if __name__ == '__main__':
    # check_crop_ratio()
    patch_time_series_2d(root_dir, 'time_series', '10099_6011')
    # copy_image(r'F:\DigitalAG\liheng\Northeast\2019_new\segmentation', root_dir, '10099_6011', 's2')