import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from skimage import io

def sample_map_accuracy_winter(dir, sensor='landsat', target_year='2018'):
    '''
    :param arr:
    :return:
    '''
    data = pd.read_csv(dir)
    crops = ['winter wheat','rapeseed', 'barley']
    crop_name = ['winter wheat','rapeseed', 'winter barley']
    precision = np.zeros([18, 3])
    number = np.zeros([18, 3])
    for idx in range(len(crops)):
        precision[:, idx] = np.array(data[crops[idx] + '_precision'])
        number[:, idx] = np.array(data[crops[idx] + '_number'])

    font1 = {
        'family': 'Arial',
        'weight': 'bold',
        'size': 10,
        'color': 'k'
    }

    n = 19
    xlim = [-1, 22]
    xticks = [[1, 6, 11, 16, 21], ['91', '116', '141', '166', '191']]
    ylimits = [[-60000, 660000],
               [-20000, 220000],
               [-2000, 22000]]
    yticks = [[[0, 300000, 600000], [0, 3.0, 6.0]],
              [[0, 100000, 200000], [0, 1.0, 2.0]],
              [[0, 10000, 20000], [0, 1.0, 2.0]]]
    y_mag = [r'($\times{10^5}$)',
             r'($\times{10^5}$)',
             r'($\times{10^4}$)']
    x_label_bool = [False, False, True]

    fig = plt.figure(1, figsize=(5.5, 7))
    c = [np.array([165, 112, 0, 255]) / 255.0,
          np.array([209, 255, 0, 255]) / 255.0,
          np.array([226, 0, 124, 255]) / 255.0]

    ## subplot 1
    for i in range(len(crops)):
        ax1 = plt.subplot(3, 1, i+1)
        # plt.rcParams["figure.titleweight"] = 'normal'
        # plt.rcParams["font.family"] = 'Arial'
        # plt.rcParams["font.size"] = '10'
        ax2 = ax1.twinx()
        crop_number = number[:, i]
        crop_pre = precision[:, i]
        ax2.bar(np.arange(1, n)[crop_number != 0], crop_number[crop_number != 0], width=0.8, color=c[i], edgecolor='k',
                label='PA', alpha=1)
        ax1.bar(np.arange(1, n), crop_pre, width=0.8, color=c[i],
                label='UA', alpha=0.6)
        labels1 = ax1.get_xticklabels() + ax1.get_yticklabels() + ax2.get_yticklabels()
        for label in labels1:
            label.set_fontname('Arial')
            label.set_fontsize(9)
            label.set_weight('bold')
        # ax1.legend(loc=1, facecolor='none', edgecolor='none', prop={
        #     'family': 'Arial',
        #     'weight': 'bold',
        #     'size': 8,
        #     'style': 'normal'
        # })

        ax1.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            left=False,
            right=False,
            labelbottom=x_label_bool[i]
            )
        ax2.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            left=False,
            right=False,
            labelbottom=False
            )

        ax1.set_xlim(xlim[0], xlim[1])
        ax1.set_ylim(-0.1, 1.1)
        ax2.set_ylim(ylimits[i][0], ylimits[i][1])
        ax1.set_yticks([0, 0.5, 1])
        ax1.set_xticks(xticks[0])
        ax1.set_xticklabels(xticks[1])
        ax2.set_yticks(yticks[i][0])
        ax2.set_yticklabels(yticks[i][1])
        ax1.set_ylabel('Agreement', font1)
        ax2.set_ylabel(r'Number of labels' + y_mag[i], font1)
        ax1.grid(linestyle='--', lw=0.5)
    plt.subplots_adjust(hspace=0.05, left=0.11, top=0.98, bottom=0.13)
    plt.show()

def sample_map_accuracy_summer(dir, sensor='landsat', target_year='2018'):
    '''
    :param arr:
    :return:
    '''
    data = pd.read_csv(dir)
    crops = ['beets', 'corn', 'sunflowers']
    precision = np.zeros([24, 3])
    number = np.zeros([24, 3])
    for idx in range(len(crops)):
        precision[:, idx] = np.array(data[crops[idx] + '_precision'])
        number[:, idx] = np.array(data[crops[idx] + '_number'])

    font1 = {
        'family': 'Arial',
        'weight': 'bold',
        'size': 10,
        'color': 'k'
    }

    n = 25
    xlim = [-1, 28]
    xticks = [[1, 6, 11, 16, 21, 26], ['151', '176', '201', '226', '251', '276']]

    ylimits = [
               [-40000, 440000],
               [-2000, 22000],
               [-3000, 33000]
              ]
    yticks = [[[0, 200000, 400000], [0, 2.0, 4.0]],
              [[0, 10000, 20000], [0, 1.0, 2.0]],
                [[0, 15000, 30000], [0, 1.5, 3.0]]]
    y_mag = [r'($\times{10^5}$)',
             r'($\times{10^3}$)',
             r'($\times{10^4}$)']
    x_label_bool = [False, False, True]

    fig = plt.figure(1, figsize=(5.5, 7))
    c = [np.array([168, 0, 226, 255]) / 255.0,
                  np.array([255, 211, 0, 255]) / 255.0,
                  np.array([112, 38, 0, 255]) / 255.0]

    ## subplot 1
    for i in range(len(crops)):
        ax1 = plt.subplot(3, 1, i + 1)
        # plt.rcParams["figure.titleweight"] = 'normal'
        # plt.rcParams["font.family"] = 'Arial'
        # plt.rcParams["font.size"] = '10'
        ax2 = ax1.twinx()
        crop_number = number[:, i]
        crop_pre = precision[:, i]
        ax2.bar(np.arange(1, n)[crop_number != 0], crop_number[crop_number != 0], width=0.8, color=c[i],
                edgecolor='k',
                label='PA', alpha=1)
        ax1.bar(np.arange(1, n), crop_pre, width=0.8, color=c[i],
                label='UA', alpha=0.6)
        labels1 = ax1.get_xticklabels() + ax1.get_yticklabels() + ax2.get_yticklabels()
        for label in labels1:
            label.set_fontname('Arial')
            label.set_fontsize(9)
            label.set_weight('bold')
        # ax1.legend(loc=1, facecolor='none', edgecolor='none', prop={
        #     'family': 'Arial',
        #     'weight': 'bold',
        #     'size': 8,
        #     'style': 'normal'
        # })

        ax1.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            left=False,
            right=False,
            labelbottom=x_label_bool[i]
        )
        ax2.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            left=False,
            right=False,
            labelbottom=False
        )

        ax1.set_xlim(xlim[0], xlim[1])
        ax1.set_ylim(-0.1, 1.1)
        ax2.set_ylim(ylimits[i][0], ylimits[i][1])
        ax1.set_yticks([0, 0.5, 1])
        ax1.set_xticks(xticks[0])
        ax1.set_xticklabels(xticks[1])
        ax2.set_yticks(yticks[i][0])
        ax2.set_yticklabels(yticks[i][1])
        ax1.set_ylabel('Agreement', font1)
        ax2.set_ylabel(r'Number of labels' + y_mag[i], font1)
        ax1.grid(linestyle='--', lw=0.5)
    plt.subplots_adjust(hspace=0.05, left=0.11, top=0.98, bottom=0.13)
    plt.show()

def crop_type_mapping(legend=True):
    font1 = {
        'family': 'Arial',
        'weight': 'bold',
        'size': 14,
        'color': 'k'
    }

    n = 25
    xlim = [-3, 30]
    xticks = [[1, 6, 11, 16, 21, 26], ['151', '176', '201', '226', '251', '276']]

    # Create some mock data
    fig = plt.figure(1, figsize=(10, 4), facecolor='w')
    plt.subplot(1,2,1)
    data1 = pd.read_csv(r'F:\DigitalAG\liheng\EU\csv\classification_points_seed70.csv')
    crops1 = ['winter wheat', 'beets', 'corn', 'potato', 'rapeseed', 'barley']
    crops1_labels = ['Winter Wheat F1', 'Beet F1', 'Corn F1', 'Potato F1', 'Rapeseed F1', 'Winter Barley F1']

    f1 = np.zeros([24, 6])
    for idx in range(len(crops1)):
        f1[:, idx] = np.array(data1[crops1[idx] + ' f1'])
    oa = np.array(data1['oa'])
    colors = [np.array([165, 112, 0, 255]) / 255.0,
              np.array([168, 0, 226, 255]) / 255.0,
              np.array([255, 211, 0, 255]) / 255.0,
              np.array([112, 38, 0, 255]) / 255.0,
              np.array([209, 255, 0, 255]) / 255.0,
              np.array([226, 0, 124, 255]) / 255.0]

    for idx in range(len(crops1)):
        plt.plot(np.arange(1, n), f1[:, idx], c=colors[idx], label=crops1_labels[idx], lw=1.5, ms=6, marker='.')
    plt.plot(np.arange(1, n), oa, c='k', label='Overall Accuracy', lw=1.5, ms=6, marker='.')
    plt.plot(np.arange(xlim[0], xlim[1] + 1), np.repeat(0.855, xlim[1] - xlim[0] + 1), lw=1, c=colors[0], ls='--')
    plt.plot(np.arange(xlim[0], xlim[1] + 1), np.repeat(0.815, xlim[1] - xlim[0] + 1), lw=1, c=colors[1], ls='--')
    plt.plot(np.arange(xlim[0], xlim[1] + 1), np.repeat(0.734, xlim[1] - xlim[0] + 1), lw=1, c=colors[2], ls='--')
    plt.plot(np.arange(xlim[0], xlim[1] + 1), np.repeat(0.690, xlim[1] - xlim[0] + 1), lw=1, c=colors[3], ls='--')
    plt.plot(np.arange(xlim[0], xlim[1] + 1), np.repeat(0.807, xlim[1] - xlim[0] + 1), lw=1, c=colors[4], ls='--')
    plt.plot(np.arange(xlim[0], xlim[1] + 1), np.repeat(0.699, xlim[1] - xlim[0] + 1), lw=1, c=colors[5], ls='--')
    plt.plot(np.arange(xlim[0], xlim[1] + 1), np.repeat(0.862, xlim[1] - xlim[0] + 1), lw=1, c='k', ls='--')
    ax1 = plt.gca()
    ax1.set_facecolor('w')
    labels = ax1.get_xticklabels() + ax1.get_yticklabels()
    ax1.set_xlabel('DOY', font1)
    ax1.set_ylabel('Accuracy', font1)
    # ax1.set_ylabel('Accuracy', font1)
    for label in labels:
        label.set_fontname('Arial')
        label.set_fontsize(12)
        label.set_weight('bold')
    plt.tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        left=False,
        right=False,
        labelleft=True,
        labelbottom=True,
        # colors='white'
    )
    if legend:
        ax1.legend(loc=4, facecolor='none', edgecolor='none', labelcolor='k', prop={
            'family': 'Arial',
            'weight': 'bold',
            'size': 10,
            'style': 'normal',
        })
    plt.grid(linestyle='--', lw=0.5)
    plt.xlim(xlim[0], xlim[1])
    plt.xticks(xticks[0], xticks[1])
    plt.ylim(-0.1, 1.1)

    plt.subplot(1, 2, 2)
    data2 = pd.read_csv(r'F:\DigitalAG\liheng\EU\csv\classification_theia.csv')
    crops2 = ['cereals', 'oilseeds', 'corn', 'tubers']
    crops2_labels = ['Straw Cereals F1', 'Oilseeds F1', 'Corn F1', 'Tubers/Roots F1']

    f1 = np.zeros([24, 4])
    for idx in range(len(crops2)):
        f1[:, idx] = np.array(data2[crops2[idx] + ' f1'])
    oa = np.array(data2['oa'])
    colors = [np.array([165, 112, 0, 255]) / 255.0,
              np.array([209, 255, 0, 255]) / 255.0,
              np.array([255, 211, 0, 255]) / 255.0,
              np.array([168, 0, 226, 255]) / 255.0]

    for idx in range(len(crops2)):
        plt.plot(np.arange(1, n), f1[:, idx], c=colors[idx], label=crops2_labels[idx], lw=1.5, ms=6, marker='.')
    plt.plot(np.arange(1, n), oa, c='k', label='Overall Accuracy', lw=1.5, ms=6, marker='.')
    plt.plot(np.arange(xlim[0], xlim[1] + 1), np.repeat(0.875, xlim[1] - xlim[0] + 1), lw=1, c=colors[0], ls='--')
    plt.plot(np.arange(xlim[0], xlim[1] + 1), np.repeat(0.740, xlim[1] - xlim[0] + 1), lw=1, c=colors[1], ls='--')
    plt.plot(np.arange(xlim[0], xlim[1] + 1), np.repeat(0.747, xlim[1] - xlim[0] + 1), lw=1, c=colors[2], ls='--')
    plt.plot(np.arange(xlim[0], xlim[1] + 1), np.repeat(0.733, xlim[1] - xlim[0] + 1), lw=1, c=colors[3], ls='--')
    plt.plot(np.arange(xlim[0], xlim[1] + 1), np.repeat(0.838, xlim[1] - xlim[0] + 1), lw=1, c='k', ls='--')
    ax1 = plt.gca()
    ax1.set_facecolor('w')
    labels = ax1.get_xticklabels() + ax1.get_yticklabels()
    ax1.set_xlabel('DOY', font1)
    # ax1.set_ylabel('Accuracy', font1)
    for label in labels:
        label.set_fontname('Arial')
        label.set_fontsize(12)
        label.set_weight('bold')
    plt.tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        left=False,
        right=False,
        labelleft=False,
        labelbottom=True,
        # colors='white'
    )
    if legend:
        ax1.legend(loc=4, facecolor='none', edgecolor='none', labelcolor='k', prop={
            'family': 'Arial',
            'weight': 'bold',
            'size': 10,
            'style': 'normal',
        })
    plt.grid(linestyle='--', lw=0.5)
    plt.xlim(xlim[0], xlim[1])
    plt.xticks(xticks[0], xticks[1])
    plt.ylim(-0.1, 1.1)

    plt.subplots_adjust(top=0.95, bottom=0.15, right=0.95, left=0.07, wspace=0.05)
    plt.show()
    # plt.savefig(dir.replace('csv', 'figures', 1).replace('csv', 'jpg'))



def compareTwoCombination(dir, legend=True):
    data = pd.read_csv(dir)
    crops = ['winter wheat', 'beets', 'corn', 'sunflowers', 'rapeseed', 'barley']
    f1 = np.zeros([24, 6])
    for idx in range(len(crops)):
        f1[:, idx] = np.array(data[crops[idx] + ' f1'])
    oa = np.array(data['oa'])


    font1 = {
        'family': 'Arial',
        'weight': 'bold',
        'size': 16,
        'color':'k'
    }

    n = 25
    xlim = [-3, 30]
    xticks = [[1, 6, 11, 16, 21, 26], ['151', '176', '201', '226', '251', '276']]

    # Create some mock data
    fig = plt.figure(1, facecolor='w')
    fig.set_size_inches(8, 4)
    c1 = np.array([205, 22, 28, 255]) / 255.0
    c2 = np.array([18, 111, 168, 255]) / 255.0
    # c1 = np.array([230, 75, 53, 255]) / 255.0
    # c2 = np.array([77, 187, 213, 255]) / 255.0
    c3 = np.array([0, 160, 135, 255]) / 255.0
    c4 = np.array([60, 84, 136, 255]) / 255.0
    c5 = np.array([243, 155, 127, 255]) / 255.0
    c6 = np.array([132,145,180, 255]) / 255.0
    c7 = np.array([145,209,194, 255]) / 255.0
    c8 = np.array([220,0,0, 255]) / 255.0
    c9 = np.array([126,97,72, 255]) / 255.0
    c10 = np.array([176,156,133, 255]) / 255.0
    color_list = [c1, c2, c3, c4, c5, c6, c7, c8, c9, c10]

    for idx in range(len(crops)):
        plt.plot(np.arange(1, n), f1[:, idx], c=color_list[idx], label=crops[idx], lw=1.5, ms=6, marker='.')
    ax1 = plt.gca()
    ax1.set_facecolor('w')
    labels = ax1.get_xticklabels() + ax1.get_yticklabels()
    ax1.set_xlabel('DOY', font1)
    ax1.set_ylabel('Accuracy', font1)
    # ax1.set_ylabel('Accuracy', font1)
    for label in labels:
        label.set_fontname('Arial')
        label.set_fontsize(12)
        label.set_weight('bold')
    plt.tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        left=False,
        right=False,
        labelleft=True,
        labelbottom=True,
        # colors='white'
    )
    if legend:
        ax1.legend(loc=4, facecolor='none', edgecolor='none', labelcolor='k', prop={
            'family': 'Arial',
            'weight': 'bold',
            'size': 12,
            'style': 'normal',
        })
    plt.grid(linestyle='--', lw=0.5)
    plt.xlim(xlim[0], xlim[1])
    plt.xticks(xticks[0], xticks[1])
    plt.ylim(-0.1, 1.1)

    plt.subplots_adjust(top=0.95, bottom=0.15, right=0.95, left=0.1)
    # plt.show()
    plt.savefig(dir.replace('csv', 'figures', 1).replace('csv', 'jpg'))

def visualize_error_map(com_dir='', om_dir=''):
    if com_dir != '':
        com = io.imread(com_dir).astype(np.int16)
        com[com==-1] = 100
        bin = np.bincount(com.ravel())
        sort = np.argsort(bin)[-6:-1]
        fig1 = plt.figure(1, figsize=[6,6])
        plt.bar(sort, bin[sort])
        plt.show()

def cal_magnitude(number):
    magnitude = 0
    while number/10.0 > 1:
        magnitude += 1
        number /= 10.0
    return magnitude

def crop_type_mapping_gt(legend=True):
    font1 = {
        'family': 'Arial',
        'weight': 'bold',
        'size': 14,
        'color': 'k'
    }

    n = 25
    xlim = [-3, 30]
    xticks = [[1, 6, 11, 16, 21, 26], ['151', '176', '201', '226', '251', '276']]

    # Create some mock data
    fig = plt.figure(1, figsize=(10, 8), facecolor='w')
    plt.subplot(2,2,1)
    data1 = pd.read_csv(r'F:\DigitalAG\liheng\EU\csv\classification_gt_500_25.csv')
    crops1 = ['winter wheat', 'beets', 'corn', 'potato', 'rapeseed', 'barley']
    crops1_labels = ['Winter Wheat F1', 'Beet F1', 'Corn F1', 'Potato F1', 'Rapeseed F1', 'Winter Barley F1']

    f1 = np.zeros([24, 6])
    for idx in range(len(crops1)):
        f1[:, idx] = np.array(data1[crops1[idx] + ' f1'])
    oa = np.array(data1['oa'])
    colors = [np.array([165, 112, 0, 255]) / 255.0,
              np.array([168, 0, 226, 255]) / 255.0,
              np.array([255, 211, 0, 255]) / 255.0,
              np.array([112, 38, 0, 255]) / 255.0,
              np.array([209, 255, 0, 255]) / 255.0,
              np.array([226, 0, 124, 255]) / 255.0]

    for idx in range(len(crops1)):
        plt.plot(np.arange(1, n), f1[:, idx], c=colors[idx], label=crops1_labels[idx], lw=1.5, ms=6, marker='.')
    plt.plot(np.arange(1, n), oa, c='k', label='Overall Accuracy', lw=1.5, ms=6, marker='.')
    plt.plot(np.arange(xlim[0], xlim[1] + 1), np.repeat(0.855, xlim[1] - xlim[0] + 1), lw=1, c=colors[0], ls='--')
    plt.plot(np.arange(xlim[0], xlim[1] + 1), np.repeat(0.815, xlim[1] - xlim[0] + 1), lw=1, c=colors[1], ls='--')
    plt.plot(np.arange(xlim[0], xlim[1] + 1), np.repeat(0.734, xlim[1] - xlim[0] + 1), lw=1, c=colors[2], ls='--')
    plt.plot(np.arange(xlim[0], xlim[1] + 1), np.repeat(0.690, xlim[1] - xlim[0] + 1), lw=1, c=colors[3], ls='--')
    plt.plot(np.arange(xlim[0], xlim[1] + 1), np.repeat(0.807, xlim[1] - xlim[0] + 1), lw=1, c=colors[4], ls='--')
    plt.plot(np.arange(xlim[0], xlim[1] + 1), np.repeat(0.699, xlim[1] - xlim[0] + 1), lw=1, c=colors[5], ls='--')
    plt.plot(np.arange(xlim[0], xlim[1] + 1), np.repeat(0.862, xlim[1] - xlim[0] + 1), lw=1, c='k', ls='--')
    ax1 = plt.gca()
    ax1.set_facecolor('w')
    labels = ax1.get_xticklabels() + ax1.get_yticklabels()
    ax1.set_ylabel('Accuracy', font1)
    # ax1.set_ylabel('Accuracy', font1)
    for label in labels:
        label.set_fontname('Arial')
        label.set_fontsize(12)
        label.set_weight('bold')
    plt.tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        left=False,
        right=False,
        labelleft=True,
        labelbottom=False,
        # colors='white'
    )
    # if legend:
    #     ax1.legend(loc=4, facecolor='none', edgecolor='none', labelcolor='k', prop={
    #         'family': 'Arial',
    #         'weight': 'bold',
    #         'size': 10,
    #         'style': 'normal',
    #     })
    plt.grid(linestyle='--', lw=0.5)
    plt.xlim(xlim[0], xlim[1])
    plt.xticks(xticks[0], xticks[1])
    plt.ylim(-0.1, 1.1)

    plt.subplot(2, 2, 2)
    data1 = pd.read_csv(r'F:\DigitalAG\liheng\EU\csv\classification_gt_500_50.csv')
    crops1 = ['winter wheat', 'beets', 'corn', 'potato', 'rapeseed', 'barley']
    crops1_labels = ['Winter Wheat F1', 'Beet F1', 'Corn F1', 'Potato F1', 'Rapeseed F1', 'Winter Barley F1']

    f1 = np.zeros([24, 6])
    for idx in range(len(crops1)):
        f1[:, idx] = np.array(data1[crops1[idx] + ' f1'])
    oa = np.array(data1['oa'])
    colors = [np.array([165, 112, 0, 255]) / 255.0,
              np.array([168, 0, 226, 255]) / 255.0,
              np.array([255, 211, 0, 255]) / 255.0,
              np.array([112, 38, 0, 255]) / 255.0,
              np.array([209, 255, 0, 255]) / 255.0,
              np.array([226, 0, 124, 255]) / 255.0]

    for idx in range(len(crops1)):
        plt.plot(np.arange(1, n), f1[:, idx], c=colors[idx], label=crops1_labels[idx], lw=1.5, ms=6, marker='.')
    plt.plot(np.arange(1, n), oa, c='k', label='Overall Accuracy', lw=1.5, ms=6, marker='.')
    plt.plot(np.arange(xlim[0], xlim[1] + 1), np.repeat(0.855, xlim[1] - xlim[0] + 1), lw=1, c=colors[0], ls='--')
    plt.plot(np.arange(xlim[0], xlim[1] + 1), np.repeat(0.815, xlim[1] - xlim[0] + 1), lw=1, c=colors[1], ls='--')
    plt.plot(np.arange(xlim[0], xlim[1] + 1), np.repeat(0.734, xlim[1] - xlim[0] + 1), lw=1, c=colors[2], ls='--')
    plt.plot(np.arange(xlim[0], xlim[1] + 1), np.repeat(0.690, xlim[1] - xlim[0] + 1), lw=1, c=colors[3], ls='--')
    plt.plot(np.arange(xlim[0], xlim[1] + 1), np.repeat(0.807, xlim[1] - xlim[0] + 1), lw=1, c=colors[4], ls='--')
    plt.plot(np.arange(xlim[0], xlim[1] + 1), np.repeat(0.699, xlim[1] - xlim[0] + 1), lw=1, c=colors[5], ls='--')
    plt.plot(np.arange(xlim[0], xlim[1] + 1), np.repeat(0.862, xlim[1] - xlim[0] + 1), lw=1, c='k', ls='--')
    ax1 = plt.gca()
    ax1.set_facecolor('w')
    labels = ax1.get_xticklabels() + ax1.get_yticklabels()
    # ax1.set_ylabel('Accuracy', font1)
    for label in labels:
        label.set_fontname('Arial')
        label.set_fontsize(12)
        label.set_weight('bold')
    plt.tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        left=False,
        right=False,
        labelleft=False,
        labelbottom=False,
        # colors='white'
    )
    # if legend:
    #     ax1.legend(loc=4, facecolor='none', edgecolor='none', labelcolor='k', prop={
    #         'family': 'Arial',
    #         'weight': 'bold',
    #         'size': 10,
    #         'style': 'normal',
    #     })
    plt.grid(linestyle='--', lw=0.5)
    plt.xlim(xlim[0], xlim[1])
    plt.xticks(xticks[0], xticks[1])
    plt.ylim(-0.1, 1.1)

    plt.subplot(2, 2, 3)
    data1 = pd.read_csv(r'F:\DigitalAG\liheng\EU\csv\classification_gt_500_75.csv')
    crops1 = ['winter wheat', 'beets', 'corn', 'potato', 'rapeseed', 'barley']
    crops1_labels = ['Winter Wheat F1', 'Beet F1', 'Corn F1', 'Potato F1', 'Rapeseed F1', 'Winter Barley F1']

    f1 = np.zeros([24, 6])
    for idx in range(len(crops1)):
        f1[:, idx] = np.array(data1[crops1[idx] + ' f1'])
    oa = np.array(data1['oa'])
    colors = [np.array([165, 112, 0, 255]) / 255.0,
              np.array([168, 0, 226, 255]) / 255.0,
              np.array([255, 211, 0, 255]) / 255.0,
              np.array([112, 38, 0, 255]) / 255.0,
              np.array([209, 255, 0, 255]) / 255.0,
              np.array([226, 0, 124, 255]) / 255.0]

    for idx in range(len(crops1)):
        plt.plot(np.arange(1, n), f1[:, idx], c=colors[idx], label=crops1_labels[idx], lw=1.5, ms=6, marker='.')
    plt.plot(np.arange(1, n), oa, c='k', label='Overall Accuracy', lw=1.5, ms=6, marker='.')
    plt.plot(np.arange(xlim[0], xlim[1] + 1), np.repeat(0.855, xlim[1] - xlim[0] + 1), lw=1, c=colors[0], ls='--')
    plt.plot(np.arange(xlim[0], xlim[1] + 1), np.repeat(0.815, xlim[1] - xlim[0] + 1), lw=1, c=colors[1], ls='--')
    plt.plot(np.arange(xlim[0], xlim[1] + 1), np.repeat(0.734, xlim[1] - xlim[0] + 1), lw=1, c=colors[2], ls='--')
    plt.plot(np.arange(xlim[0], xlim[1] + 1), np.repeat(0.690, xlim[1] - xlim[0] + 1), lw=1, c=colors[3], ls='--')
    plt.plot(np.arange(xlim[0], xlim[1] + 1), np.repeat(0.807, xlim[1] - xlim[0] + 1), lw=1, c=colors[4], ls='--')
    plt.plot(np.arange(xlim[0], xlim[1] + 1), np.repeat(0.699, xlim[1] - xlim[0] + 1), lw=1, c=colors[5], ls='--')
    plt.plot(np.arange(xlim[0], xlim[1] + 1), np.repeat(0.862, xlim[1] - xlim[0] + 1), lw=1, c='k', ls='--')
    ax1 = plt.gca()
    ax1.set_facecolor('w')
    labels = ax1.get_xticklabels() + ax1.get_yticklabels()
    ax1.set_xlabel('DOY', font1)
    ax1.set_ylabel('Accuracy', font1)
    # ax1.set_ylabel('Accuracy', font1)
    for label in labels:
        label.set_fontname('Arial')
        label.set_fontsize(12)
        label.set_weight('bold')
    plt.tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        left=False,
        right=False,
        labelleft=True,
        labelbottom=True,
        # colors='white'
    )
    # if legend:
    #     ax1.legend(loc=4, facecolor='none', edgecolor='none', labelcolor='k', prop={
    #         'family': 'Arial',
    #         'weight': 'bold',
    #         'size': 10,
    #         'style': 'normal',
    #     })
    plt.grid(linestyle='--', lw=0.5)
    plt.xlim(xlim[0], xlim[1])
    plt.xticks(xticks[0], xticks[1])
    plt.ylim(-0.1, 1.1)

    plt.subplot(2, 2, 4)
    data1 = pd.read_csv(r'F:\DigitalAG\liheng\EU\csv\classification_gt_500_100.csv')
    crops1 = ['winter wheat', 'beets', 'corn', 'potato', 'rapeseed', 'barley']
    crops1_labels = ['Winter Wheat F1', 'Beet F1', 'Corn F1', 'Potato F1', 'Rapeseed F1', 'Winter Barley F1']

    f1 = np.zeros([24, 6])
    for idx in range(len(crops1)):
        f1[:, idx] = np.array(data1[crops1[idx] + ' f1'])
    oa = np.array(data1['oa'])
    colors = [np.array([165, 112, 0, 255]) / 255.0,
              np.array([168, 0, 226, 255]) / 255.0,
              np.array([255, 211, 0, 255]) / 255.0,
              np.array([112, 38, 0, 255]) / 255.0,
              np.array([209, 255, 0, 255]) / 255.0,
              np.array([226, 0, 124, 255]) / 255.0]

    for idx in range(len(crops1)):
        plt.plot(np.arange(1, n), f1[:, idx], c=colors[idx], label=crops1_labels[idx], lw=1.5, ms=6, marker='.')
    plt.plot(np.arange(1, n), oa, c='k', label='Overall Accuracy', lw=1.5, ms=6, marker='.')
    plt.plot(np.arange(xlim[0], xlim[1] + 1), np.repeat(0.855, xlim[1] - xlim[0] + 1), lw=1, c=colors[0], ls='--')
    plt.plot(np.arange(xlim[0], xlim[1] + 1), np.repeat(0.815, xlim[1] - xlim[0] + 1), lw=1, c=colors[1], ls='--')
    plt.plot(np.arange(xlim[0], xlim[1] + 1), np.repeat(0.734, xlim[1] - xlim[0] + 1), lw=1, c=colors[2], ls='--')
    plt.plot(np.arange(xlim[0], xlim[1] + 1), np.repeat(0.690, xlim[1] - xlim[0] + 1), lw=1, c=colors[3], ls='--')
    plt.plot(np.arange(xlim[0], xlim[1] + 1), np.repeat(0.807, xlim[1] - xlim[0] + 1), lw=1, c=colors[4], ls='--')
    plt.plot(np.arange(xlim[0], xlim[1] + 1), np.repeat(0.699, xlim[1] - xlim[0] + 1), lw=1, c=colors[5], ls='--')
    plt.plot(np.arange(xlim[0], xlim[1] + 1), np.repeat(0.862, xlim[1] - xlim[0] + 1), lw=1, c='k', ls='--')
    ax1 = plt.gca()
    ax1.set_facecolor('w')
    labels = ax1.get_xticklabels() + ax1.get_yticklabels()
    ax1.set_xlabel('DOY', font1)
    # ax1.set_ylabel('Accuracy', font1)
    for label in labels:
        label.set_fontname('Arial')
        label.set_fontsize(12)
        label.set_weight('bold')
    plt.tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        left=False,
        right=False,
        labelleft=False,
        labelbottom=True,
        # colors='white'
    )
    if legend:
        ax1.legend(loc=4, facecolor='none', edgecolor='none', labelcolor='k', prop={
            'family': 'Arial',
            'weight': 'bold',
            'size': 10,
            'style': 'normal',
        })
    plt.grid(linestyle='--', lw=0.5)
    plt.xlim(xlim[0], xlim[1])
    plt.xticks(xticks[0], xticks[1])
    plt.ylim(-0.1, 1.1)

    plt.subplots_adjust(top=0.97, bottom=0.1, right=0.95, left=0.07, wspace=0.03, hspace=0.05)
    plt.show()
    # plt.savefig(dir.replace('csv', 'figures', 1).replace('csv', 'jpg'))

if __name__ == '__main__':
    # compareTwoCombination(r'F:\DigitalAG\liheng\EU\csv\classification_gt_500_25.csv')
    sample_map_accuracy_summer(r'F:\DigitalAG\liheng\EU\labels\sample_map_accuracy_accu_summer.csv')
    # crop_type_mapping_gt()