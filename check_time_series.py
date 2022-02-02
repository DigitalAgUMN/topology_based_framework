import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def main():
    data = pd.read_csv(r'G:\My Drive\GEE_france\france_crop_features.csv')
    variable = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7',
                'B8', 'B8A', 'B11', 'B12', 'NDVI', 'EVI', 'GCVI', 'LSWI']
    l = ['background', 'winter wheat', 'beets', 'corn', 'sunflowers', 'rapeseed', 'winter barley']

    columns = data.columns
    def filter(item):
        if item[:len(v)] == v:
            return item
    for v in variable:
        feature = list(map(filter, columns))
        feature = [x for x in feature if x is not None]
        std_times = 0.5
        for i in [0,3,4]:
            d = data[data['class']==i][feature].mean()
            std = data[data['class']==i][feature].mean()
            plt.plot(np.arange(0, 9), d, label=l[i])
            # plt.fill_between(np.arange(0, 9), d - std * std_times,
            #                  d - std * std_times,
            #                  alpha=0.3)
            ax = plt.gca()
            labels1 = ax.get_xticklabels() + ax.get_yticklabels()
            for label in labels1:
                label.set_fontname('Arial')
                label.set_fontsize(9)
                label.set_weight('bold')
            ax.legend(loc=1, facecolor='none', edgecolor='none', prop={
                'family': 'Arial',
                'weight': 'bold',
                'size': 8,
                'style': 'normal'
            })
        plt.savefig(r'F:\DigitalAG\liheng\EU\\'+ v +'.jpg')
        plt.cla()
        # plt.show()

if __name__ == '__main__':
    main()
