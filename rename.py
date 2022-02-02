import os

def main(reverse=False):
    root_dir = r'F:\DigitalAG\liheng\EU\wt_sfl_cn\model'
    if not reverse:
        for item in os.listdir(root_dir):
            if item.split('_')[-1] == 'notselected':
                os.rename(os.path.join(root_dir, item), os.path.join(root_dir, item.replace('_notselected', 'e_187')))
    if reverse:
        for item in os.listdir(root_dir):
            if item.split('_')[-1] != 'notselected':
                os.rename(os.path.join(root_dir, item), os.path.join(root_dir, item.replace('e_187', '_notselected')))

if __name__ == '__main__':
    main(True)