
import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from toolkits.utils import extract_meta_av, extract_meta_cv


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, help='Source dir',
                        default='/mnt/8T/hou/-deepfake_dataset/ff')
    parser.add_argument('--videodataset', type=Path, default='data/ffpp_videos.pkl',
                        help='Path to save the videos DataFrame')

    # return parser.parse_args(argv)


def main(argv):
    # Parameters parsing
    # args = parse_args(argv)
    source_dir: Path = Path('/mnt/8T/hou/deepfake_dataset/ff')
    videodataset_path: Path = Path('/home/hou/mycodes/paper reimplement/DFDC2/icpr2020dfdc/data/ffppNew_videos_c40.pkl')

    # Create ouput folder (if doesn't exist)
    videodataset_path.parent.mkdir(parents=True, exist_ok=True)

    # DataFrame
    # 先检测dataframe是否已经存在
    if videodataset_path.exists():
        print('Loading video DataFrame')
        df_videos = pd.read_pickle(videodataset_path)
    else:
        print('Creating video DataFrame')
        # rglob会返回一个生成器，可以递归返回该文件下所以匹配该正则的文件
        ff_videos = Path(source_dir).rglob('*.mp4')
        # 过滤掉所有带mask，或者raw的文件，将文件路径转化为相对路径全部存放在dataframe中
        df_videos = pd.DataFrame(
            {'path': [f.relative_to(source_dir) for f in ff_videos if 'mask' not in str(f) and 'raw' not in str(f) and 'DeepFakeDetection' not in str(f) and 'actors' not in str(f) and 'c23' not in str(f)]})
        # print(df_videos)
        df_videos['height'] = df_videos['width'] = df_videos['frames'] = np.zeros(
            len(df_videos), dtype=np.uint16)
        # 默认会开启20条子进程
        with Pool() as p:
            # 后面这个无名函数的作用就是将路径转化为绝对路径，然后作为参数，传给进程p，执行extract_meta_av，dataframe传给参数为str的函数
            meta = p.map(extract_meta_av, df_videos['path'].map(
                lambda x: str(source_dir.joinpath(x))))
        # 通过np.stack将array，转化为ndarray
        meta = np.stack(meta)
        # 关于dataframe的切片，第一个切的是行号，第二个切的是列号
        #print(df_videos.loc[:, ['height', 'width', 'frames']])
        df_videos.loc[:, ['height', 'width', 'frames']] = meta
        #print(df_videos.loc[:, ['height', 'width', 'frames']])
        # Fix for videos that av cannot decode properly
        for idx, record in df_videos[df_videos['frames'] == 0].iterrows():
            meta = extract_meta_cv(str(source_dir.joinpath(record['path'])))
            df_videos.loc[idx, ['height', 'width', 'frames']] = meta

        df_videos['class'] = df_videos['path'].map(
            lambda x: x.parts[0]).astype('category')
        # 当视频是假的，label为True   ，当视频为真的，label为False
        df_videos['label'] = df_videos['class'].map(
            lambda x: True if x == 'manipulated_sequences' else False)  # True is FAKE, False is REAL
        df_videos['source'] = df_videos['path'].map(
            lambda x: x.parts[1]).astype('category')
        df_videos['quality'] = df_videos['path'].map(
            lambda x: x.parts[2]).astype('category')
        # path.with_suffix用于修改文件的后缀
        df_videos['name'] = df_videos['path'].map(
            lambda x: x.with_suffix('').parts[-1])

        df_videos['original'] = -1 * np.ones(len(df_videos), dtype=np.int16)
        
        # 进行过滤，不要deepfakedetection，也不要actor的
        # df_videos = df_videos[(df_videos['source'] != 'DeepFakeDetection') & (df_videos['source'] != 'actors')] 
        
        
        ##
        df_videos.loc[(df_videos['label'] == True) & (df_videos['source'] != 'DeepFakeDetection'), 'original'] = \
            df_videos[(df_videos['label'] == True) & (df_videos['source'] != 'DeepFakeDetection')]['name'].map(
                lambda x: df_videos.index[np.flatnonzero(
                    df_videos['name'] == x.split('_')[0])[0]]
        )
        df_videos.loc[(df_videos['label'] == True) & (df_videos['source'] == 'DeepFakeDetection'), 'original'] = \
            df_videos[(df_videos['label'] == True) & (df_videos['source'] == 'DeepFakeDetection')]['name'].map(
                lambda x: df_videos.index[
                    np.flatnonzero(df_videos['name'] == x.split('_')[0] + '__' + x.split('__')[1])[0]]
        )

        print('Saving video DataFrame to {}'.format(videodataset_path))
        df_videos.to_pickle(str(videodataset_path))

    print('Real videos: {:d}'.format(sum(df_videos['label'] == 0)))
    print('Fake videos: {:d}'.format(sum(df_videos['label'] == 1)))
    print(df_videos[df_videos["quality"] == 'c23'].size)
    print(df_videos[df_videos["quality"] == 'c40'].size)


if __name__ == '__main__':
    main(sys.argv[1:])
