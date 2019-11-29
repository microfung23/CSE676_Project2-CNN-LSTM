#From Microsoft/CNTK 
##https://github.com/microsoft/CNTK/tree/master/Examples/Video/DataSets/UCF11

import sys
import argparse
import csv
import numpy as np
import os
import imageio

def load_groups(input_folder):
    '''
    Load the list of sub-folders into a python list with their
    corresponding label.
    '''
    groups         = []
    label_folders  = os.listdir(input_folder)
    index          = 0
    for label_folder in sorted(label_folders):
        label_folder_path = os.path.join(input_folder, label_folder)
        if os.path.isdir(label_folder_path):
            group_folders = os.listdir(label_folder_path)
            for group_folder in group_folders:
                if group_folder != 'Annotation':
                    groups.append([os.path.join(label_folder_path, group_folder), index])
            index += 1

    return groups


def split_data(groups, file_ext):
    '''
    Split the data at random for train, eval and test set.
    '''
    group_count = len(groups)
    indices = np.arange(group_count)

    np.random.seed(0) # Make it deterministic.
    np.random.shuffle(indices)

    # 80% training and 20% test.
    train_count = int(0.8 * group_count)
    test_count  = group_count - train_count

    train = []
    test  = []
    train_max = 0
    train_min = 100
    test_max = 0
    test_min = 100
    omitted_train = 0
    omitted_test = 0
    for i in range(train_count):
        group = groups[indices[i]]
        video_files = os.listdir(group[0])
        for video_file in video_files:
            video_file_path = os.path.join(group[0], video_file)
            if os.path.isfile(video_file_path):
                video_file_path = os.path.abspath(video_file_path)
                ext = os.path.splitext(video_file_path)[1]
                if (ext == file_ext):
                    # make sure we have enough frames and the file isn't corrupt
                    video_reader = imageio.get_reader(video_file_path, 'ffmpeg')                    
                    if video_reader.count_frames() >= 5:
                        train.append([video_file_path, group[1]])
                        if(video_reader.count_frames() > train_max):
                            train_max = video_reader.count_frames()
                        if(video_reader.count_frames() < train_min):
                            train_min = video_reader.count_frames()
                    else:
                        omitted_train = omitted_train+1

    for i in range(train_count, train_count + test_count):
        group = groups[indices[i]]
        video_files = os.listdir(group[0])
        for video_file in video_files:
            video_file_path = os.path.join(group[0], video_file)
            if os.path.isfile(video_file_path):
                video_file_path = os.path.abspath(video_file_path)
                ext = os.path.splitext(video_file_path)[1]
                if (ext == file_ext):
                    # make sure we have enough frames and the file isn't corrupt
                    video_reader = imageio.get_reader(video_file_path, 'ffmpeg')
                    if video_reader.count_frames() >= 5:
                        test.append([video_file_path, group[1]])
                        if(video_reader.count_frames() > test_max):
                            test_max = video_reader.count_frames()
                        if(video_reader.count_frames() < test_min):
                            test_min = video_reader.count_frames()
                    else:
                        omitted_test = omitted_test + 1
    print(omitted_train)
    print(omitted_test)
    print(train_max)
    print(test_max)  
    print(train_min)
    print(test_min)                 
    return train, test

def write_to_csv(items, file_path):
    '''
    Write file path and its target pair in a CSV file format.
    '''
    if sys.version_info[0] < 3:
        with open(file_path, 'wb') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            for item in items:
                writer.writerow(item)
    else:
        with open(file_path, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            for item in items:
                writer.writerow(item)

def main(input_folder, output_folder):
    '''
    Main entry point, it iterates through all the video files in a folder or through all
    sub-folders into a list with their corresponding target label. It then split the data
    into training set and test set.
    :param input_folder: input folder contains all the video contents.
    :param output_folder: where to store the result.
    '''
    groups = load_groups(input_folder)
    train, test = split_data(groups, '.avi')

    write_to_csv(train, os.path.join(output_folder, 'train_map.csv'))
    write_to_csv(test, os.path.join(output_folder, 'test_map.csv'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i",
                        "--input_folder",
                        type = str,
                        help = "Input folder containing the raw data.",
                        required = True)

    parser.add_argument("-o",
                        "--output_folder",
                        type = str,
                        help = "Output folder for the generated training, validation and test text files.",
                        required = True)

    args = parser.parse_args()

    main(args.input_folder, args.output_folder)