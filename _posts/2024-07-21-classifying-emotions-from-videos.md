---
layout: post
title: "Classifying Emotions from Videos"
description: "Sentiment Analysis is widely used across industries to track how customers react to various stimuli."
date: 2024-07-21
feature_image: images/road.jpg
tags: [computervision, machinelearning, research]
---

Sentiment Analysis is widely used across industries to track how customers react to various stimuli. Now, with the rise of social media and advanced machine learning algorithms, we are able to analyze many different types of data to make informed decisions on business strategy. One such recent breakthrough is the use of video and audio data to classify human emotions, and this project covers a detailed approach in doing so without any complex neural networks. No matter how experienced you are, I'm sure that you will find this project very interesting!
<!--more-->

## Contents

This project consists of four main steps. If you are here just for a casual read, then skip over to the second section.
1.  [Loading the Data](#loading-the-data)
2.  [Video Feature Extraction](#video-feature-extraction)
3.  [Audio Feature Extraction](#audio-feature-extraction)
4.  [Emotion Classification](#emotion-classification)

## Loading the Data
To start this project off, we need to download data from [Zenodo's website](https://zenodo.org/records/1188976). The data is called the Ryerson Audio-Visual Database of Emotional Speech and Song. For this project, we will only be focusing on the speech data. Here's the libraries you will need to start off:

```python
import requests
import os
from tqdm.notebook import tqdm
import zipfile
import cv2
import numpy as np
import gc
from io import BytesIO
import tempfile
import matplotlib.pyplot as plt
%matplotlib inline
```

#### Downloading the Data

If you choose to get an api key from their website, you can use the code below to download the data to your computer. But, this data is extremely large, so unless your computer is capable of handling large data, I would not recommend you do this locally.

<summary>Click here for the code</summary>
<details>

  
```python
record_id = '1188976'
response = requests.get(f'https://zenodo.org/api/records/{record_id}', params={'access_token': ACCESS_TOKEN})
data = response.json()

all_files = data['files']
speech_files = []
song_files = []
for d in all_files:
    components = d['key'].split('_')
    if components[1] == 'Speech':
        speech_files.append(d)
    else:
        song_files.append(d)
os.makedirs('Speech', exist_ok=True)

for file in tqdm(speech_files):
    file_url = file['links']['self']
    file_name = os.path.join('Speech', file['key'])

    if not os.path.exists(file_name):
        response = requests.get(file_url, params={'access_token': ACCESS_TOKEN})
        with open(file_name, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded: {file_name}")

print("Download complete.")
```
</details>

Once you download the data, you should have 1440 videos in a folder called Speech. Here are some example frames from the videos:
![RAVDESS_Examples](https://github.com/user-attachments/assets/cd35df22-1211-4d2e-b98b-43d539f7e4f0)

#### Processing Each Video

Once we have the data downloaded and ready to go, we need to reduce the size of each frame and standardize the number of frames in each video to a desired number. This keeps our data compact and reduces the amount of redundancy in our data. For example, we can standardize the frame count to 50 so that every video only has 50 frames. This also makes our analysis easier because we can combine all the videos into one tensor.

<details>
<summary>Click here for the code</summary>

```python
def resize_frame(frame, scale_factor):
    """Resizes a frame by a given scale factor."""

    height, width = frame.shape[:2]
    new_dimensions = (int(width * scale_factor), int(height * scale_factor))
    resized_frame = cv2.resize(frame, new_dimensions, interpolation=cv2.INTER_AREA)
    return resized_frame

def interpolate_frames(frames, target_frame_count):
    """Interpolates frames to a given frame count."""

    original_frame_count = len(frames)
    if original_frame_count == target_frame_count:
        return frames

    indices = np.linspace(0, original_frame_count - 1, target_frame_count)
    interpolated_frames = []
    for i in indices:
        lower_index = int(np.floor(i))
        upper_index = int(np.ceil(i))
        if lower_index == upper_index:
            interpolated_frames.append(frames[lower_index])
        else:
            lower_weight = upper_index - i
            upper_weight = i - lower_index
            interpolated_frame = (lower_weight * frames[lower_index] + upper_weight * frames[upper_index]).astype(np.uint8)
            interpolated_frames.append(interpolated_frame)

    return interpolated_frames
```
</details>

#### Processing Each Zip File

Using these two functions, we can now process each video, but we still need to create our training and testing tensors. To do this, we need to access data for each of the actors in the speech dataset. Each actor contains about 60 videos that are relevant to this project. We randomly split the videos into training and testing (40 training and 20 testing) for each actor, and merge the videos (and labels) into combined tensors. Once we have tensors for each zip file, we combine them together into our full training and testing datasets. Keep in mind, the training and testing tensors so far is only for the video data, so we will need other code for processing the audio data. 

<details>
<summary>Click here for the code</summary>

```python

def process_zip_file(zip_path, target_frame_count, scale_factor, train_size):
    """Processes a zip file and returns training and testing tensors."""

    tensors_shape = (int(720*scale_factor), int(1280*scale_factor), 3, target_frame_count, 60)
    video_tensors = np.zeros(tensors_shape, dtype=np.uint8)
    identifier = np.zeros((60, 7), dtype=np.uint8)

    i = 0
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for file_info in tqdm(zip_ref.infolist()):
            file_name = file_info.filename.split('/')[1]
            if (file_name.startswith('02')) and (file_name.endswith('.mp4')):
                try:
                    with zip_ref.open(file_info) as video_file:
                        video_bytes = BytesIO(video_file.read())
                        video_tensor = video_to_array(video_bytes, target_frame_count, scale_factor)

                        if video_tensor is not None:
                            id = np.array(file_name[:-4].split('-'), dtype=np.uint8)
                            identifier[i, :] = id
                            video_tensors[:, :, :, :, i] = video_tensor

                            del file_name
                            del video_bytes
                            del video_tensor
                            del id
                            gc.collect()

                except Exception as e:
                    print(f"Failed to process {file_info.filename}: {e}")
                i += 1

    permutation = np.random.permutation(i)
    training_indices = permutation[:train_size]
    all_indices = np.arange(i)
    testing_indices = np.array([j for j in all_indices if j not in training_indices])

    training_tensor = video_tensors[:, :, :, :, training_indices]
    testing_tensor = video_tensors[:, :, :, :, testing_indices]
    del video_tensors
    gc.collect()

    training_identifier = identifier[training_indices, :]
    testing_identifier = identifier[testing_indices, :]
    del identifier
    gc.collect()

    return training_tensor, training_identifier, testing_tensor, testing_identifier

folder_path = 'Speech'
target_frame_count = 50
scale_factor = 1/3

train_size = 40  # out of 60 videos (per actor)
test_size = 60 - train_size

train_shape = (int(720*scale_factor), int(1280*scale_factor), 3, target_frame_count, 24*train_size)
test_shape = (int(720*scale_factor), int(1280*scale_factor), 3, target_frame_count, 24*test_size)

train_videos = np.zeros(train_shape, dtype=np.uint8)
test_videos = np.zeros(test_shape, dtype=np.uint8)

id_train = np.zeros((24*train_size, 7), dtype=np.uint8)
id_test = np.zeros((24*test_size, 7), dtype=np.uint8)

i = 0
for zip_filename in tqdm(os.listdir(folder_path), desc='Total Progress'):
    if zip_filename.endswith(".zip") and zip_filename.startswith("Video"):
        zip_path = os.path.join(folder_path, zip_filename)
        print(f"Processing {zip_filename}...")
        training_tensor, training_identifier, testing_tensor, testing_identifier = \
            process_zip_file(zip_path, target_frame_count, scale_factor, train_size)

        print('Training Tensor Shape:', training_tensor.shape)
        print('Testing Tensor Shape:', testing_tensor.shape)

        train_videos[:, :, :, :, i*train_size:(i+1)*train_size] = training_tensor
        id_train[i*train_size:(i+1)*train_size, :] = training_identifier
        test_videos[:, :, :, :, i*test_size:(i+1)*test_size] = testing_tensor
        id_test[i*test_size:(i+1)*test_size, :] = testing_identifier

        plt.imshow(training_tensor[:, :, :, 0, 0]) # Showing a random frame from each actor's video tensors
        plt.show()
        del training_tensor
        del training_identifier
        del testing_tensor
        del testing_identifier

        gc.collect()
        i += 1

```
</details>

## Video Feature Extraction
Now that we have our video data loaded, we can start doing some fun analysis!






