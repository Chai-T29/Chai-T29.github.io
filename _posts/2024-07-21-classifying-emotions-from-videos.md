---
layout: post
title: "Classifying Emotions from Videos"
description: "Sentiment Analysis is widely used across industries to track how customers react to various stimuli."
date: 2024-07-21
feature_image: images/emotions.png
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

<Details markdown="block">
<summary>Click here to view the code</summary>
    
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
</Details>
<br>

Once you download the data, you should have 1440 videos in a folder called Speech. Here are some example frames from the videos:

![RAVDESS_Examples](https://github.com/user-attachments/assets/cd35df22-1211-4d2e-b98b-43d539f7e4f0)

#### Processing Each Video

Once we have the data downloaded and ready to go, we need to reduce the size of each frame and standardize the number of frames in each video to a desired number. This keeps our data compact and reduces the amount of redundancy in our data. For example, we can standardize the frame count to 50 so that every video only has 50 frames. This also makes our analysis easier because we can combine all the videos into one tensor.

<Details markdown="block">
<summary>Click here to view the code</summary>

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
</Details>
<br>
    
#### Processing Each Zip File

Using these two functions, we can now process each video, but we still need to create our training and testing tensors. To do this, we need to access data for each of the actors in the speech dataset. Each actor contains about 60 videos that are relevant to this project. We randomly split the videos into training and testing (40 training and 20 testing) for each actor, and merge the videos (and labels) into combined tensors. Once we have tensors for each zip file, we combine them together into our full training and testing datasets. Keep in mind, the training and testing tensors so far is only for the video data, so we will need other code for processing the audio data. 

<Details markdown="block">
<summary>Click here to view the code</summary>

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
</Details>
<br>

## Video Feature Extraction
Now that we have our video data loaded, we need to extract relevant information from the videos. To do this, I implemented a customized function based on the popular histogram of oriented gradients algorithm, but in 3D! The image below shows an example of how the 2D approach works.

![2D-HOG](https://github.com/user-attachments/assets/9e43d1b6-4b51-4bf5-a2b6-69f338c7e672)

#### Breakdown of Custom 3D Histogram of Oriented Gradients for Dimensionality Reduction

The formulation of this Histogram of Oriented Gradients algorithm is loosely based off of recent research in the field of computer vision. However, this project translates this approach for 3 dimensions. Below is an overview of the methodology that is applied to both training and testing datasets. If you are not too fond of math, you can skip to the [next section](#visualizing-gradient-magnitude-azimuthal-angle-and-polar-angle)
)!

**1.** Iterate through every sample, convert the frames to grayscale, and (optionally) apply the following Gaussian filter to each frame of the video ($V$):

$$
Gaussian\>Filter \> \Longrightarrow \> \frac{1}{1115}\begin{pmatrix}
  1 & 4 & 7 & 10 & 7 & 4 & 1 \\
  4 & 12 & 26 & 33 & 26 & 12 & 4 \\
  7 & 26 & 55 & 71 & 55 & 26 & 7 \\
  10 & 33 & 71 & 91 & 71 & 33 & 10 \\
  7 & 26 & 55 & 71 & 55 & 26 & 7 \\
  4 & 12 & 26 & 33 & 26 & 12 & 4 \\
  1 & 4 & 7 & 10 & 7 & 4 & 1
\end{pmatrix}
$$

**2.** We then compute the gradients with respect to the height, width [1], and frames ($\frac{\partial V}{\partial x}$, $\frac{\partial V}{\partial y}$, $\frac{\partial V}{\partial z}$).

**3.** Using these gradients, we can compute the three-dimensional gradient magnitude for each video.

$$
G = \sqrt{\left( \frac{\partial V}{\partial x}\right)^2 + \left( \frac{\partial V}{\partial y} \right)^2 + \left( \frac{\partial V}{\partial z}\right)^2}
$$

**4.** Generally for images ($I$), we compute the gradient direction by $\theta = \arctan \left( \frac{\frac{\partial I}{\partial y}}{\frac{\partial I}{\partial x}} \right)$ [2]. Since we have videos, we must compute the azimuthal angle and the polar angle to capture the 3D feature-space [4].

$$
\theta_{azimuth} = \arctan \left( \frac{\frac{\partial V}{\partial y}}{\frac{\partial V}{\partial x}} \right)
$$

$$
\phi_{polar} = \arctan \left( \frac{\sqrt{\left( \frac{\partial V}{\partial x}\right)^2 + \left( \frac{\partial V}{\partial y} \right)^2}}{\frac{\partial V}{\partial z}} \right)
$$

**5.** With our three sets of features, we can now partition the video into cells [3]. In our case, the cell size is ($5$, $6$, $5$), which will group sets of 180 pixels together.

**6.** With our grouped pixels, we cluster the gradient magnitudes of each pixel ($G_{(i, j, k)}$) into bins based on the azimuthal and polar angles. With $9$ bins, we sum the gradient magnitudes for all the pixels belonging to each bin for both types of angles, which reduces the dimensionality from $180$ points in each cell to $9 \times 2 = 18$ points per cell. We can then save these results to disk.

<br>

#### Visualizing Gradient Magnitude, Azimuthal Angle, and Polar Angle

You might be wondering--what does all this crazy math look like if you were to visualize it? This is what it would look like:

<video width="1500" height="700" height="500" controls muted loop autoplay>
  <source src="https://github.com/Chai-T29/Chai-T29.github.io/tree/e6f7b47ec4726d30cc5a94c85a8e4b61af953dc8/visualizations/video_features.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

<video width="640" height="480" controls>
  <source src="{{ '/assets/visualizations/video_features.mp4' | relative_url }}" type="video/mp4">
  Your browser does not support the video tag.
</video>





