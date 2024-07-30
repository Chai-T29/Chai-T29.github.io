---
layout: post
title: "Classifying Emotions from Videos"
description: "Sentiment Analysis is widely used across industries to track how customers react to various stimuli."
date: 2024-07-21
feature_image: images/emotions.png
---

Sentiment Analysis is widely used across industries to track how customers react to various stimuli, like in social media algorithms. With the rise of advanced machine learning algorithms, we can analyze many different types of data to make informed decisions on business strategy. One such recent breakthrough is the use of video and audio data to classify human emotions, and this project covers a detailed approach to doing so without any complex neural networks. No matter how experienced you are, I'm sure you will find this project very interesting!
<!--more-->

## Contents

Here are the contents of this project. If you are here just for a casual read, then you can just go ahead and skip over to the [second section](#video-feature-extraction).
1.  [Loading the Data](#loading-the-data)
2.  [Video Feature Extraction](#video-feature-extraction)
3.  [Audio Feature Extraction](#audio-feature-extraction)
4.  [Emotion Classification](#emotion-classification)
5.  [Conclusion](#conclusion)
6.  [References](#references)

## Loading the Data
To start this project, we download data from [Zenodo's website](https://zenodo.org/records/1188976). The dataset is called the Ryerson Audio-Visual Database of Emotional Speech and Song. For this project, we will only be focusing on the speech dataset. Here are the libraries you will need to start:

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

<br>

#### Downloading the Data

If you choose to get an API key from [their website](https://zenodo.org/records/1188976), you can use the code below to download the dataset to your computer (you can download it manually as well). But, the dataset is large, so unless your computer is powerful, I would not recommend you do this locally.

<Details markdown="block">
<summary>Click here to view the code</summary>
    
```python
record_id = '1188976'
response = requests.get(f'https://zenodo.org/api/records/{record_id}', params={'access_token': YOUR_API_KEY})
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
        response = requests.get(file_url, params={'access_token': YOUR_API_KEY})
        with open(file_name, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded: {file_name}")

print("Download complete.")
```
</Details>

<br>

Once you download the data, you should have 1440 videos in a folder called 'Speech'. Here are some example frames from the videos:

![RAVDESS_Examples](https://github.com/user-attachments/assets/cd35df22-1211-4d2e-b98b-43d539f7e4f0)

<br>

#### Processing Each Video

Once we have the dataset downloaded and ready to go, we need to reduce the size of each frame and standardize the number of frames in each video to a desired count. This keeps our data compact and reduces redundancy. For example, we can standardize the frame count to 50 so that every video only has 50 frames. This makes our analysis easier because we can combine all the videos into training and testing tensors.

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

With the two functions defined earlier, we can now process each video, but we still need to create our training and testing tensors. To do this, we need every video for all actors in the speech dataset. Each actor contains about 60 videos that are relevant to this project. We randomly split the videos into training and testing (40 training and 20 testing) for each actor and merged the videos (and labels) into combined tensors. Once we have tensors for each zip file, we combine them into our full training and testing datasets. Keep in mind, that the training and testing tensors so far are only for the video data, so we will need other code for processing the audio data.

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

y_train = id_train[:, 2].astype(np.int8) # Extracting the emotion labels from ids
y_test = id_test[:, 2].astype(np.int8)
```
</Details>
<br>


## Video Feature Extraction
Now that we have our video tensors, we need to extract relevant information from the videos. To do this, I implemented a customized function based on the popular Histogram of Oriented Gradients algorithm, but in 3D! The image below shows an example of how the 2D approach works.

![2D-HOG](https://github.com/user-attachments/assets/9e43d1b6-4b51-4bf5-a2b6-69f338c7e672)

Source: https://www.sciencedirect.com/topics/computer-science/histogram-of-oriented-gradient

Here are some libraries to get us started:

```python
import os
import numpy as np
from scipy.signal import convolve2d
import tensorly as tl
import gc
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
```

<br>

#### Breakdown of Custom 3D Histogram of Oriented Gradients for Dimensionality Reduction

The formulation of this Histogram of Oriented Gradients algorithm is loosely based on recent research in the field of computer vision. However, this project translates this approach into 3 dimensions. Below is an overview of the methodology applied to the training and testing datasets. If you are not too fond of math, you can skip to the [next section](#visualizing-gradient-magnitude-azimuthal-angle-and-polar-angle)!

**1.** Iterate through every sample and convert the frames to grayscale.

**2.** We then compute the gradients with respect to the height, width [[1]](#references), and frames ($\frac{\partial V}{\partial x}$, $\frac{\partial V}{\partial y}$, $\frac{\partial V}{\partial z}$).

**3.** Using these gradients, we can compute the three-dimensional gradient magnitude for each video.

$$
G = \sqrt{\left( \frac{\partial V}{\partial x}\right)^2 + \left( \frac{\partial V}{\partial y} \right)^2 + \left( \frac{\partial V}{\partial z}\right)^2}
$$

**4.** Generally for images ($I$), we compute the gradient direction by $\theta = \arctan \left( \frac{\frac{\partial I}{\partial y}}{\frac{\partial I}{\partial x}} \right)$ [[2]](#references). Since we have videos, we must compute the azimuthal angle and the polar angle to capture the 3D feature space [[4]](#references).

$$
\theta_{azimuth} = \arctan \left( \frac{\frac{\partial V}{\partial y}}{\frac{\partial V}{\partial x}} \right)
$$

$$
\phi_{polar} = \arctan \left( \frac{\sqrt{\left( \frac{\partial V}{\partial x}\right)^2 + \left( \frac{\partial V}{\partial y} \right)^2}}{\frac{\partial V}{\partial z}} \right)
$$

**5.** With our three sets of features, we can now partition the video into cells [[3]](#references). In our case, the cell size is ($5$, $6$, $5$), which will group sets of 180 pixels together.

**6.** With our grouped pixels, we cluster the gradient magnitudes of each pixel ($G_{(i, j, k)}$) into bins based on the azimuthal and polar angles. With $9$ bins, we sum the gradient magnitudes for all the pixels belonging to each bin for both types of angles, which reduces the dimensionality from $180$ points in each cell to $9 \times 2 = 18$ points per cell. We can then save these results to disk.

<br>


#### Implementing the Algorithm

Now that we have a foundational understanding of the HOG3D algorithm, we can create our custom implementation. I use two main functions--one for calculating the shape of the new data, and one for implementing the HOG3D algorithm itself.

<Details markdown="block">
<summary>Click here to view the code</summary>

```python
def calculate_total_descriptor_shape(width, height, n_frames, cell_size, nbins):
    """Calculate the total shape of the descriptor."""

    cells_per_width = width // cell_size[0]
    cells_per_height = height // cell_size[1]
    cells_per_depth = n_frames // cell_size[2]

    return cells_per_width, cells_per_height, cells_per_depth, nbins, 2

def compute_hog3d_rgb(frames, cell_size, nbins, v, gaussian_filter):
    """Compute the HOG3D features for a video."""

    width, height, channels, n_frames = frames.shape
    gray_frames = tl.tenalg.mode_dot(frames, np.array([0.2989, 0.5870, 0.1140]), mode=2)

    if gaussian_filter:
        print('Applying Gaussian Filter')
        gaussian_filter = np.array([[1, 4, 7, 10, 7, 4, 1],
                                [4, 12, 26, 33, 26, 12, 4],
                                [7, 26, 55, 71, 55, 26, 7],
                                [10, 33, 71, 91, 71, 33, 10],
                                [7, 26, 55, 71, 55, 26, 7],
                                [4, 12, 26, 33, 26, 12, 4],
                                [1, 4, 7, 10, 7, 4, 1]]) / 1115



        data = np.zeros_like(gray_frames)
        for i in range(n_frames):
            data[:, :, i] = convolve2d(gray_frames[:, :, i], gaussian_filter, mode='same')
    else:
        data = gray_frames
        del gray_frames
        gc.collect()

    gx = np.gradient(data, axis=0)
    gy = np.gradient(data, axis=1)
    gz = np.gradient(data, axis=2)

    magnitude = np.sqrt(gx**2 + gy**2 + gz**2)
    azimuthal_angle = np.arctan2(gy, gx)
    polar_angle = np.arctan2(np.sqrt(gx**2, gy**2), gz)

    output_shape = calculate_total_descriptor_shape(width, height, n_frames, cell_size, nbins)
    hog3d_descriptors = np.zeros(output_shape, dtype=np.float32)

    for i in range(0, n_frames - cell_size[2], cell_size[2]):
        for y in range(0, height - cell_size[1], cell_size[1]):
            for x in range(0, width - cell_size[0], cell_size[0]):
                cell_magnitude = magnitude[x:x + cell_size[0], y:y + cell_size[1], i:i + cell_size[2]]
                cell_azimuthal = azimuthal_angle[x:x + cell_size[0], y:y + cell_size[1], i:i + cell_size[2]]
                cell_polar = polar_angle[x:x + cell_size[0], y:y + cell_size[1], i:i + cell_size[2]]

                hist_azimuthal, _ = np.histogram(cell_azimuthal, bins=nbins, range=(-np.pi, np.pi), weights=cell_magnitude)
                hist_polar, _ = np.histogram(cell_polar, bins=nbins, range=(0, np.pi), weights=cell_magnitude)

                hist_azimuthal = hist_azimuthal / (np.linalg.norm(hist_azimuthal) + 1e-6)
                hist_polar = hist_polar / (np.linalg.norm(hist_polar) + 1e-6)

                hog3d_descriptors[x//cell_size[0], y//cell_size[1], i//cell_size[2], :, 0] = hist_azimuthal
                hog3d_descriptors[x//cell_size[0], y//cell_size[1], i//cell_size[2], :, 1] = hist_polar

    del cell_magnitude, cell_azimuthal, cell_polar, hist_azimuthal, hist_polar, gx, gy, gz, magnitude, azimuthal_angle, polar_angle
    gc.collect()

    return hog3d_descriptors
```
</Details>
<br>

#### Visualizing Gradient Magnitude, Azimuthal Angle, and Polar Angle

You might be wondering--what does all this crazy math look like if you were to visualize it? Well, this is what it would look like:

<iframe src="https://www.youtube.com/embed/Vm-0o4YNdD4?autoplay=1&loop=1&playlist=Vm-0o4YNdD4" width="1500" height="700" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen></iframe>

#### Creating our HOG3D Dataset

Now that we have our algorithms set up, we just need to process each video, apply our dimensionality reduction algorithm, and combine these new datasets.

<Details markdown="block">
<summary>Click here to view the code</summary>

```python
def process_videos(videos, cell_size=(5, 6, 5), nbins=9, gaussian_filter=False):
    """Process a batch of videos."""

    width, height, channels, n_frames, video_count = videos.shape

    for v in tqdm(range(video_count)):
        hog3d_descriptors = compute_hog3d_rgb(videos[:, :, :, :, v].astype(np.float32), cell_size, nbins, v, gaussian_filter)
        if v == 0:
            descriptors_shape = hog3d_descriptors.shape
            all_descriptors = np.zeros((video_count, *descriptors_shape), dtype=np.float32)

        all_descriptors[v] += hog3d_descriptors

        del hog3d_descriptors
        gc.collect()

    return all_descriptors

H_train = process_videos(train_videos)
H_test = process_videos(test_videos)
```
</Details>
<br>

## Audio Feature Extraction

Once we have our video data set up, we need to extract some features for our audio data. Here are some libraries to get us started:

```python
import os
import numpy as np
import librosa
import librosa.display
from IPython.display import Audio
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import zipfile
from skfda.representation.basis import BSplineBasis
from tqdm.notebook import tqdm
import gc
```

#### Breakdown of Feature Engineering and Extraction Process
The audio feature extraction process in the code below is a customized multi-step process that is formulated as shown below.

**1.** Process the audio files using Librosa and manipulate the audio by adding noise, changing the pitch, and changing the pitch of the audio to get multiple variations of the same audio sample.

Here are some examples of what this sounds like:

Original Audio
<audio controls>
  <source src="{{ '/assets/audio/03-01-01-01-01-01-01_original.wav' | relative_url }}" type="audio/wav">
</audio>

Noise Audio
<audio controls>
  <source src="{{ '/assets/audio/03-01-01-01-01-01-01_noise.wav' | relative_url }}" type="audio/wav">
</audio>

Pitch Changed Audio (+4)
<audio controls>
  <source src="{{ '/assets/audio/03-01-01-01-01-01-01_pitch_up.wav' | relative_url }}" type="audio/wav">
</audio>

Pitch Changed Audio (-6)
<audio controls>
  <source src="{{ '/assets/audio/03-01-01-01-01-01-01_pitch_down.wav' | relative_url }}" type="audio/wav">
</audio>

<br>

**2.** For each of the transformed audio files, we extract the following features using Librosa:
  -  Mel-Frequency Cepstral Coefficients (MFCCs): A representation of the short-term power spectrum of a sound, based on a linear cosine transform of a log power spectrum on a nonlinear mel scale of frequency. [Source](https://librosa.org/doc/main/generated/librosa.feature.mfcc.html)

  - Mel Spectogram: A spectrogram where the frequencies are converted to the Mel scale, which approximates the human ear's response more closely than the linear frequency scale. [Source](https://librosa.org/doc/main/generated/librosa.feature.melspectrogram.html)

  - Zero Crossing Rate (ZCR): The rate at which the audio signal changes sign from positive to negative or vice versa. It is a measure of the noisiness of the signal. [Source](https://librosa.org/doc/main/generated/librosa.feature.zero_crossing_rate.html)

  - Root Mean Square Energy (RMSE): A measure of the energy in the audio signal, calculated as the square root of the mean squared values of an audio signal. [Source](https://librosa.org/doc/main/generated/librosa.feature.rms.html)

  - Chromagram: A representation of 12 different pitch classes, or semitones, of a musical octave. They are calculated by mapping the entire frequency spectrum onto these 12 bins. [Source](https://librosa.org/doc/main/generated/librosa.feature.chroma_stft.html)

  - Spectral Contrast: A measure of the difference in amplitude between peaks and valleys in a sound spectrum. [Source](https://librosa.org/doc/main/generated/librosa.feature.spectral_contrast.html)

It might help to visualize what these features look like:

![Visualizing the features](https://github.com/user-attachments/assets/e7960f14-a8f9-4c37-a3e6-83b59b183fe3)

<br>

**3.** Once we collect this data, we combine all the extracted features and create a B-Spline feature space for this data. B-splines, or Basis Splines, are piece-wise polynomial approximations of a curve. They are defined recursively as such [[5]](#references):

$$
B_{i, j}(x) = \frac{x - t_i}{t_{i+j} - t_i} B_{i, j-1}(x) \\ + \frac{t_{i+j+1} - x}{t_{i+j+1} - t_{i+1}} B_{i+1, j-1}(x)
$$

for $j \ge 1$ with the initial condition:

$$
B_i^0(x) =
\begin{cases}
1 & \text{if } t_i \le x < t_{i+1} \\
0 & \text{otherwise}
\end{cases}
$$

Here:
- $B_{i, j}(x)$ is the B-Spline basis function of degree $k$.
- $x$ is the parameter.
- $t_i$ are the knots.

The B-Spline curve $C(x)$ of degree $j$ can be defined as a linear combination of these basis functions:

$$
C(t) = \sum_{i=0}^{n} P_i B_{i, j}(x)
$$

where $P_i$ are the control points.

Here's what the basis functions look like for various orders:

![basis-functions](https://github.com/user-attachments/assets/7abf7bd8-445f-4e7b-ae1c-800a2c196a51)

<br>

This is a heatmap of a basis matrix with an order of 4:

![basis-function heatmap](https://github.com/user-attachments/assets/f981616d-200b-4f18-8d69-1fdbf1cd80d6)
<br>

**4.** Once we develop this feature space, we project the extracted feature space onto the B-Spline feature space, which transforms the data into a lower dimensional approximation. For example, if we have $12$ knots, then no matter how many columns our data has, we would have knots + $2$, or $14$, columns.

The B-Spline transformation reduces the dimensionality from the image earlier to this:

![reduced-dimensionality](https://github.com/user-attachments/assets/d826733a-bf3b-4e32-8b8c-07364cc541a5)
<br>

**5.** Now, we just have to combine the data and save it to disk.

#### Implementing the algorithm

With our understanding of the feature extraction process, we can now implement the Python code to extract these features and combine it into our audio data.

<Details markdown="block">
<summary>Click here to view the code</summary>

```python
def add_noise(y, noise_factor=0.005):
    """Add noise to the audio signal"""

    noise = np.random.randn(len(y))
    augmented_data = y + noise_factor * noise
    return augmented_data

def change_pitch(y, sr, n_steps):
    """Change the pitch of the audio signal"""

    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

def change_speed(y, speed_factor):
    """Change the speed of the audio signal"""

    return librosa.effects.time_stretch(y, rate=speed_factor)

def lse_solver(X, y):
    """Least Squares Estimation Solver"""

    return np.linalg.inv(X.T @ X) @ X.T @ y

def transform_features(y, sr):
    """Transform the audio signal and merge the data"""

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T
    mels = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=512, n_mels=128)
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=2048, hop_length=512).T
    rmse = librosa.feature.rms(y=y, frame_length=2048, hop_length=512).T
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=512).T
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=512).T

    features = np.concatenate((mfccs, zcr, rmse, chroma, spectral_contrast), axis=1)

    del mfccs
    del mels
    del zcr
    del rmse
    del chroma
    del spectral_contrast
    gc.collect()

    return features

def extract_features(y, sr, nknots=10, order=4):
    """Extract features from the audio signal using B-Splines"""

    features = transform_features(y, sr)
    m, n = features.shape

    knots = np.linspace(0, 1, nknots)
    xx = np.linspace(0, 1, m)

    bs = BSplineBasis(knots=knots, order=order)
    bs_basis = bs(xx).T[0]

    bs_features = lse_solver(bs_basis, features)

    del bs
    del bs_basis
    del features
    gc.collect()

    return bs_features

nknots = 12
Audio_data = np.zeros((60*24, nknots+2, 34*8), dtype=np.float32)

inds = np.zeros((60*24, 7), dtype=np.uint8)
i = 0
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    for folder in tqdm(zip_ref.infolist()):
        file_name = folder.filename
        if file_name.endswith('.wav'):
            with zip_ref.open(folder) as audio_file:
                idx = np.array(file_name.split('/')[1][:-4].split('-')).astype(np.uint8)
                inds[i, :] = idx

                y, sr = librosa.load(audio_file)

                y_noise = add_noise(y)
                y_pitch1 = change_pitch(y, sr, 4)
                y_pitch2 = change_pitch(y, sr, -6)
                y_pitch3 = change_pitch(y, sr, 3)
                y_speed1 = change_speed(y, 1.5)
                y_speed2 = change_speed(y, 2.0)
                y_speed3 = change_speed(y, 0.8)
                y_speed4 = change_speed(y, 0.5)

                features1 = extract_features(y_noise, sr, nknots, 4)
                features2 = extract_features(y_pitch1, sr, nknots, 4)
                features3 = extract_features(y_pitch2, sr, nknots, 4)
                features4 = extract_features(y_pitch3, sr, nknots, 4)
                features5 = extract_features(y_speed1, sr, nknots, 4)
                features6 = extract_features(y_speed2, sr, nknots, 4)
                features7 = extract_features(y_speed3, sr, nknots, 4)
                features8 = extract_features(y_speed4, sr, nknots, 4)

                features = np.concatenate((features1, features2, features3, features4, features5, features6, features7, features8), axis=1)
                m, n = features.shape

                Audio_data[i, ...] = features

                del features, features1, features2, features3, features4, \
                    features5, features6, features7, features8, y_noise, y_pitch1, y_pitch2, y_pitch3, \
                    y_speed1, y_speed2, y_speed3, y_speed4, y, sr

                gc.collect()
                i += 1
```
</Details>
<br>

## Emotion Classification

Now that we finished most of the tedious work, we just have a couple more steps till the finish line! Here's the necessary imports for this section:

<Details markdown="block">
<summary>Click here to view the code</summary>

```python
import os
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neural_network import MLPClassifier
from tensorly.tenalg import multi_mode_dot
from tensorly.decomposition import partial_tucker
from tqdm.notebook import tqdm
```
</Details>
<br>

#### Partial Tucker Decomposition for Additional Dimensionality Reduction

Currently, the shapes of the data are as follows:

- video_train --> ($960$, $48$, $71$, $10$, $9$, $2$)
- video_test --> ($480$, $48$, $71$, $10$, $9$, $2$)
- audio_train --> ($960$, $14$, $272$)
- audio_test --> ($480$, $14$, $272$)

So, if we were to flatten out the data, we would have $48 * 71 * 10 * 9 * 2 = 613440$ columns for the videos, and $14 * 272 = 3808$ columns for the audio data, which is just too large. We need to reduce the dimensionality further using higher-order PCA, or Tucker Decomposition.

Below is a brief overview of how Tucker Decomposition works. Again, if you're not the biggest fan of math, you can skip to the [next section]().

Given a tensor $\mathcal{X} \in \mathbb{R}^{I_1 \times I_2 \times \cdots \times I_N}$, Tucker decomposition approximates $\mathcal{X}$ as [[6]](#references):

$$
\mathcal{X} \approx \mathcal{G} \times_1 A^{(1)} \times_2 A^{(2)} \times_3 \cdots \times_N A^{(N)}
$$

where:
- $\mathcal{G} \in \mathbb{R}^{J_1 \times J_2 \times \cdots \times J_N}$ is the core tensor.
- $A^{(n)} \in \mathbb{R}^{I_n \times J_n}$ are the factor matrices for each mode $n$.

The operator $\times_n$ denotes the mode-$n$ product between a tensor and a matrix. Specifically, the mode-$n$ product of a tensor $\mathcal{G}$ with a matrix $A$ is defined as:

$$
(\mathcal{G} \times_n A)_{i_1 i_2 \cdots i_{n-1} j i_{n+1} \cdots i_N} = \sum_{i_n} \mathcal{G}_{i_1 i_2 \cdots i_N} A_{j i_n}
$$
<br>

This image more clearly demonstrates the math:

![Third-order-Tucker-decomposition](https://github.com/user-attachments/assets/235fa466-0aaf-49df-95a8-876c40171dd5)

Source: https://www.researchgate.net/figure/Third-order-Tucker-decomposition_fig1_257482079

#### Implementing the algorithm

Tensorly makes it super easy for use to implement Partial Tucker Decomposition, which is a customized form of the full Tucker, but it maintains the integrity of dimensions that aren't being reduced, like the number/order of samples. Once we reduce the video and audio data and flatten it, we can combine them together into our X_train and X_test for classification!

<Details markdown="block">
<summary>Click here to view the code</summary>

```python
# Video Tucker Decomposition
video_ranks = [8, 8, 3, 1]
video_modes = [1, 2, 3, 5]

temp_data, _ = partial_tucker(video_train, video_ranks, modes=video_modes, verbose=True, tol=1e-4)

V_train = temp_data[0]
video_train_factors = temp_data[1]

del temp_data
gc.collect()

V_test = multi_mode_dot(video_test, [U.T for U in video_train_factors], modes=video_modes)

V_train = V_train.reshape(V_train.shape[0], -1)
V_test = V_test.reshape(V_test.shape[0], -1)


# Audio Tucker Decomposition
audio_ranks = [10, 150]
audio_modes = [1, 2]

temp_data, _ = partial_tucker(audio_train, audio_ranks, modes=audio_modes, verbose=True, tol=1e-4)

A_train = temp_data[0]
audio_train_factors = temp_data[1]

del data
gc.collect()

A_test = multi_mode_dot(audio_test, [U.T for U in audio_train_factors], modes=audio_modes)

A_train = A_train.reshape(A_train.shape[0], -1)
A_test = A_test.reshape(A_test.shape[0], -1)

# Combining the Data
X_train = np.concatenate((V_train, A_train), axis=1)
X_test = np.concatenate((V_test, A_test), axis=1)

print(X_train.shape)
print(X_test.shape)
```
</Details>
<br>

#### Training a Multilayer Perceptron Model using Sklearn

We could just train a Support Vector Classifier or Random Forest (and I do on my [Github](https://github.com/Chai-T29)), but I decided to use a simple neural network instead because a lot of the data is not linearly separable.

```python
mlp = MLPClassifier(alpha=0.07, batch_size=200, epsilon=1e-8, n_iter_no_change=10, learning_rate='adaptive', hidden_layer_sizes=(250, 500, 100), max_iter=1000, random_state=42)
mlp.fit(X_train, y_train)

y_pred = mlp.predict(X_test)
```
Using this model, we get an accuracy of $85.625$%! Considering that we did not use any complex deep-learning architectures, this is pretty impressive! Here's a more detailed overview of the performance:

![performance](https://github.com/user-attachments/assets/0673d41a-b006-4420-9a91-6dcb5b949686)

As we can see, most of the labels performed extremely well, especially disgust. This is most likely because disgust is a very strong emotion, which can be easier to pick up with our algorithm. However, sad and surprised emotions did not perform as well. These errors can easily be fixed with a more robust model or more trial and error with the current model architecture.

<br>

## Conclusion

This project showcases the potential of a custom approach to emotion classification using multimodal data. By combining 3D Histograms of Oriented Gradients for video features and advanced audio feature extraction with B-Spline transformations, we were able to create a robust and efficient feature extraction pipeline. The integration of these features through Partial Tucker Decomposition significantly reduced the dimensionality, making our models more efficient without sacrificing performance. The impressive accuracy of our ensemble method underscores the power of combining video and audio data, paving the way for more sophisticated and accurate emotion detection systems in real-world applications. This custom approach not only enhances the accuracy of emotion classification but also demonstrates the potential for broader applications in any field requiring nuanced analysis of multimedia data.

If you've made it this far, thank you for giving this a read, and happy learning!

<br>

## References:

[1] SpringerLink (Online service), Panigrahi, C. R., Pati, B., Mohapatra, P., Buyya, R., & Li, K. (2021). Progress in Advanced Computing and Intelligent Engineering: Proceedings of ICACIE 2019, Volume 1 (1st ed. 2021.). Springer Singapore : Imprint: Springer. https://doi.org/10.1007/978-981-15-6584-7

[2] Zoubir, Hajar & Rguig, Mustapha & Aroussi, Mohamed & Chehri, Abdellah & Rachid, Saadane. (2022). Concrete Bridge Crack Image Classification Using Histograms of Oriented Gradients, Uniform Local Binary Patterns, and Kernel Principal Component Analysis. Electronics. 11. 3357. 10.3390/electronics11203357. https://doi.org/10.3390/electronics11203357

[3]  S V Shidlovskiy et al 2020 J. Phys.: Conf. Ser. 1611 012072 https://doi:10.1088/1742-6596/1611/1/012072

[4] Nykamp DQ, “Spherical coordinates.” From Math Insight. http://mathinsight.org/spherical_coordinates

[5] Hastie, T., Tibshirani, R., Friedman, J. (2009). Basis Expansions and Regularization. In: The Elements of Statistical Learning. Springer Series in Statistics. Springer, New York, NY. https://doi.org/10.1007/978-0-387-84858-7_5

[6] Kolda, Tamara G., and Brett W. Bader. "Tensor Decompositions and Applications." *SIAM Review*, vol. 51, no. 3, 2009, pp. 455-500. Society for Industrial and Applied Mathematics. https://doi.org/10.1137/07070111X.
