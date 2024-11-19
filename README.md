# Real-Time-Wake-Word-Detection
Real-time wake word detection using TensorFlow Lite with MFCC feature extraction and silence handling.

This project implements a real-time Wake Word Detection system using machine learning and TFLite models for efficient inference on edge devices. The system uses audio data to continuously monitor the environment and detect whether a predefined "Wake Word" has been spoken. It utilizes **MFCC (Mel Frequency Cepstral Coefficients)** features and a **Convolutional Neural Network (CNN)** model to classify the audio segments as either "Wake Word Detected" or "Wake Word NOT Detected."

## Table of Contents
- [Project Description](#project-description)
- [Setup Instructions](#setup-instructions)
- [How to Use](#how-to-use)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Description
The Wake Word Detection system works by continuously recording audio and feeding it into a trained CNN model. The model classifies each 1-second audio segment, and if it detects the Wake Word, it triggers an action. This system is designed to be lightweight and can be deployed on embedded systems with TensorFlow Lite (TFLite) for efficient real-time performance.

Key steps involved:
1. **Audio Data Collection**: Record Wake Word audio samples and background sounds.
2. **Preprocessing**: Extract MFCC features from the audio.
3. **Model Training**: Train a CNN model to classify the Wake Word and background audio.
4. **Model Conversion**: Convert the trained model to TFLite format for deployment.
5. **Real-Time Inference**: Use the TFLite model to classify real-time audio segments from a microphone input.

## Setup Instructions
1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/real-time-wake-word-detection.git
   cd real-time-wake-word-detection
   ```

2. **Install dependencies**:
   You will need Python 3 and the following libraries:
   ```bash
   pip install tensorflow librosa sounddevice numpy scikit-learn matplotlib
   ```

3. **Download or Record Audio Data**:
   - You can either record your own audio data using the provided scripts or use the included sample audio files in the `audio_data/` and `background_sound/` folders.
   
4. **Train the Model**:
   Use the provided `WWRTD.ipynb` notebook or script to preprocess the data, train the model, and convert it to TFLite format.

   ```bash
   python train_model.py
   ```

5. **Deploy the Model**:
   - Use the TFLite model for real-time wake word detection on your microphone input.

## How to Use
Once the model is trained and saved as a TFLite file (e.g., `WWD.tflite`), use the `real_time_detection.py` script to run the system and detect the Wake Word in real-time.

```bash
python real_time_detection.py
```

The system will continuously record audio, process it in chunks, and output a detection result (`Wake Word Detected` or `Wake Word NOT Detected`). Press `Ctrl+C` to stop the detection.

### Example Output:
```text
Recording... Press i deux fois to stop.
.....
Wake Word Detected: 1
```

## Results
Here is an example of how the model performs on an audio segment:

![Waveform and MFCC Visualization](/result_image.png)

### Performance:
- **Accuracy**: The model achieved an accuracy of X% on the test set.
- **Inference Time**: The TFLite model performs inference in less than X milliseconds per sample on a Raspberry Pi.

## Contributing
We welcome contributions to improve this project. Feel free to fork the repository, create a branch, and submit pull requests. Please ensure that your code follows the existing coding style and includes appropriate tests.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
