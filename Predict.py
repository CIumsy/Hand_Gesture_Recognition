import time
import math
import numpy as np
import websocket
import socket
import threading
from queue import Queue
import pandas as pd
import tensorflow as tf
import joblib
from pylsl import StreamInfo, StreamOutlet

# Load prediction model and scaler (trained on 18 features for 5 classes)
model = tf.keras.models.load_model("model.h5")
scaler = joblib.load("scaler.pkl")


def calculate_features(data, window_size):
    """
    Computes 6 features per channel (iEMG, MAV, STD, MF, RMS, ZC) over the last window_size samples.
    Sampling rate is 250 Hz.
    Returns a 1D NumPy array of 18 features (for 3 channels).
    """
    data = np.array(data)  # shape: (n_samples, 3)
    features = []
    sampling_rate = 250  # Hz
    for channel in data.T:  # iterate over each channel
        window = channel[-window_size:] if len(channel) >= window_size else channel
        # iEMG: sum of absolute values
        iEMG = np.sum(np.abs(window))
        # MAV: mean of absolute values
        MAV = np.mean(np.abs(window))
        # STD: standard deviation
        std_val = np.std(window)
        # MF: mean frequency using FFT
        fft_vals = np.abs(np.fft.rfft(window))
        freqs = np.fft.rfftfreq(len(window), d=1/sampling_rate)
        mean_freq = np.sum(freqs * fft_vals) / np.sum(fft_vals) if np.sum(fft_vals) != 0 else 0
        # RMS: root mean square
        RMS = np.sqrt(np.mean(np.square(window)))
        # ZC: zero crossing count
        zc = np.sum(np.abs(np.diff(np.sign(window))) > 0)
        features.extend([iEMG, MAV, std_val, mean_freq, RMS, zc])
    return np.array(features)

def process_and_predict(data, window_size):
    features = calculate_features(data, window_size)
    scaled_features = scaler.transform(features.reshape(1, -1))
    predictions = model.predict(scaled_features)
    predicted_class = np.argmax(predictions, axis=1)[0]
    return predicted_class

# Queue and prediction thread
data_queue = Queue()

def prediction_worker():
    while True:
        try:
            data_chunk = data_queue.get()
            if data_chunk is None:
                break
            predicted_class = process_and_predict(data_chunk, window_size=WINDOW_SIZE)
            print(predicted_class)
        except Exception as e:
            # Optionally, print(e) for debugging
            pass
        finally:
            data_queue.task_done()

prediction_thread = threading.Thread(target=prediction_worker, daemon=True)
prediction_thread.start()

# Setup LSL outlet and WebSocket
stream_name = 'NPG'
lsl_info = StreamInfo(stream_name, 'EXG', 3, 250, 'float32', 'uid007')
outlet = StreamOutlet(lsl_info)
ws = websocket.WebSocket()
ws.connect("ws://" + socket.gethostbyname("multi-emg.local") + ":81")

block_size = 13
packet_size = 0  
data_size = 0
sample_size = 0
previousSampleNumber = -1
previousData = []
start_time = time.time()

def calculate_rate(quantity, elapsed_time):
    return quantity / elapsed_time

def normalize_sample(sample):
    a = 2**12
    return (sample - a/2) * (2/a)

# Buffer for accumulating samples (each sample is 3 channels)
sample_buffer = []
WINDOW_SIZE = 25  # 125 samples = 0.5 sec at 250 Hz

while True:
    data_recv = ws.recv()
    data_size += len(data_recv)
    current_time = time.time()
    elapsed_time = current_time - start_time

    if elapsed_time >= 1.0:
        packet_size = 0
        sample_size = 0
        data_size = 0
        start_time = current_time

    if data_recv and (isinstance(data_recv, list) or isinstance(data_recv, bytes)):
        packet_size += 1
        for blockLocation in range(0, len(data_recv), block_size):
            sample_size += 1
            block = data_recv[blockLocation:blockLocation + block_size]
            if len(block) < block_size:
                continue
            sample_number = block[0]
            channel_data = []
            for channel in range(0, 3):
                channel_offset = 1 + (channel * 2)
                sample = int.from_bytes(block[channel_offset:channel_offset + 2], byteorder='big', signed=True)
                normalized_sample = normalize_sample(sample)
                channel_data.append(normalized_sample)
            if previousSampleNumber == -1:
                previousSampleNumber = sample_number
                previousData = channel_data
            else:
                if sample_number - previousSampleNumber > 1:
                    pass
                elif sample_number == previousSampleNumber:
                    pass
                else:
                    previousSampleNumber = sample_number
                    previousData = channel_data

            outlet.push_sample(channel_data)
            sample_buffer.append(channel_data)
            if len(sample_buffer) >= WINDOW_SIZE:
                data_queue.put(sample_buffer.copy())
                sample_buffer.clear()
