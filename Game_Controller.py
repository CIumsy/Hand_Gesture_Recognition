import time
import numpy as np
import websocket
import socket
import threading
from queue import Queue
import tensorflow as tf
import joblib
from pylsl import StreamInfo, StreamOutlet
import keyboard  # Import the keyboard library

# Load prediction model and scaler (trained on 18 features for 5 classes)
model = tf.keras.models.load_model("model.h5")
scaler = joblib.load("scaler.pkl")

# iEMG threshold for rest detection
IEMG_THRESHOLD = 9.9  # If iEMG is below this, no keystrokes are sent

# Key hold and release parameters
HOLD_DURATION = 0.3  # Release key if the same gesture is not detected within 300ms
DEBOUNCE_DELAY = 0.2  # 200ms delay after a keystroke is sent
IGNORE_AFTER_RIGHT = 0.3  # Ignore left and down keystrokes for 300ms after right gesture

# Queue and prediction thread
data_queue = Queue()

# Variables to track the last gesture and key state
last_gesture = None
last_key_time = None
key_held = False
last_prediction = None
consistent_predictions = 0  # Counter for consistent predictions
last_keystroke_time = 0  # Time when the last keystroke was sent
ignore_left_until = 0  # Time until which left keystrokes are ignored
ignore_down_until = 0  # Time until which down keystrokes are ignored

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

def prediction_worker():
    global last_gesture, last_key_time, key_held, last_prediction, consistent_predictions, last_keystroke_time, ignore_left_until, ignore_down_until
    while True:
        try:
            data_chunk = data_queue.get()
            if data_chunk is None:
                break

            # Calculate iEMG value
            iEMG_value = np.sum(np.abs(np.array(data_chunk).flatten()))
            print(f"iEMG: {iEMG_value:.4f}")

            # Check if iEMG is below the threshold (rest state)
            if iEMG_value < IEMG_THRESHOLD:
                print("Hand at rest. No keystroke sent.")
                # Release the key if it is currently held
                if key_held:
                    keyboard.release(last_gesture)
                    key_held = False
                    last_gesture = None
                    last_key_time = None
                consistent_predictions = 0  # Reset consistent predictions counter
            else:
                # Process and predict gesture
                predicted_class = process_and_predict(data_chunk, window_size=WINDOW_SIZE)
                print(f"Predicted: {predicted_class}")

                # Check if the same gesture is predicted twice in a row
                if predicted_class == last_prediction:
                    consistent_predictions += 1
                else:
                    consistent_predictions = 1  # Reset counter if prediction changes
                last_prediction = predicted_class

                # Only send keystrokes if the same gesture is predicted twice in a row
                if consistent_predictions >= 2:
                    # Map predicted_class to key
                    gesture_to_key = {
                        0: 'left',
                        1: 'right',
                        2: 'up',
                        3: 'down'
                    }
                    current_key = gesture_to_key.get(predicted_class, None)

                    # Handle key press and hold
                    if current_key is not None:
                        # Ignore left and down keystrokes for 300ms after right gesture
                        if (current_key == 'left' and time.time() < ignore_left_until) or \
                           (current_key == 'down' and time.time() < ignore_down_until):
                            print(f"Ignoring {current_key} keystroke after right gesture.")
                            continue

                        # Check if 200ms has passed since the last keystroke
                        if (time.time() - last_keystroke_time) >= DEBOUNCE_DELAY:
                            if current_key != last_gesture:
                                # Release the previous key if a new gesture is detected
                                if key_held:
                                    keyboard.release(last_gesture)
                                    key_held = False
                                # Press and hold the new key
                                keyboard.press(current_key)
                                key_held = True
                                last_gesture = current_key
                                last_key_time = time.time()
                                last_keystroke_time = time.time()  # Update last keystroke time

                                # Set ignore timers if right gesture is performed
                                if current_key == 'right':
                                    ignore_left_until = time.time() + IGNORE_AFTER_RIGHT
                                    ignore_down_until = time.time() + IGNORE_AFTER_RIGHT
                            else:
                                # Update the timer for the same gesture
                                last_key_time = time.time()
                    else:
                        # Invalid gesture, release the key if held
                        if key_held:
                            keyboard.release(last_gesture)
                            key_held = False
                            last_gesture = None
                            last_key_time = None

            # Check if the key should be released due to timeout
            if key_held and (time.time() - last_key_time) >= HOLD_DURATION:
                keyboard.release(last_gesture)
                key_held = False
                last_gesture = None
                last_key_time = None
                print("Key released due to timeout.")
        except Exception as e:
            print(f"Error in prediction worker: {e}")
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