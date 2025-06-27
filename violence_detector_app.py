import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import threading
import time
import asyncio
import telegram
import os
import random
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import TimeDistributed, LSTM, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from collections import deque

#Config 
#Replace with your actual bot token and chat ID(use chatinfobot in telegram)
TELEGRAM_BOT_TOKEN = "xxxxxxxxxx:AAAccccXXXXwwwwweeeee"
TELEGRAM_CHAT_ID = "-100000xxxxx"     

#Data Paths
DATASET_ROOT = 'D:\draftt\data\Real Life Violence Dataset' # <--- IMPORTANT: SET THIS TO YOUR DOWNLOADED DATASET PATH
MODEL_SAVE_PATH = 'violence_detector_mobilenet_lstm.h5'

#Hyperparams  

FRAME_HEIGHT, FRAME_WIDTH = 128, 128
SEQUENCE_LENGTH = 16  
BATCH_SIZE = 8        
EPOCHS = 20           
VIOLENCE_THRESHOLD = 0.75  
ALERT_COOLDOWN_SECONDS = 30  

#Global variables for detection and display

violence_model = None
cap = None
processing_thread = None
stop_event = threading.Event()
last_alert_time = 0

#Telegram Bot Function  

async def send_telegram_alert(message_text, image_path=None, video_path=None):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram bot token or chat ID not set. Cannot send alert.")
        return
    try:
        bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message_text)
        if image_path and os.path.exists(image_path):
            with open(image_path, 'rb') as image:
                await bot.send_photo(chat_id=TELEGRAM_CHAT_ID, photo=image)
            #Optional: Delete the image after sending to clean up
            os.remove(image_path)
        if video_path and os.path.exists(video_path):
            with open(video_path, 'rb') as video:
                await bot.send_video(chat_id=TELEGRAM_CHAT_ID, video=video)
            #Optional: Delete the video after sending
            os.remove(video_path)
        print("Telegram alert sent successfully!")
    except telegram.error.TelegramError as e:
        print(f"Error sending Telegram alert: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during Telegram alert: {e}")

def run_telegram_async(func, *args, **kwargs):
    def wrapper():
        asyncio.run(func(*args, **kwargs))
    threading.Thread(target=wrapper).start()

#Model Definition 

def build_violence_detection_model(input_shape=(SEQUENCE_LENGTH, FRAME_HEIGHT, FRAME_WIDTH, 3), num_classes=1):
    #Load a pre-trained 2D CNN (MobileNetV2) 
    base_cnn = MobileNetV2(
        weights='imagenet',
        include_top=False,  
        input_shape=(input_shape[1], input_shape[2], input_shape[3]) # (height, width, channels)
    )
    #Freeze the base CNN layers initially. Fine-tuning later (optional)
    base_cnn.trainable = False

    model = Sequential([
        #Apply the base CNN to each frame in the sequence
        TimeDistributed(base_cnn, input_shape=input_shape),
        TimeDistributed(GlobalAveragePooling2D()),                     #Pooling  
        LSTM(128, return_sequences=False),                             #LSTM  
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='sigmoid') #Sigmoid for binary classification (violence/non-violence)
    ])

    model.compile(optimizer=Adam(learning_rate=0.0001),  
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    return model

#Data Preprocessing
def get_video_paths_and_labels(dataset_root):
    violence_dir = os.path.join(dataset_root, 'Violence')
    non_violence_dir = os.path.join(dataset_root, 'NonViolence')

    if not os.path.exists(violence_dir) or not os.path.exists(non_violence_dir):
        raise FileNotFoundError(f"Dataset directories not found. Please ensure '{DATASET_ROOT}' contains 'Violence' and 'NonViolence' folders.")

    video_paths = []
    labels = []

    print("Collecting violent video paths...")
    for filename in tqdm(os.listdir(violence_dir)):
        if filename.endswith(('.mp4', '.avi', '.mov')):
            video_paths.append(os.path.join(violence_dir, filename))
            labels.append(1) #1 for violence

    print("Collecting non-violent video paths...")
    for filename in tqdm(os.listdir(non_violence_dir)):
        if filename.endswith(('.mp4', '.avi', '.mov')):
            video_paths.append(os.path.join(non_violence_dir, filename))
            labels.append(0) #0 for non-violence

    return video_paths, labels

def extract_frames_from_video_for_sequence(video_path, start_frame, sequence_length, target_size):
    """
    Extracts a specific sequence of frames from a video, resizes, and normalizes.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Warning: Could not open video {video_path}. Skipping.")
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)  

    frames = []
    for _ in range(sequence_length):
        ret, frame = cap.read()
        if not ret:
            while len(frames) < sequence_length:
                black_frame = np.zeros((*target_size, 3), dtype=np.float32)
                frames.append(black_frame)
            break

        frame = cv2.resize(frame, target_size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
        frame = frame / 255.0  
        frames.append(frame)
    cap.release()

    if len(frames) == sequence_length:
        return np.array(frames, dtype=np.float32)
    return None  


class VideoSequenceDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, video_paths, labels, sequence_length, target_size, batch_size, shuffle=True):
        self.video_paths = video_paths
        self.labels = labels
        self.sequence_length = sequence_length
        self.target_size = target_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sequence_indices = [] #Stores (video_path_idx, start_frame_idx) for each potential sequence

        print("Indexing video sequences for generator...")
        for i, video_path in enumerate(tqdm(self.video_paths)):
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Warning: Could not open video {video_path}. Skipping for indexing.")
                continue
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            for start_frame_idx in range(0, total_frames - self.sequence_length + 1, self.sequence_length // 2):
                self.sequence_indices.append((i, start_frame_idx))  

        print(f"Total sequence indices generated: {len(self.sequence_indices)}")
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.sequence_indices) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_sequence_indices = [self.sequence_indices[k] for k in indexes]
        X, y = self.__data_generation(batch_sequence_indices)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.sequence_indices))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_sequence_indices):
        X = np.empty((self.batch_size, self.sequence_length, self.target_size[0], self.target_size[1], 3), dtype=np.float32)
        y = np.empty((self.batch_size), dtype=int)

        for i, (video_path_idx, start_frame_idx) in enumerate(batch_sequence_indices):
            video_path = self.video_paths[video_path_idx]
            label = self.labels[video_path_idx] 
            
            sequence_frames = extract_frames_from_video_for_sequence(video_path, start_frame_idx, self.sequence_length, self.target_size)
            
            if sequence_frames is not None:
                X[i,] = sequence_frames
                y[i] = label
            else: 
                X[i,] = np.zeros((self.sequence_length, self.target_size[0], self.target_size[1], 3), dtype=np.float32)
                y[i] = 0 
                

        return X, y

#Model Training

def train_model():
    global violence_model

    if not os.path.exists(DATASET_ROOT):
        messagebox.showerror("Dataset Error", f"Dataset not found at '{DATASET_ROOT}'.\n"
                                           "Please download 'Real Life Violence Situations Dataset' from Kaggle "
                                           "and extract it to this path.")
        return

    #Check if model already exists and ask to retrain
    if os.path.exists(MODEL_SAVE_PATH):
        if not messagebox.askyesno("Model Exists", "Model already trained. Do you want to retrain it?"):
            try:
                violence_model = load_model(MODEL_SAVE_PATH)
                messagebox.showinfo("Model Loaded", "Pre-trained model loaded successfully.")
                #Update status  
                status_label.config(text="Status: Model loaded from disk.", style="Green.TLabel")
                return
            except Exception as e:
                messagebox.showerror("Model Load Error", f"Failed to load existing model: {e}. Retraining.")
                status_label.config(text=f"Error loading model: {e}", style="Red.TLabel")


    status_label.config(text="Status: Preparing data for training...", style="Blue.TLabel")
    root.update_idletasks()
    root.update()

    video_paths, labels = get_video_paths_and_labels(DATASET_ROOT)

    #train and validation sets
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        video_paths, labels, test_size=0.2, random_state=42, stratify=labels 
    )

    try:
        train_generator = VideoSequenceDataGenerator(train_paths, train_labels, SEQUENCE_LENGTH, (FRAME_HEIGHT, FRAME_WIDTH), BATCH_SIZE)
        val_generator = VideoSequenceDataGenerator(val_paths, val_labels, SEQUENCE_LENGTH, (FRAME_HEIGHT, FRAME_WIDTH), BATCH_SIZE)
    except Exception as e:
        messagebox.showerror("Data Generator Error", f"Error creating data generators: {e}. Check dataset integrity.")
        status_label.config(text=f"Status: Data generator error: {e}", style="Red.TLabel")
        return

    if len(train_generator) == 0 or len(val_generator) == 0:
        messagebox.showerror("Data Error", "Not enough sequences generated for training or validation. Check your dataset and sequence length.")
        status_label.config(text="Status: Not enough data sequences.", style="Red.TLabel")
        return

    status_label.config(text="Status: Building and training model...", style="Blue.TLabel")
    root.update_idletasks()
    root.update()

    violence_model = build_violence_detection_model() 
     
    checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_loss', save_best_only=True, mode='min', verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

    try:
        history = violence_model.fit(
            train_generator,
            epochs=EPOCHS,
            validation_data=val_generator,
            callbacks=[checkpoint, early_stopping, reduce_lr],
            #workers=4, #Adjust based on CPU cores
            #use_multiprocessing=True for faster data loading
        )
        messagebox.showinfo("Training Complete", "Model training finished successfully. Best model saved.")
        status_label.config(text="Status: Model trained and ready.", style="Green.TLabel")
    except Exception as e:
        messagebox.showerror("Training Error", f"An error occurred during training: {e}")
        status_label.config(text="Status: Training failed.", style="Red.TLabel")

#Core Detection Logic for GUI 
def process_video_stream(video_source):
    global cap, last_alert_time, violence_model, stop_event
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        messagebox.showerror("Error", "Could not open video source.")
        return

    frame_buffer = deque(maxlen=SEQUENCE_LENGTH) #Store frames for sequence prediction
    violence_detected_frames_count = 0 #Counter for consecutive violent frames
    REQUIRED_CONSECUTIVE_VIOLENT_FRAMES = 5 #Number of consecutive violent frames to trigger alert

    print("Starting video stream processing...")
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret: 
            if isinstance(video_source, str) and video_source.endswith(('.mp4', '.avi', '.mov')):
                print("End of video file.")
                break
            continue

        #Convert frame to RGB for PIL and display
        frame_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_display)
        imgtk = ImageTk.PhotoImage(image=img.resize((640, 480)))  

        panel.imgtk = imgtk
        panel.config(image=imgtk) 
        frame_model_input = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        frame_model_input = cv2.cvtColor(frame_model_input, cv2.COLOR_BGR2RGB) / 255.0 #Normalize
        frame_buffer.append(frame_model_input)

        if len(frame_buffer) == SEQUENCE_LENGTH: 
            processed_sequence = np.array(frame_buffer, dtype=np.float32)  
            processed_sequence = np.expand_dims(processed_sequence, axis=0)  

            if violence_model:
                try:
                    violence_prob = violence_model.predict(processed_sequence, verbose=0)[0][0]  
                    status_label.config(text=f"Violence Probability: {violence_prob:.4f}", style="Blue.TLabel")  

                    current_time = time.time()
                    if violence_prob > VIOLENCE_THRESHOLD:
                        violence_detected_frames_count += 1
                        if violence_detected_frames_count >= REQUIRED_CONSECUTIVE_VIOLENT_FRAMES:
                            status_label.config(text="Status: VIOLENCE DETECTED!", style="Red.TLabel")  
                            if (current_time - last_alert_time) > ALERT_COOLDOWN_SECONDS:
                                alert_message = f"🚨 VIOLENCE ALERT! Probability: {violence_prob:.2f}" 
                                alert_image_path = "detected_violence.jpg"
                             
                                cv2.imwrite(alert_image_path, cv2.cvtColor(frame_display, cv2.COLOR_RGB2BGR))
                                run_telegram_async(send_telegram_alert, alert_message, image_path=alert_image_path)
                                last_alert_time = current_time
                                violence_detected_frames_count = 0  
                    else:
                        violence_detected_frames_count = 0
                        status_label.config(text="Status: Normal", style="Green.TLabel")  
                except Exception as e:
                    status_label.config(text=f"Prediction Error: {e}", style="Orange.TLabel")  
                    print(f"Prediction Error: {e}")
            else:
                status_label.config(text="Status: Model not loaded.", style="Orange.TLabel")  
        else:
            status_label.config(text=f"Status: Buffering frames... ({len(frame_buffer)}/{SEQUENCE_LENGTH})", style="Gray.TLabel")  

        root.update_idletasks()
        root.update()

    cap.release()
    cv2.destroyAllWindows()
    panel.config(image='')
    status_label.config(text="Status: Idle", style="Black.TLabel")  
    if not stop_event.is_set():  
        messagebox.showinfo("Finished", "Video processing complete.")


#GUI  
def start_webcam():
    global processing_thread, stop_event, last_alert_time
    if processing_thread and processing_thread.is_alive():
        messagebox.showinfo("Info", "Already processing a stream. Please stop first.")
        return

    if violence_model is None:
        messagebox.showerror("Error", "Violence detection model not loaded. Please train or load model first.")
        return

    stop_event.clear()
    last_alert_time = 0  
    processing_thread = threading.Thread(target=process_video_stream, args=(0,))  
    processing_thread.daemon = True
    processing_thread.start()
    status_label.config(text="Status: Webcam active", style="Blue.TLabel")  

def upload_video():
    global processing_thread, stop_event, last_alert_time
    if processing_thread and processing_thread.is_alive():
        messagebox.showinfo("Info", "Already processing a stream. Please stop first.")
        return

    if violence_model is None:
        messagebox.showerror("Error", "Violence detection model not loaded. Please train or load model first.")
        return

    video_path = filedialog.askopenfilename(
        title="Select a video file",
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
    )
    if video_path:
        stop_event.clear()
        last_alert_time = 0  
        processing_thread = threading.Thread(target=process_video_stream, args=(video_path,))
        processing_thread.daemon = True
        processing_thread.start()
        status_label.config(text=f"Status: Processing '{os.path.basename(video_path)}'", style="Blue.TLabel")  

def stop_processing():
    global cap, processing_thread, stop_event
    if processing_thread and processing_thread.is_alive():
        stop_event.set()
        processing_thread.join(timeout=5)  
        if processing_thread.is_alive():
            print("Warning: Processing thread did not terminate gracefully.")
        status_label.config(text="Status: Stopped", style="Black.TLabel")  
        if cap:
            cap.release()
            cv2.destroyAllWindows()
        panel.config(image='')
    else:
        messagebox.showinfo("Info", "No active stream to stop.")

def on_closing():
    stop_processing()
    root.destroy()

#Tkinter  
root = tk.Tk()
root.title("Violence Detection System")
root.geometry("800x700") 
style = ttk.Style()
style.configure("Black.TLabel", foreground="black")
style.configure("Green.TLabel", foreground="green")
style.configure("Blue.TLabel", foreground="blue")
style.configure("Red.TLabel", foreground="red")
style.configure("Orange.TLabel", foreground="orange")
style.configure("Gray.TLabel", foreground="gray")
 
button_frame = ttk.Frame(root, padding="10")
button_frame.pack(side=tk.TOP, fill=tk.X)

train_btn = ttk.Button(button_frame, text="Train/Load Model", command=train_model)
train_btn.pack(side=tk.LEFT, padx=5, pady=5)

start_webcam_btn = ttk.Button(button_frame, text="Start Webcam", command=start_webcam)
start_webcam_btn.pack(side=tk.LEFT, padx=5, pady=5)

upload_video_btn = ttk.Button(button_frame, text="Upload Video", command=upload_video)
upload_video_btn.pack(side=tk.LEFT, padx=5, pady=5)

stop_btn = ttk.Button(button_frame, text="Stop", command=stop_processing)
stop_btn.pack(side=tk.LEFT, padx=5, pady=5)

panel = tk.Label(root, bg="black")
panel.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

status_label = ttk.Label(root, text="Status: Idle", font=("Helvetica", 12), style="Black.TLabel")
status_label.pack(side=tk.BOTTOM, pady=10)

if os.path.exists(MODEL_SAVE_PATH):
    try:
        violence_model = load_model(MODEL_SAVE_PATH)
        status_label.config(text="Status: Model loaded from disk.", style="Green.TLabel") 
        print(f"Model loaded from {MODEL_SAVE_PATH}")
    except Exception as e:
        status_label.config(text=f"Error loading model: {e}", style="Red.TLabel")  
        print(f"Error loading model: {e}")
else:
    status_label.config(text="Status: No model found. Click 'Train/Load Model'.", style="Orange.TLabel")  


root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()
