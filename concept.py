
# Import necessary libraries
import whisper
import pyaudio
import numpy as np
import threading
import queue
import time
from transformers import Gemma3nForConditionalGeneration,AutoModel, AutoProcessor, BertTokenizer, BertModel,BertConfig
from torch.amp import autocast, GradScaler
import pybullet as p
import pybullet_data
from PIL import Image
import torch
import torch.nn as nn
import os
import warnings

# Suppress warnings for a cleaner output.
warnings.filterwarnings("ignore")

try:
    import noisereduce as nr
    NOISE_REDUCTION_AVAILABLE = True
except ImportError:
    NOISE_REDUCTION_AVAILABLE = False
    print("noisereduce not available, skipping noise reduction")

try:
    from scipy.signal import butter, filtfilt
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("scipy not available, skipping bandpass filtering")

class Router(nn.Module):
    """
    A lightweight text classification model based on a truncated multilingual BERT.
    This model is designed to be fast and efficient for routing/classification tasks.
    """
    def __init__(self, model_name="bert-base-multilingual-uncased", num_labels=2, num_layers=2,dropout=0.2):
        super(Router, self).__init__()

        # Load the configuration from the pre-trained model and set custom dropout rates.
        self.config = BertConfig.from_pretrained(model_name)
        self.config.hidden_dropout_prob = dropout
        self.config.attention_probs_dropout_prob = dropout

        # Load the pre-trained BERT model with the modified config.
        self.bert = BertModel.from_pretrained(model_name, config=self.config)

        # Truncate the BERT model to only use the first `num_layers`.
        # This significantly reduces the model size and inference time.
        self.bert.encoder.layer = nn.ModuleList(
            self.bert.encoder.layer[:num_layers]
        )

        # Define the classification head layers.
        self.dropout = nn.Dropout(dropout)
        self.norm=nn.RMSNorm(self.config.hidden_size)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        Defines the forward pass of the model.
        """
        # Pass inputs through the truncated BERT model.
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # The 'pooled_output' is the representation of the [CLS] token,
        # which summarizes the entire input sequence.
        pooled_output = outputs[1]

        # Apply dropout, normalization, and the final classification layer.
        pooled_output = self.dropout(pooled_output)
        pooled_output = self.norm(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
    



# Seting up pybullet simulation environment
def setup_world():
    """Sets up the static environment: ground, table, and robot."""
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")
    p.loadURDF("table/table.urdf", basePosition=[0, 0, 0])
    # Load the Franka Panda robot with a fixed base.
    robot_id = p.loadURDF("franka_panda/panda.urdf", basePosition=[-0.45, 0, 0.625], useFixedBase=True)
    return robot_id

def reset_environment():
    """Resets the entire simulation and rebuilds the world for a new episode."""
    # Resetting the simulation provides a clean slate for each episode.
    p.resetSimulation()
    
    # Re-create the static world (ground, table, robot).
    robot_id = setup_world()

    # Set the robot's joints to a default starting pose.
    for i, angle in enumerate([0, -0.4, 0, -2.4, 0, 2.0, 0.8]):
        p.resetJointState(robot_id, i, targetValue=angle, targetVelocity=0)

    # Set the gripper fingers (joints 9 and 10) to an open position.
    p.setJointMotorControl2(robot_id, 9, p.POSITION_CONTROL, 0.04, force=50)
    p.setJointMotorControl2(robot_id, 10, p.POSITION_CONTROL, 0.04, force=50)

    # Define two zones on the table for object placement.
    left_zone = {'x': (0.1, 0.3), 'y': (0.1, 0.3)}
    right_zone = {'x': (0.1, 0.3), 'y': (-0.3, -0.1)}
    # Randomly assign the cube and tray to these zones.
    cube_zone, tray_zone = (left_zone, right_zone) if np.random.rand() > 0.5 else (right_zone, left_zone)

    # Spawn the cube at a random position within its assigned zone.
    initial_cube_pos = [np.random.uniform(*cube_zone['x']), np.random.uniform(*cube_zone['y']), 0.64]
    cube_id = p.loadURDF("cube_small.urdf", basePosition=initial_cube_pos)

    # Give the cube a random color for better generalization.
    p.changeVisualShape(cube_id, -1, rgbaColor=[np.random.uniform(0.2, 0.8), np.random.uniform(0.2, 0.8), np.random.uniform(0.2, 0.8), 1.0])

    # Spawn the tray at a random position within its assigned zone.
    tray_pos = [np.random.uniform(*tray_zone['x']), np.random.uniform(*tray_zone['y']), 0.63]
    tray_id = p.loadURDF("tray/tray.urdf", basePosition=tray_pos, globalScaling=0.6)

    # Run the simulation for a short period to let objects settle.
    for _ in range(50):
        p.stepSimulation()
        
    return robot_id, cube_id, tray_id

# Initialize PyBullet in GUI mode for visualization.
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
robot_id, cube_id, tray_id = reset_environment()
    
# Configure the GUI camera for a consistent view.
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
p.resetDebugVisualizerCamera(0.9, 60, -35, [0.1, 0, 0.65])



# Setup all models
print("Setting up models...")
# Router model for command classification
print("Loading Router model...")
router = Router(model_name="bert-base-multilingual-uncased", num_labels=2, num_layers=6, dropout=0.2)
router.load_state_dict(torch.load('models/router_best.p'))
# router.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
router = router.to(device)
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-uncased")
print("Router model loaded.")
# VLM model for visual-language tasks
# Load the pre-trained model from the specified ID.
print("Loading VLM model...")
vlm_model = Gemma3nForConditionalGeneration.from_pretrained("google/gemma-3n-e2b-it", torch_dtype=torch.bfloat16,).cuda().eval()
# Load the appropriate processor associated with the model.
vlm_processor = AutoProcessor.from_pretrained("google/gemma-3n-e2b-it")
print("VLM model loaded.")
# VLA model for visual-language-action tasks
print("Loading VLA model...")
vla_model = AutoModel.from_pretrained("models/VLA_model", trust_remote_code=True, torch_dtype=torch.bfloat16).eval().to("cuda")
vla_processor = AutoProcessor.from_pretrained("models/VLA_model", trust_remote_code=True)
print("VLA model loaded.")





def vlm_head(task,image):
    """
    Process a task with the Vision-Language Model (VLM) using text and image inputs.
    Args:
        task: Text description of the task.
        image: PIL image from the simulation environment.
    Returns:
        Decoded text response from the VLM model.
    """
    # Define system and user messages for VLM input
    messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant for a user operating a mechanical hand. Answer questions clearly and provide relevant information based on user needs. If a mechanical hand is visible in the image, ignore it and focus on describing or analyzing other parts of the image as requested."}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": task}
                ]
            }
        ]
    # Use the processor to apply the chat template, tokenize, and convert to PyTorch tensors.
    inputs = vlm_processor.apply_chat_template(
        messages,
        add_generation_prompt=True, 
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(vlm_model.device, dtype=torch.bfloat16)

    # Get the length of the input tokens to later separate it from the generated output.
    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        # Generate a response from the model.
        generation = vlm_model.generate(**inputs, max_new_tokens=256, do_sample=False)
        # Slice the output tensor to get only the newly generated tokens.
        generation = generation[0][input_len:]
    # Decode the generated token IDs back into a human-readable string.
    decoded = vlm_processor.decode(generation, skip_special_tokens=True)

    return decoded

def vla_head(task,image):
    """
    Process a task with the Vision-Language-Action (VLA) model to generate robotic actions.
    Args:
        task: Text description of the task.
        image: PIL image from the simulation environment.
    Returns:
        List of action commands for the robot.
    """
    # Prepare inputs for VLA model using the processor
    inputs = vla_processor(images=[image], text=task, unnorm_key="my_robot_dataset/1.0.0", return_tensors="pt").to("cuda")
    # Generate action predictions without gradient computation
    with torch.no_grad():
        generation_outputs = vla_model.predict_action(inputs)
    # Decode actions using the processor
    actions_chunk = vla_processor.decode_actions(generation_outputs, unnorm_key="my_robot_dataset/1.0.0")['actions']
    return actions_chunk


class RealTimeWhisperSTT:
    def __init__(self, 
                 model_size="medium.en",
                 device="cuda",
                 sample_rate=16000,
                 chunk_duration=3.0,
                 overlap_duration=0.5,
                 energy_threshold=0.001,
                 noise_reduction=True):
        
        # Initialize Whisper model
        print("Loading Whisper model...")
        self.model = whisper.load_model(model_size, device=device, download_root="models")
        print(f"Model loaded on {device}")
        
        # Audio parameters
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.overlap_duration = overlap_duration
        self.chunk_size = int(sample_rate * chunk_duration)
        self.overlap_size = int(sample_rate * overlap_duration)
        self.energy_threshold = energy_threshold
        self.noise_reduction = noise_reduction and NOISE_REDUCTION_AVAILABLE
        
        # Audio processing
        self.audio_queue = queue.Queue()
        self.text_queue = queue.Queue()
        self.is_recording = False
        self.background_noise = None
        self.background_energy = 0.0
        
        # # Hallucination prevention
        self.common_hallucinations = {
            "thank you", "thanks", "thank you for watching", "thank you for listening",
            "subscribe", "like and subscribe", "please subscribe", "hit the like button",
            "see you next time", "see you later", "goodbye", "bye", "see you soon",
            "that's all for today", "that's it for now", "thanks for watching",
            "you", ".", "?", "!", "", " ", "the", "a", "and", "or", "but", "so"
        }
        
        # Initialize PyAudio
        self.p = pyaudio.PyAudio()
        self.stream = None
        
        # Threading
        self.processing_thread = None
    
    def _get_audio_devices(self):
        """List available audio input devices"""
        print("\nAvailable audio input devices:")
        for i in range(self.p.get_device_count()):
            info = self.p.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                print(f"  {i}: {info['name']} - {info['maxInputChannels']} channels")
        return self.p.get_default_input_device_info()['index']
    
    def _butter_bandpass_filter(self, data, lowcut=300, highcut=3400, order=4):
        """Apply bandpass filter to focus on speech frequencies"""
        if not SCIPY_AVAILABLE:
            return data
            
        try:
            data = data.astype(np.float32)
            nyquist = 0.5 * self.sample_rate
            low = lowcut / nyquist
            high = highcut / nyquist
            
            # Ensure frequencies are within valid range
            if low <= 0:
                low = 0.01
            if high >= 1:
                high = 0.99
            if low >= high:
                return data
                
            b, a = butter(order, [low, high], btype='band')
            filtered_data = filtfilt(b, a, data)
            return filtered_data.astype(np.float32)
        except Exception as e:
            print(f"Filter error: {e}, skipping bandpass filter")
            return data.astype(np.float32)
    
    def _apply_noise_reduction(self, audio):
        """Apply noise reduction if available"""
        if not self.noise_reduction or self.background_noise is None:
            return audio.astype(np.float32)
            
        try:
            audio = audio.astype(np.float32)
            background_noise = self.background_noise.astype(np.float32)
            
            # Ensure compatible lengths
            if len(audio) > len(background_noise):
                noise_sample = np.tile(background_noise, 
                                     (len(audio) // len(background_noise) + 1))[:len(audio)]
            else:
                noise_sample = background_noise[:len(audio)]
            
            reduced_audio = nr.reduce_noise(
                y=audio, 
                sr=self.sample_rate,
                y_noise=noise_sample,
                prop_decrease=0.6,
                stationary=False
            )
            return reduced_audio.astype(np.float32)
        except Exception as e:
            print(f"Noise reduction error: {e}, skipping noise reduction")
            return audio.astype(np.float32)
    
    def _normalize_audio(self, audio):
        """Normalize audio to [-1, 1] range"""
        audio = audio.astype(np.float32)
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            return (audio / max_val).astype(np.float32)
        return audio.astype(np.float32)
    
    def _preprocess_audio(self, audio):
        """
        Apply preprocessing steps to audio data (filtering, noise reduction, normalization).
        Args:
            audio: Raw audio data as numpy array.
        Returns:
            Preprocessed audio as numpy array.
        """
        try:
            # Convert to float32 and normalize
            audio = audio.astype(np.float32) / 32768.0
            
            # Check for valid audio data
            if len(audio) == 0 or np.all(audio == 0):
                return audio
            
            # Apply bandpass filter
            audio = self._butter_bandpass_filter(audio)
            
            # Apply noise reduction
            audio = self._apply_noise_reduction(audio)
            
            # Final normalization
            audio = self._normalize_audio(audio)
            
            return audio
        except Exception as e:
            print(f"Preprocessing error: {e}")
            # Return basic normalized audio if preprocessing fails
            audio_float = audio.astype(np.float32) / 32768.0
            return self._normalize_audio(audio_float)
    
    def _simple_voice_detection(self, audio):
        """Simple but effective voice activity detection"""
        audio = audio.astype(np.float32)
        
        # Calculate RMS energy
        rms_energy = np.sqrt(np.mean(audio**2))
        
        # Dynamic threshold based on background
        threshold = max(self.energy_threshold, self.background_energy * 2.0)
        
        # Check if energy is above threshold
        energy_check = rms_energy > threshold
        
        # Check for reasonable audio characteristics
        if energy_check:
            # Check for dynamic range (speech has variation)
            dynamic_range = np.max(audio) - np.min(audio)
            range_check = dynamic_range > 0.01
            
            # Check zero crossing rate (speech has moderate ZCR)
            zero_crossings = np.sum(np.abs(np.diff(np.sign(audio))))
            zcr = zero_crossings / len(audio) if len(audio) > 0 else 0
            zcr_check = zcr > 0.01  # Has some variation
            
            return energy_check and range_check and zcr_check
        
        return False
    
    def _is_likely_hallucination(self, text):
        """Check if text is likely a hallucination"""
        text_lower = text.lower().strip()
        
        # Too short or empty
        if len(text_lower) <= 2:
            return True
            
        # Check against common hallucinations
        if text_lower in self.common_hallucinations:
            return True
            
        # Check for common hallucination phrases
        for hallucination in self.common_hallucinations:
            if len(hallucination) > 3 and hallucination in text_lower:
                return True
        
        return False
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback function for audio stream"""
        audio_data = np.frombuffer(in_data, dtype=np.int16)
        self.audio_queue.put(audio_data)
        return (in_data, pyaudio.paContinue)
    
    def _audio_processor(self):
        """Process audio chunks in separate thread"""
        audio_buffer = np.array([], dtype=np.int16)
        
        while self.is_recording:
            try:
                # Get audio data from queue
                if not self.audio_queue.empty():
                    new_audio = self.audio_queue.get(timeout=0.1)
                    audio_buffer = np.concatenate([audio_buffer, new_audio])
                
                # Process when we have enough audio
                if len(audio_buffer) >= self.chunk_size:
                    # Extract chunk with overlap
                    chunk = audio_buffer[:self.chunk_size]
                    audio_buffer = audio_buffer[self.chunk_size - self.overlap_size:]
                    
                    # Preprocess audio
                    processed_chunk = self._preprocess_audio(chunk)
                    
                    # Simple voice activity detection
                    if self._simple_voice_detection(processed_chunk):
                        try:
                            # Transcribe with Whisper
                            result = self.model.transcribe(
                                processed_chunk.astype(np.float32),
                                language="en",
                                fp16=False,
                                temperature=0.0,
                                condition_on_previous_text=False,
                                no_speech_threshold=0.5,
                                logprob_threshold=-1.0,
                                compression_ratio_threshold=2.4
                            )
                            
                            text = result['text'].strip()
                            
                            # Filter out hallucinations and short/empty text
                            if (text and len(text) > 3 and 
                                not self._is_likely_hallucination(text)):
                                self.text_queue.put(text)
                                
                        except Exception as e:
                            print(f"Transcription error: {e}")
                            continue
                else:
                    time.sleep(0.01)
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Audio processing error: {e}")
                continue
    
    def calibrate_background_noise(self, duration=2.0):
        """Calibrate background noise for better detection"""
        print(f"Calibrating background noise for {duration} seconds...")
        print("Please stay quiet during calibration...")
        
        # Record background noise
        frames = []
        stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=1024
        )
        
        for _ in range(int(self.sample_rate * duration / 1024)):
            data = stream.read(1024)
            frames.append(np.frombuffer(data, dtype=np.int16))
        
        stream.stop_stream()
        stream.close()
        
        # Process background noise
        background_audio = np.concatenate(frames).astype(np.float32) / 32768.0
        self.background_noise = background_audio
        
        # Calculate background energy
        self.background_energy = float(np.sqrt(np.mean(background_audio**2)))
        
        # Set adaptive threshold
        self.energy_threshold = max(0.001, self.background_energy * 3.0)
        
        print(f"Background noise calibrated.")
        print(f"Background energy: {self.background_energy:.6f}")
        print(f"Detection threshold: {self.energy_threshold:.6f}")
    
    def start_recording(self, device_index=None):
        """Start real-time recording and transcription"""
        if device_index is None:
            device_index = self._get_audio_devices()
        
        print(f"\nStarting recording from device {device_index}...")
        print("Speak clearly into the microphone...")
        
        # Start audio stream
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=1024,
            stream_callback=self._audio_callback
        )
        
        self.is_recording = True
        self.stream.start_stream()
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._audio_processor)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        print("Recording started! Press Ctrl+C to stop.\n")
        
        # Main loop to display transcriptions
        try:
            while self.is_recording:
                try:
                    # Retrieve transcribed task from the text queue
                    task = self.text_queue.get(timeout=0.1)
                    # Process tasks with more than one word
                    if len(task.split(" "))>1:
                        print(f"[{time.strftime('%H:%M:%S')}] Task: {task}")
                        # Tokenize the task for Router model input
                        task_tokens = tokenizer(task, return_tensors="pt", padding=True, truncation=True).to(device)
                        input_ids = task_tokens['input_ids'].to(device)
                        attention_mask = task_tokens['attention_mask'].to(device)
                        # Classify task using Router model
                        with torch.no_grad():
                            with autocast(device_type=device.type):# Use mixed precision for efficiency
                                logits = router(input_ids=input_ids, attention_mask=attention_mask)
                            predicted_class = torch.argmax(logits, dim=1).item() # Get predicted class (0: VLM, 1: VLA)
                        print(f"Predicted class: {predicted_class} ({'VLM' if predicted_class==0 else 'VLA'})")

                        # Capture simulation image for model input
                        view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.1, 0, 0.65], distance=0.9, yaw=60, pitch=-35, roll=0, upAxisIndex=2)
                        projection_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=1.0, nearVal=0.01, farVal=2.0)
                        img_data = p.getCameraImage(224, 224, view_matrix, projection_matrix)
                        image = Image.fromarray(np.reshape(img_data[2], (224, 224, 4))[:, :, :3])
                        
                        
                        if predicted_class==0:
                            print("-> Routing to VLM model")
                            vlm_output = vlm_head(task,image) # Process task with VLM for text response
                            print(f"VLM Output: {vlm_output}\n")
                        else:
                            print("-> Routing to VLA model")
                            smoothed_pos_delta = np.zeros(3) # Initialize smoothed position delta for robot movement
                            for step in range(5): 
                                vla_output = vla_head(task, image) # Generate actions from VLA model
                                for action_delta in vla_output:
                                    # Smooth the position delta to avoid jerky movements.
                                    raw_pos_delta = action_delta[:3]
                                    smoothed_pos_delta = 0.7 * raw_pos_delta + (1 - 0.7) * smoothed_pos_delta

                                    # Clip the movement magnitude to a safe maximum.
                                    max_step_magnitude = 0.25
                                    delta_magnitude = np.linalg.norm(smoothed_pos_delta)
                                    clipped_pos_delta = smoothed_pos_delta / delta_magnitude * max_step_magnitude if delta_magnitude > max_step_magnitude else smoothed_pos_delta

                                    # Get the current position and orientation of the gripper.
                                    link_state = p.getLinkState(robot_id, 11, computeForwardKinematics=True)
                                    current_pos, current_orn_quat = np.array(link_state[0]), np.array(link_state[1])
                                    
                                    # Calculate the target position.
                                    target_pos = current_pos + clipped_pos_delta

                                    # Determine gripper command (open or close).
                                    gripper_cmd = action_delta[6]
                                    is_opening = gripper_cmd <= 0.5
                                    gripper_target = 0.04 if is_opening else 0.00

                                    # Execute the movement over a short duration for smoothness.
                                    for _ in range(30):
                                        try:
                                            joint_poses = p.calculateInverseKinematics(robot_id, 11, target_pos, current_orn_quat)
                                            for i in range(7): p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, joint_poses[i], force=120)
                                            p.setJointMotorControl2(robot_id, 9, p.POSITION_CONTROL, gripper_target, force=50)
                                            p.setJointMotorControl2(robot_id, 10, p.POSITION_CONTROL, gripper_target, force=50)
                                            p.stepSimulation()
                                            time.sleep(1/240.)
                                        except p.error:
                                            break  # Skip this movement if IK fails.
                                view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.1, 0, 0.65], distance=0.9, yaw=60, pitch=-35, roll=0, upAxisIndex=2)
                                projection_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=1.0, nearVal=0.01, farVal=2.0)
                                img_data = p.getCameraImage(224, 224, view_matrix, projection_matrix)
                                image = Image.fromarray(np.reshape(img_data[2], (224, 224, 4))[:, :, :3])

                except queue.Empty:
                    continue
        except KeyboardInterrupt:
            print("\nStopping recording...")
            self.stop_recording()
    
    def stop_recording(self):
        """Stop recording and clean up"""
        self.is_recording = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)
        
        print("Recording stopped.")
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        self.stop_recording()
        if hasattr(self, 'p'):
            self.p.terminate()

def AudioProcessing():
    """Main function to run the real-time STT system"""
    print("Real-time Whisper Speech-to-Text System")
    print("=" * 50)
    
    # Initialize the STT system
    stt = RealTimeWhisperSTT(
        model_size="medium.en",
        device="cuda",
        sample_rate=16000,
        chunk_duration=5.0,  # 3-second chunks
        overlap_duration=0.5,  # 0.5-second overlap
        energy_threshold=0.001,  # Will be auto-adjusted
        noise_reduction=True
    )
    
    try:
        # Calibrate background noise
        stt.calibrate_background_noise(duration=3.0)
        
        # Start recording
        stt.start_recording()
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        stt.stop_recording()

def commandProcessing():
    while True:
        task=input("Enter your command (or 'exit' to quit): ")
        if task.lower() in ['exit', 'quit']:
            print("Exiting command processing.")
            break
        if len(task)>1:
            print(f"Task: {task}")

            # Tokenize the task for Router model input
            task_tokens = tokenizer(task, return_tensors="pt", padding=True, truncation=True).to(device)
            input_ids = task_tokens['input_ids'].to(device)
            attention_mask = task_tokens['attention_mask'].to(device)

            # Classify task using Router model
            with torch.no_grad():
                with autocast(device_type=device.type):
                    logits = router(input_ids=input_ids, attention_mask=attention_mask)
                predicted_class = torch.argmax(logits, dim=1).item()
            print(f"Predicted class: {predicted_class} ({'VLM' if predicted_class==0 else 'VLA'})")

            # Capture simulation image for model input
            view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.1, 0, 0.65], distance=0.9, yaw=60, pitch=-35, roll=0, upAxisIndex=2)
            projection_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=1.0, nearVal=0.01, farVal=2.0)
            img_data = p.getCameraImage(224, 224, view_matrix, projection_matrix)
            image = Image.fromarray(np.reshape(img_data[2], (224, 224, 4))[:, :, :3])
            if predicted_class==0:
                print("-> Routing to VLM model")
                vlm_output = vlm_head(task,image) # Process task with VLM for text response
                print(f"VLM Output: {vlm_output}\n")
            else:
                print("-> Routing to VLA model")
                smoothed_pos_delta = np.zeros(3)
                for step in range(5):
                    vla_output = vla_head(task, image)  # Generate actions from VLA model
                    for action_delta in vla_output:
                        # Smooth the position delta to avoid jerky movements.
                        raw_pos_delta = action_delta[:3]
                        smoothed_pos_delta = 0.7 * raw_pos_delta + (1 - 0.7) * smoothed_pos_delta

                        # Clip the movement magnitude to a safe maximum.
                        max_step_magnitude = 0.25
                        delta_magnitude = np.linalg.norm(smoothed_pos_delta)
                        clipped_pos_delta = smoothed_pos_delta / delta_magnitude * max_step_magnitude if delta_magnitude > max_step_magnitude else smoothed_pos_delta

                        # Get the current position and orientation of the gripper.
                        link_state = p.getLinkState(robot_id, 11, computeForwardKinematics=True)
                        current_pos, current_orn_quat = np.array(link_state[0]), np.array(link_state[1])
                        
                        # Calculate the target position.
                        target_pos = current_pos + clipped_pos_delta

                        # Determine gripper command (open or close).
                        gripper_cmd = action_delta[6]
                        is_opening = gripper_cmd <= 0.5
                        gripper_target = 0.04 if is_opening else 0.00

                        # Execute the movement over a short duration for smoothness.
                        for _ in range(30):
                            try:
                                joint_poses = p.calculateInverseKinematics(robot_id, 11, target_pos, current_orn_quat)
                                for i in range(7): p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, joint_poses[i], force=120)
                                p.setJointMotorControl2(robot_id, 9, p.POSITION_CONTROL, gripper_target, force=50)
                                p.setJointMotorControl2(robot_id, 10, p.POSITION_CONTROL, gripper_target, force=50)
                                p.stepSimulation()
                                time.sleep(1/240.)
                            except p.error:
                                break  # Skip this movement if IK fails.
                    view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.1, 0, 0.65], distance=0.9, yaw=60, pitch=-35, roll=0, upAxisIndex=2)
                    projection_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=1.0, nearVal=0.01, farVal=2.0)
                    img_data = p.getCameraImage(224, 224, view_matrix, projection_matrix)
                    image = Image.fromarray(np.reshape(img_data[2], (224, 224, 4))[:, :, :3])
        else:
            print("Please enter a valid command with more than one word.\n")






while True:
    method=input("Choose method (1: Command, 2: Audio): ")
    if method=="1":
        commandProcessing()
    elif method=="2":
        AudioProcessing()
    else:
        print("Invalid method selected.")
        method=input("Choose method (1: Command, 2: Audio): ")