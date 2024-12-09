import os
import sys
import argparse
from time import time

import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QLabel, QProgressBar, \
    QMainWindow, QMessageBox
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt
from pydub import AudioSegment
import pygame
from tqdm import tqdm

from myDecoder import decode_DCT, unpad_frame
from myEncoder import read_rgb_video, pad_frame, compute_motion_vectors, classify_macroblocks, visualize_segmentation, \
    encode_DCT
import cv2
import numpy as np
from tqdm import tqdm


class VideoPlayer(QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.setWindowTitle("Video Player")
        self.framerate = args.framerate
        self.timer_interval = 1000 // self.framerate
        self.timer = QTimer()
        self.timer.timeout.connect(self.play_next_frame)
        self.videovalues = None
        self.current_frame = 0
        self.rgb_file = os.path.abspath(args.input_file)
        self.wav_file = os.path.abspath(args.wav_file)
        self.width = 960
        self.height = 540
        self.args = args

        video_basename = os.path.basename(rgb_file)
        output_cmp_file_basename = os.path.splitext(video_basename)[0] + '.cmp'
        self.output_cmp_file_name = os.path.abspath(os.path.join(args.output_path, output_cmp_file_basename))

        output_mp4_file_basename = os.path.splitext(video_basename)[0] + '.mp4'
        self.output_mp4_file_name = os.path.abspath(os.path.join(args.output_path, output_mp4_file_basename))

        pygame.mixer.init()
        self.init_ui()

    def init_ui(self):
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setText("Player")
        self.video_label.setFixedSize(self.width, self.height)


        self.encoded_video_label = QLabel(self)
        self.encoded_video_label.setAlignment(Qt.AlignCenter)
        self.encoded_video_label.setText("Encoding Progress Viewer")
        self.encoded_video_label.setFixedSize(self.width, self.height)


        self.status = QLabel(self)
        self.status.setAlignment(Qt.AlignCenter)
        self.status.setText("Source Video/Compressed Video")

        self.parameters = QLabel(self)
        self.parameters.setAlignment(Qt.AlignCenter)
        self.parameters.setText("Parameters: n1 = " + str(self.args.n1) + ", n2 = " + str(self.args.n2) + ", block_size = " + str(self.args.block_size) + ", search_range = " + str(self.args.search_range) + ", threshold = " + str(self.args.threshold))

        self.videoname_label = QLabel(self)
        self.videoname_label.setAlignment(Qt.AlignCenter)
        self.videoname_label.setText("Video Source is " + self.rgb_file)

        self.audioname_label = QLabel(self)
        self.audioname_label.setAlignment(Qt.AlignCenter)
        self.audioname_label.setText("Audio Source is " + self.wav_file)

        self.output_cmp_path_label = QLabel(self)
        self.output_cmp_path_label.setAlignment(Qt.AlignCenter)
        self.output_cmp_path_label.setText("Cmp File Output path is " + self.output_cmp_file_name)

        self.output_mp4_path_label = QLabel(self)
        self.output_mp4_path_label.setAlignment(Qt.AlignCenter)
        self.output_mp4_path_label.setText("MP4 File Output path is " + self.output_mp4_file_name)

        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_playback)

        self.next_frame_button = QPushButton("Next Frame")
        self.next_frame_button.clicked.connect(self.next_frame)

        self.prev_frame_button = QPushButton("Previous Frame")
        self.prev_frame_button.clicked.connect(self.prev_frame)

        self.load_video_button = QPushButton("Load Original Video")
        self.load_video_button.clicked.connect(self.load_video)

        self.load_decoded_video_button = QPushButton("Load Decoded Video")
        self.load_decoded_video_button.clicked.connect(self.load_mp4_video)

        self.encode_button = QPushButton("Encode Video")
        self.encode_button.clicked.connect(self.encode_refresh_in_window)

        self.decode_button = QPushButton("Decode Video")
        self.decode_button.clicked.connect(self.decode)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)

        self.progress_label = QLabel(self)  # New label for progress and ETA
        self.progress_label.setText("")  # Initialize as empty



        video_layout = QHBoxLayout()
        video_layout.addWidget(self.video_label)
        video_layout.addWidget(self.encoded_video_label)



        # Vertical layout for main widgets
        main_layout = QVBoxLayout()
        main_layout.addLayout(video_layout)
        main_layout.addWidget(self.status)
        main_layout.addWidget(self.parameters)

        main_layout.addWidget(self.videoname_label)
        main_layout.addWidget(self.audioname_label)
        main_layout.addWidget(self.output_cmp_path_label)
        main_layout.addWidget(self.output_mp4_path_label)
        main_layout.addWidget(self.progress_bar)
        main_layout.addWidget(self.progress_label)

        # Horizontal layout for buttons
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.play_button)
        button_layout.addWidget(self.next_frame_button)
        button_layout.addWidget(self.prev_frame_button)
        button_layout.addWidget(self.load_video_button)
        button_layout.addWidget(self.encode_button)
        button_layout.addWidget(self.decode_button)
        button_layout.addWidget(self.load_decoded_video_button)

        # Add button layout to the main layout
        main_layout.addLayout(button_layout)

        # Set up the central widget
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self.load_video()

    def load_video(self):
        self.videovalues = read_rgb_video(self.rgb_file, self.width, self.height)
        self.current_frame = 0
        self.status.setText(f"Playing Source Video")
        self.show_frame(0)
        self.pause()
        self.play_button.setText("Play")
        self.timer.stop()
        pygame.mixer.music.pause()

    def load_mp4_video(self):
        self.status.setText(f"Playing Decoded Video")
        video_frames = []
        cap = cv2.VideoCapture(self.output_mp4_file_name)

        if not cap.isOpened():
            self.show_error_popup(f"Failed to load video: {self.output_mp4_file_name}. Please encode the video before decoding it! ")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert frame from BGR (OpenCV default) to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_frames.append(frame_rgb)

        cap.release()
        self.videovalues = video_frames
        self.current_frame = 0
        self.show_frame(0)
        self.pause()
        self.play_button.setText("Play")
        self.timer.stop()
        pygame.mixer.music.pause()

    def show_frame(self, frame_id):
        if frame_id < 0 or frame_id >= len(self.videovalues):
            return
        frame = self.videovalues[frame_id]
        height, width, _ = frame.shape
        qimage = QImage(frame.data, width, height, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        self.video_label.setPixmap(pixmap)

    def toggle_playback(self):
        if self.timer.isActive():
            self.pause()
            self.play_button.setText("Play")
        else:
            self.play()
            self.play_button.setText("Pause")

    def play(self):
        self.timer.start(self.timer_interval)
        self.play_audio()

    def pause(self):
        self.timer.stop()
        pygame.mixer.music.pause()

    def play_audio(self):
        if not self.wav_file:
            return

        playback_position = int((self.current_frame / self.framerate) * 1000)

        pygame.mixer.music.stop()
        pygame.mixer.music.load(self.wav_file)
        pygame.mixer.music.play(start=playback_position / 1000)

    def play_next_frame(self):
        if self.current_frame >= len(self.videovalues):
            self.pause()
            return
        self.current_frame += 1
        self.show_frame(self.current_frame)

    def next_frame(self):
        self.pause()
        self.current_frame = min(self.current_frame + 1, len(self.videovalues) - 1)
        self.show_frame(self.current_frame)

    def prev_frame(self):
        self.pause()
        self.current_frame = max(self.current_frame - 1, 0)
        self.show_frame(self.current_frame)

    def encode(self):
        frame_width = self.width
        frame_height = self.height
        frames = self.videovalues

        input_filename = self.args.input_file
        n1 = self.args.n1
        n2 = self.args.n2
        block_size = self.args.block_size
        search_range = self.args.search_range
        threshold = self.args.threshold

        zigzag = [(0, 0), (0, 1), (1, 0), (2, 0), (1, 1), (0, 2), (0, 3), (1, 2),
                  (2, 1), (3, 0), (4, 0), (3, 1), (2, 2), (1, 3), (0, 4), (0, 5),
                  (1, 4), (2, 3), (3, 2), (4, 1), (5, 0), (6, 0), (5, 1), (4, 2),
                  (3, 3), (2, 4), (1, 5), (0, 6), (0, 7), (1, 6), (2, 5), (3, 4),
                  (4, 3), (5, 2), (6, 1), (7, 0), (7, 1), (6, 2), (5, 3), (4, 4),
                  (3, 5), (2, 6), (1, 7), (2, 7), (3, 6), (4, 5), (5, 4), (6, 3),
                  (7, 2), (7, 3), (6, 4), (5, 5), (4, 6), (3, 7), (4, 7), (5, 6),
                  (6, 5), (7, 4), (7, 5), (6, 6), (5, 7), (6, 7), (7, 6), (7, 7)]

        zigzag1 = zigzag[:n1]
        zigzag2 = zigzag[:n2]

        print("Reading video frames...")
        # frames = read_rgb_video(input_filename, frame_width, frame_height)
        num_frames = len(frames)
        print(f"Total frames read: {num_frames}")

        # Pad frames to be multiples of block_size
        padded_frames = [pad_frame(frame, block_size) for frame in frames]

        mvectors_list = []
        classifications = []

        encoded_frames = []
        total_frames = num_frames - 1

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_label.setText("")
        self.status.setText("Start Encoding!")
        start_time = time()
        for idx in tqdm(range(1, num_frames), desc="Processing frames"):
            elapsed_time = time() - start_time
            progress = int((idx / total_frames) * 100)
            eta = elapsed_time / idx * (total_frames - idx)

            # Update progress bar and new progress label
            self.progress_bar.setValue(progress)
            self.progress_label.setText(
                f"Frame {idx} of {total_frames} processed. ETA: {int(eta)} seconds"
            )

            prev_frame = padded_frames[idx - 1]
            curr_frame = padded_frames[idx]

            # Compute motion vectors
            mvectors = compute_motion_vectors(prev_frame, curr_frame, block_size, search_range)
            mvectors_list.append(mvectors)

            # Classify macroblocks
            classification, global_motion_vector = classify_macroblocks(mvectors, threshold, curr_frame, block_size)
            classifications.append(classification)

            # dummy classification
            # classification = np.zeros((34, 60), dtype=np.uint8)

            encoded_frames.append(encode_DCT(curr_frame, classification, zigzag1, zigzag2))

            # Visualize segmentation for testing
            vis_frame = visualize_segmentation(curr_frame, classification, block_size, global_motion_vector)
            display_frame = cv2.resize(vis_frame, (960, 540))
            cv2.imshow('Segmentation', display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # if idx > 2:
            #     break
        cv2.destroyAllWindows()
        self.progress_bar.setValue(100)
        self.status.setText("Encoding complete!")
        self.progress_bar.setVisible(False)
        # store the compressed video file with the same name as the input video file but with a different extension
        # output_filename = input_filename.replace('.rgb', '.cmp')
        with open(self.output_cmp_file_name, 'wb') as f:
            np.save(f, np.array(encoded_frames))
        print(f"Compressed video saved as {self.output_cmp_file_name}")

        # Compression step
        # Save the motion vectors and classifications for further processing


    def encode_refresh_in_window(self):
        frame_width = self.width
        frame_height = self.height
        frames = self.videovalues

        input_filename = self.args.input_file
        n1 = self.args.n1
        n2 = self.args.n2
        block_size = self.args.block_size
        search_range = self.args.search_range
        threshold = self.args.threshold

        zigzag = [(0, 0), (0, 1), (1, 0), (2, 0), (1, 1), (0, 2), (0, 3), (1, 2),
                  (2, 1), (3, 0), (4, 0), (3, 1), (2, 2), (1, 3), (0, 4), (0, 5),
                  (1, 4), (2, 3), (3, 2), (4, 1), (5, 0), (6, 0), (5, 1), (4, 2),
                  (3, 3), (2, 4), (1, 5), (0, 6), (0, 7), (1, 6), (2, 5), (3, 4),
                  (4, 3), (5, 2), (6, 1), (7, 0), (7, 1), (6, 2), (5, 3), (4, 4),
                  (3, 5), (2, 6), (1, 7), (2, 7), (3, 6), (4, 5), (5, 4), (6, 3),
                  (7, 2), (7, 3), (6, 4), (5, 5), (4, 6), (3, 7), (4, 7), (5, 6),
                  (6, 5), (7, 4), (7, 5), (6, 6), (5, 7), (6, 7), (7, 6), (7, 7)]

        zigzag1 = zigzag[:n1]
        zigzag2 = zigzag[:n2]

        print("Reading video frames...")
        num_frames = len(frames)
        print(f"Total frames read: {num_frames}")

        # Pad frames to be multiples of block_size
        padded_frames = [pad_frame(frame, block_size) for frame in frames]

        mvectors_list = []
        classifications = []
        encoded_frames = []

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_label.setText("")
        self.status.setText("Start Encoding!")

        self.current_frame_idx = 1
        self.total_frames = num_frames - 1
        self.padded_frames = padded_frames
        self.encoded_frames = encoded_frames
        self.mvectors_list = mvectors_list
        self.classifications = classifications
        self.start_time = time()

        def process_frame():
            if self.current_frame_idx >= num_frames:
                self.progress_bar.setValue(100)
                self.status.setText("Encoding complete!")
                self.progress_bar.setVisible(False)
                return

            idx = self.current_frame_idx
            prev_frame = self.padded_frames[idx - 1]
            curr_frame = self.padded_frames[idx]

            # Compute motion vectors
            mvectors = compute_motion_vectors(prev_frame, curr_frame, block_size, search_range)
            self.mvectors_list.append(mvectors)

            # Classify macroblocks
            classification, global_motion_vector = classify_macroblocks(mvectors, threshold, curr_frame, block_size)
            self.classifications.append(classification)

            encoded_frame = encode_DCT(curr_frame, classification, zigzag1, zigzag2)
            self.encoded_frames.append(encoded_frame)

            # Visualize segmentation
            vis_frame = visualize_segmentation(curr_frame, classification, block_size, global_motion_vector)

            resized_vis_frame = cv2.resize(vis_frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

            vis_frame_rgb = cv2.cvtColor(resized_vis_frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
            height, width, channel = vis_frame_rgb.shape
            bytes_per_line = channel * width
            qimg = QImage(vis_frame_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            self.encoded_video_label.setPixmap(pixmap)

            elapsed_time = time() - self.start_time
            progress = int((idx / self.total_frames) * 100)
            eta = elapsed_time / idx * (self.total_frames - idx)
            self.progress_bar.setValue(progress)
            self.progress_label.setText(f"Frame {idx} of {self.total_frames} processed. ETA: {int(eta)} seconds")

            self.current_frame_idx += 1

        # Use QTimer to call process_frame incrementally
        self.timer2 = QTimer(self)
        self.timer2.timeout.connect(process_frame)
        self.timer2.start(10)  # Adjust interval as needed (in milliseconds)
        with open(self.output_cmp_file_name, 'wb') as f:
            np.save(f, np.array(encoded_frames))
        print(f"Compressed video saved as {self.output_cmp_file_name}")


    def show_error_popup(self, message):
        # Create a message box to display the error
        app = QApplication.instance()  # Check if an app instance exists
        if app is None:
            app = QApplication(sys.argv)  # Create a new app instance if none exists

        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Critical)
        msg_box.setWindowTitle("Error")
        msg_box.setText(message)
        msg_box.exec_()

    def decode(self):
        # parser = argparse.ArgumentParser(description='Video Segmentation Decoder')
        # parser.add_argument('input_video', type=str, help='Input .cmp video file')
        # parser.add_argument('input_audio', type=str, help='Input .wav audio file')
        # args = parser.parse_args()

        # input_video = args.input_video
        # input_audio = args.input_audio

        input_video = self.output_cmp_file_name
        height = self.height
        width = self.width

        try:
            # Attempt to load the video file
            with open(input_video, 'rb') as f:
                encoded_frames = np.load(f, allow_pickle=True)
        except Exception as e:
            # If there's an error, show a popup window with the error
            self.show_error_popup(f"Failed to load video: {e}. Please encode the video before decoding it! ")
            return

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' codec for mp4 files
        # video = input_video.split('.')[0] + '.mp4'
        out = cv2.VideoWriter(self.output_mp4_file_name, fourcc, self.framerate, (width, height))

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_label.setText("")
        self.status.setText("Start Decoding!")
        start_time = time()
        for idx in tqdm(range(1, len(encoded_frames)), desc='Decoding frames'):
            elapsed_time = time() - start_time
            progress = int((idx / len(encoded_frames)) * 100)
            eta = elapsed_time / idx * (len(encoded_frames) - idx)

            # Update progress bar and new progress label
            self.progress_bar.setValue(progress)
            self.progress_label.setText(
                f"Frame {idx} of {len(encoded_frames)} processed. ETA: {int(eta)} seconds"
            )

            frame = encoded_frames[idx]

            # Decode the frame
            decoded_frame = decode_DCT(frame)
            unpadded_frame = unpad_frame(decoded_frame, height, width)
            unpadded_frame = np.clip(unpadded_frame, 0, 255).astype(np.uint8)
            unpadded_frame = cv2.cvtColor(unpadded_frame, cv2.COLOR_RGB2BGR)

            # Write the frame to the output video
            out.write(unpadded_frame)

            # Display the frame in a window
            cv2.imshow('Decoded Video', unpadded_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the video writer and close the display window
        out.release()
        cv2.destroyAllWindows()
        self.progress_bar.setValue(100)
        self.status.setText("Decoding complete!")
        self.progress_bar.setVisible(False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Video Segmentation Encoder')
    parser.add_argument('input_file', type=str, help='Input .rgb video file')
    parser.add_argument('n1', type=int, help='Quantization step for foreground macroblocks')
    parser.add_argument('n2', type=int, help='Quantization step for background macroblocks')
    parser.add_argument('--frame_width', type=int, default=960, help='Width of video frames')
    parser.add_argument('--frame_height', type=int, default=540, help='Height of video frames')
    parser.add_argument('--block_size', type=int, default=16, help='Size of macroblocks')
    parser.add_argument('--search_range', type=int, default=7, help='Search range for motion vectors')
    parser.add_argument('--threshold', type=int, default=2, help='Motion vector magnitude threshold')
    parser.add_argument('--framerate', type=int, default=30, help='Framerate of the video')
    parser.add_argument('--audio_file', type=str, default="", help='Input .wav audio file')
    parser.add_argument('--output_path', type=str, default="output", help='Output .cmp video file dir')

    args = parser.parse_args()

    rgb_file = args.input_file
    if args.audio_file != "":
        wav_file = args.audio_file
    else:
        if rgb_file and rgb_file.endswith('.rgb'):
            video_dir = os.path.dirname(os.path.dirname(rgb_file))
            video_basename = os.path.basename(rgb_file)
            wav_basename = os.path.splitext(video_basename)[0] + '.wav'
            wav_dir = os.path.join(video_dir, 'wavs')
            wav_file = os.path.join(wav_dir, wav_basename)
        else:
            raise ValueError("No audio file provided and video file is invalid or missing.")
    print(f"Using audio file: {wav_file}")
    setattr(args, 'wav_file', wav_file)
    os.makedirs(args.output_path, exist_ok=True)

    app = QApplication(sys.argv)

    player = VideoPlayer(args)
    player.show()
    sys.exit(app.exec_())
