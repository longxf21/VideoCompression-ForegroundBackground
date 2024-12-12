import os
import sys

import numpy as np
import cv2
from PyQt5.QtWidgets import QApplication
from tqdm import tqdm
import argparse




def unpad_frame(frame, height, width):
    return frame[:height, :width]


def decode_DCT(quantized_frame):
    height, width, channels = quantized_frame.shape
    reconstructed_frame = np.zeros_like(quantized_frame, dtype=np.float32)

    for i in range(0, height, 8):
        for j in range(0, width, 8):
            for k in range(channels):
                quantized_block = quantized_frame[i:i+8, j:j+8, k]
                reconstructed_block = dequantize_block(quantized_block)
                reconstructed_frame[i:i+8, j:j+8, k] = reconstructed_block

    return reconstructed_frame

def dequantize_block(quantized_block):
    # Ensure the block is in float32 format for cv2.idct
    quantized_block = quantized_block.astype(np.float32)
    
    # Perform inverse DCT using OpenCV
    reconstructed_block = cv2.idct(quantized_block)
    
    reconstructed_block = np.clip(reconstructed_block + 128.0, 0, 255)

    return reconstructed_block



if __name__ == "__main__":
    from GUI import VideoPlayer
    parser = argparse.ArgumentParser(description='Video Segmentation Encoder')
    parser.add_argument('output_path', type=str, help='Output .cmp video file dir')
    parser.add_argument('n1', type=int, help='Quantization step for foreground macroblocks')
    parser.add_argument('n2', type=int, help='Quantization step for background macroblocks')
    parser.add_argument('--input_file', type=str, default=None, help='Input .rgb video file')
    parser.add_argument('--frame_width', type=int, default=960, help='Width of video frames')
    parser.add_argument('--frame_height', type=int, default=540, help='Height of video frames')
    parser.add_argument('--block_size', type=int, default=16, help='Size of macroblocks')
    parser.add_argument('--search_range', type=int, default=7, help='Search range for motion vectors')
    parser.add_argument('--threshold', type=int, default=1, help='Motion vector magnitude threshold')
    parser.add_argument('--framerate', type=int, default=30, help='Framerate of the video')
    parser.add_argument('--audio_file', type=str, default="", help='Input .wav audio file')

    args = parser.parse_args()

    output_file_name = args.output_path
    video_basename = os.path.basename(output_file_name)
    input_file_basename = os.path.splitext(video_basename)[0] + '.rgb'
    input_file_name = os.path.abspath(os.path.join("rgbs", input_file_basename))

    args.input_file = input_file_name
    args.output_path = os.path.abspath(os.path.dirname(args.output_path))


    app = QApplication(sys.argv)

    player = VideoPlayer(args)
    player.show()
    sys.exit(app.exec_())
