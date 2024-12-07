import numpy as np
import cv2
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


