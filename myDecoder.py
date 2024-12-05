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


def main():
    parser = argparse.ArgumentParser(description='Video Segmentation Decoder')
    parser.add_argument('input_video', type=str, help='Input .cmp video file')
    # parser.add_argument('input_audio', type=str, help='Input .wav audio file')
    args = parser.parse_args()

    input_video = args.input_video
    # input_audio = args.input_audio
    
    height = 540
    width = 960
    fps = 30
    
    with open(input_video, 'rb') as f:
        encoded_frames = np.load(f, allow_pickle=True)
        
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' codec for mp4 files
    # video path should be the same as the input
    video = input_video.split('.')[0] + '.mp4'
    out = cv2.VideoWriter(video, fourcc, fps, (width, height))
    # decoded_frames = []
    
    for idx in tqdm(range(len(encoded_frames)), desc='Decoding frames'):
        frame = encoded_frames[idx]
        
        decoded_frame = decode_DCT(frame)
    
        unpadded_frame = unpad_frame(decoded_frame, height, width)

        unpadded_frame = np.clip(unpadded_frame, 0, 255).astype(np.uint8)
        unpadded_frame = cv2.cvtColor(unpadded_frame, cv2.COLOR_RGB2BGR)
        out.write(unpadded_frame)
        # decoded_frames.append(unpadded_frame)
    
    # decoded_frames = np.array(decoded_frames) # a 4D numpy array of shape (frames, height, width, 3)
        
    out.release()
    
if __name__=="__main__":
    main()