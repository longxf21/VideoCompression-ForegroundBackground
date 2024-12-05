import numpy as np
import cv2
from tqdm import tqdm
import argparse

def read_rgb_video(filename, frame_width, frame_height):
    frame_size = frame_width * frame_height * 3  # 3 bytes per pixel (R, G, B)
    video_frames = []

    with open(filename, 'rb') as f:
        while True:
            raw_frame = f.read(frame_size)
            if not raw_frame:
                break
            frame = np.frombuffer(raw_frame, dtype=np.uint8)
            frame = frame.reshape((frame_height, frame_width, 3))
            video_frames.append(frame)
    return video_frames

def pad_frame(frame, block_size):
    height, width, channels = frame.shape
    pad_height = block_size - (height % block_size) if (height % block_size) != 0 else 0
    pad_width = block_size - (width % block_size) if (width % block_size) != 0 else 0
    padded_frame = cv2.copyMakeBorder(frame, 0, pad_height, 0, pad_width, cv2.BORDER_REPLICATE)
    return padded_frame

def compute_motion_vectors(prev_frame, curr_frame, block_size, search_range):
    # Convert grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2GRAY)

    # Apply Gaussian blur to reduce noise
    prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 0)
    curr_gray = cv2.GaussianBlur(curr_gray, (5, 5), 0)

    height, width = curr_gray.shape
    mvectors = np.zeros((height // block_size, width // block_size, 2), dtype=np.int32)

    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            curr_block = curr_gray[i:i+block_size, j:j+block_size]
            min_mad = float('inf')
            best_dx = 0
            best_dy = 0

            # Define search window in the previous frame
            i_min = max(i - search_range, 0)
            i_max = min(i + search_range, height - block_size)
            j_min = max(j - search_range, 0)
            j_max = min(j + search_range, width - block_size)

            for m in range(i_min, i_max + 1):
                for n in range(j_min, j_max + 1):
                    ref_block = prev_gray[m:m+block_size, n:n+block_size]
                    if ref_block.shape != curr_block.shape:
                        continue  # Skip if blocks are not the same size (edge case)
                    mad = np.mean(np.abs(curr_block.astype(np.int16) - ref_block.astype(np.int16)))
                    if mad < min_mad:
                        min_mad = mad
                        best_dx = n - j
                        best_dy = m - i
            mvectors[i // block_size, j // block_size] = [best_dx, best_dy]
    return mvectors

def classify_macroblocks(mvectors, threshold, curr_frame, block_size):
    # Reshape motion vectors
    vectors = mvectors.reshape(-1, 2)

    # Compute the global motion vector using median
    global_motion_vector = np.median(vectors, axis=0).astype(np.int32)

    compensated_vectors = vectors - global_motion_vector
    magnitudes = np.linalg.norm(compensated_vectors, axis=1)
    magnitudes = magnitudes.reshape(mvectors.shape[0], mvectors.shape[1])

    # Compute variance within  macroblocks
    height, width = mvectors.shape[:2]
    variance_map = np.zeros((height, width))
    gray_frame = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2GRAY)

    for i in range(height):
        for j in range(width):
            y = i * block_size
            x = j * block_size
            block = gray_frame[y:y+block_size, x:x+block_size]
            variance = np.var(block)
            variance_map[i, j] = variance

    variance_threshold = 50  # Adjust based on data (higher means ignore low-texture areas)
    low_texture = variance_map < variance_threshold

    # Classify macroblocks
    classification = np.zeros((height, width), dtype=np.uint8)
    classification[(magnitudes > threshold) & (~low_texture)] = 1  # Foreground

    return classification, global_motion_vector

def encode_DCT(padded_frame, classification, zigzag1, zigzag2):
    height, width, channels = padded_frame.shape
    quantized_frame = np.zeros_like(padded_frame, dtype=np.float32)

    for i in range(0, height, 8):
        for j in range(0, width, 8):
            segment = classification[i//16, j//16]
            for k in range(channels):
                macroblock = padded_frame[i:i+8, j:j+8, k]
                if segment == 1:
                    quantized_macroblock = quantize_block(macroblock, zigzag1)
                else:
                    quantized_macroblock = quantize_block(macroblock, zigzag2)
                quantized_frame[i:i+8, j:j+8, k] = quantized_macroblock

    return quantized_frame

def quantize_block(block, zigzag):
    block = block.astype(np.float32)
    block -= 128.0

    dct_block = cv2.dct(block)

    quantized_block = np.zeros_like(dct_block, dtype=np.float32)

    for idx in zigzag:
        i, j = idx
        quantized_block[i, j] = dct_block[i, j]

    return quantized_block

def visualize_segmentation(frame, classification, block_size, global_motion_vector=None):
    vis_frame = frame.copy()
    height, width = classification.shape
    for i in range(height):
        for j in range(width):
            y = i * block_size
            x = j * block_size
            if classification[i, j] == 1:
                # Red rectangle for foreground
                cv2.rectangle(vis_frame, (x, y), (x + block_size - 1, y + block_size - 1), (0, 0, 255), 2)
            else:
                # Green rectangle for background
                cv2.rectangle(vis_frame, (x, y), (x + block_size - 1, y + block_size - 1), (0, 255, 0), 1)

    return vis_frame

def main():
    parser = argparse.ArgumentParser(description='Video Segmentation Encoder')
    parser.add_argument('input_file', type=str, help='Input .rgb video file')
    parser.add_argument('n1', type=int, help='Quantization step for foreground macroblocks')
    parser.add_argument('n2', type=int, help='Quantization step for background macroblocks')
    args = parser.parse_args()

    input_filename = args.input_file
    n1 = args.n1
    n2 = args.n2

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

    frame_width = 960
    frame_height = 540
    block_size = 16
    search_range = 7
    threshold = 2  # Motion vector magnitude threshold

    print("Reading video frames...")
    frames = read_rgb_video(input_filename, frame_width, frame_height)
    num_frames = len(frames)
    print(f"Total frames read: {num_frames}")

    # Pad frames to be multiples of block_size
    padded_frames = [pad_frame(frame, block_size) for frame in frames]

    mvectors_list = []
    classifications = []

    encoded_frames = []

    for idx in tqdm(range(1, num_frames), desc="Processing frames"):
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

    cv2.destroyAllWindows()
    
    #store the compressed video file with the same name as the input video file but with a different extension
    output_filename = input_filename.replace('.rgb', '.cmp')
    with open(output_filename, 'wb') as f:
        np.save(f, np.array(encoded_frames))
    print(f"Compressed video saved as {output_filename}")

    # Compression step
    # Save the motion vectors and classifications for further processing

if __name__ == '__main__':
    main()
