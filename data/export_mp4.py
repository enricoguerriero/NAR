import os
import cv2
import numpy as np
import torch

def tensor_to_images(tensor, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for i, frame in enumerate(tensor):
        frame = frame.numpy() if isinstance(frame, torch.Tensor) else frame

        # Convert from (C, H, W) to (H, W, C)
        if frame.shape[0] in [1, 3]:
            frame = np.transpose(frame, (1, 2, 0))
        else:
            raise ValueError(f"Unexpected channel size in frame shape: {frame.shape}")

        # Normalize and convert to uint8
        frame = np.clip(frame * 255, 0, 255).astype(np.uint8)

        # Convert RGB to BGR for OpenCV
        if frame.shape[2] == 3:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        elif frame.shape[2] == 1:
            frame_bgr = frame.squeeze(2)
        else:
            raise ValueError(f"Unsupported channel count: {frame.shape[2]}")

        filename = os.path.join(output_folder, f"frame_{i:04d}.png")
        cv2.imwrite(filename, frame_bgr)

    print(f"Saved {len(tensor)} frames to {output_folder}")


if __name__ == "__main__":
    tensor = torch.load("data/tokens/VideoLLaVA/0-shot/2sec_4fps/video_1_clip_2332_1_1_0_0.pt")["pixel_values_videos"].squeeze(0)
    output_folder = "output_frames"
    tensor_to_images(tensor, output_folder)
