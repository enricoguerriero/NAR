import os
import cv2
import numpy as np
import torch

def tensor_to_images(tensor, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for i, frame in enumerate(tensor):
        frame = frame.numpy() if isinstance(frame, torch.Tensor) else frame
        frame = np.clip(frame * 255, 0, 255).astype(np.uint8)

        if frame.shape[-1] == 3:  # RGB
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        elif len(frame.shape) == 2:  # Already grayscale
            frame_bgr = frame
        elif frame.shape[-1] == 1:  # Grayscale with channel dim
            frame_bgr = frame.squeeze(-1)
        else:
            raise ValueError(f"Unsupported frame shape: {frame.shape}")

        filename = os.path.join(output_folder, f"frame_{i:04d}.png")
        cv2.imwrite(filename, frame_bgr)

    print(f"Saved {len(tensor)} frames to {output_folder}")


if __name__ == "__main__":
    tensor = torch.load("data/tokens/VideoLLaVA/0-shot/2sec_4fps/video_0_clip_0_0_0_0_0.pt")["pixel_values_videos"].squeeze(0)
    output_folder = "output_frames"
    tensor_to_images(tensor, output_folder)
