import os
import cv2
import numpy as np
import torch

def tensor_to_images(tensor, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    height, width = tensor.shape[1], tensor.shape[2]
    is_color = tensor.shape[3] == 3

    for i, frame in enumerate(tensor):
        if is_color:
            frame_bgr = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        else:
            frame_bgr = (frame.squeeze() * 255).astype(np.uint8)
        
        filename = os.path.join(output_folder, f"frame_{i:04d}.png")
        cv2.imwrite(filename, frame_bgr)

    print(f"Saved {len(tensor)} frames to {output_folder}")


if __name__ == "__main__":
    # Example usage
    tensor = torch.load("data/tokens/VideoLLaVA/0-shot/2sec_4fps/video_0_clip_0_0_0_0_0.pt")["pixel_values_videos"].squeeze(0)
    output_folder = "output_frames"
    tensor_to_images(tensor.numpy(), output_folder)
