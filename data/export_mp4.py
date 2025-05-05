import cv2
import numpy as np
import torch

def tensor_to_video(tensor, output_path, fps=4):
    height, width = tensor.shape[1], tensor.shape[2]
    is_color = tensor.shape[3] == 3

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can also try 'XVID'
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=is_color)

    for frame in tensor:
        frame_bgr = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2BGR) if is_color else (frame * 255).astype(np.uint8)
        out.write(frame_bgr)

    out.release()
    print(f"Saved video to {output_path}")


if __name__ == "__main__":
    # Example usage
    tensor = torch.load("data/tokens/VideoLLaVA/0-shot/2sec_4fps/video_0_clip_0_0_0_0_0.pt")["pixel_values_videos"].squeeze(0)
    output_path = "output_video.mp4"
    tensor_to_video(tensor, output_path)