import os
import torch
import imageio
import numpy as np

def save_clips_as_visuals(
    data_dir: str,
    out_dir: str,
    tensor_key: str = 'pixel_values',
    format: str = 'gif',
    fps: int = 5
):
    """
    Saves each tensor clip in data_dir as a GIF or MP4 in out_dir, 
    embedding its label in the filename.

    Args:
        data_dir (str): Directory containing .pt files.
        out_dir (str): Directory to save visual clips.
        tensor_key (str): Key in the loaded dict where the clip tensor is stored.
        format (str): 'gif' or 'mp4'.
        fps (int): Frames per second for the output clip.
    """
    os.makedirs(out_dir, exist_ok=True)

    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith('.pt'):
            continue

        path = os.path.join(data_dir, fname)
        data = torch.load(path)
        
        # Check for tensor_key
        if tensor_key not in data:
            print(f"[Warning] '{tensor_key}' not found in {fname}. Keys: {list(data.keys())}")
            continue

        clip: torch.Tensor = data[tensor_key]  # Expect shape [T, C, H, W]
        clip = clip.squeeze(0)  # Remove batch dimension if present
        # Move to CPU and convert to numpy (T, H, W, C)
        frames = clip.cpu().permute(0, 2, 3, 1).numpy()

        # Normalize to uint8 if needed
        if frames.dtype != np.uint8:
            frames = (frames * 255).clip(0, 255).astype(np.uint8)

        # Build output filename with labels
        base_name = os.path.splitext(fname)[0]
        out_ext = format.lower()
        out_name = f"{base_name}.{out_ext}"
        out_path = os.path.join(out_dir, out_name)

        # Write out
        if out_ext == 'gif':
            with imageio.get_writer(out_path, mode='I', fps=fps) as writer:
                for frame in frames:
                    writer.append_data(frame)
        elif out_ext == 'mp4':
            # Requires imageio-ffmpeg
            with imageio.get_writer(out_path, mode='I', fps=fps, codec='libx264') as writer:
                for frame in frames:
                    writer.append_data(frame)
        else:
            raise ValueError("Unsupported format: choose 'gif' or 'mp4'")

    print(f"Saved visuals to {out_dir}")


# Example usage:

if __name__ == "__main__":
    # Example usage
    save_clips_as_visuals(
        data_dir='data/tokens/TimeSformer/validation/2sec_8fps',
        out_dir='data/visuals/TimeSformer/validation',
        tensor_key='pixel_values',
        format='mp4',
        fps=8
    )