import numpy as np
import torch

from torch.utils.data import Dataset

import torch.nn.functional as F


class ReacherMotionFrameDataset(Dataset):

    def __init__(
        self,
        npz_path,
        episode_indices
    ):

        data = np.load(npz_path)

        frames = data["frames"]
        episode_lengths = data["episode_lengths"]
        frame_offsets = data["frame_offsets"]

        self.frames_t = []
        self.frames_next = []

        for ep in episode_indices:

            start = frame_offsets[ep]
            length = episode_lengths[ep]

            ep_frames = frames[
                start:
                start + length + 1
            ]

            for t in range(length):

                self.frames_t.append(
                    ep_frames[t]
                )

                self.frames_next.append(
                    ep_frames[t + 1]
                )


        self.frames_t = np.stack(
            self.frames_t
        )

        self.frames_next = np.stack(
            self.frames_next
        )

    def __len__(self):

        return len(self.frames_t)

    def preprocess(self, frame):

        # uint8 [0,255] -> float32 [0,1]
        frame = frame.astype(np.float32) / 255.0

        # HWC -> CHW
        frame = frame.transpose(2, 0, 1)

        # Crop the robot workspace and resize to the model input size.
        # This matches the 256x256 renders and the configured 64x64 autoencoder.
        frame = torch.from_numpy(frame)
        frame = frame[
            :,
            64:208,
            48:208
        ]
        frame = frame.unsqueeze(0)
        frame = F.interpolate(
            frame,
            size=(128, 128),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        return frame
    


    def __getitem__(self, idx):

        frame_t = self.preprocess(
            self.frames_t[idx]
        )

        frame_next = self.preprocess(
            self.frames_next[idx]
        )

        return (
            frame_t,
            frame_next
        )



