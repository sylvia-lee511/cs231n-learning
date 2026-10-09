import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset


# ============================================================
# Shared image preprocessing
# ============================================================

def preprocess_reacher_frame(
    frame
):

    # uint8 [0,255] -> float32 [0,1]
    frame = (
        frame.astype(
            np.float32
        )
        /
        255.0
    )


    # HWC -> CHW
    frame = frame.transpose(
        2,
        0,
        1
    )


    # 256x256 render:
    # crop workspace and resize to 128x128
    frame = torch.from_numpy(
        frame
    )


    frame = frame[
        :,
        64:208,
        48:208
    ]


    frame = frame.unsqueeze(
        0
    )


    frame = F.interpolate(
        frame,
        size=(128, 128),
        mode="bilinear",
        align_corners=False
    ).squeeze(
        0
    )


    return frame


# ============================================================
# Original frame-pair dataset
#
# Kept for compatibility with earlier scripts.
# ============================================================

class ReacherMotionFrameDataset(
    Dataset
):

    def __init__(
        self,
        npz_path,
        episode_indices
    ):

        data = np.load(
            npz_path
        )

        frames = data[
            "frames"
        ]

        episode_lengths = data[
            "episode_lengths"
        ]

        frame_offsets = data[
            "frame_offsets"
        ]


        self.frames_t = []
        self.frames_next = []


        for ep in episode_indices:

            start = int(
                frame_offsets[ep]
            )

            length = int(
                episode_lengths[ep]
            )


            ep_frames = frames[
                start:
                start + length + 1
            ]


            for t in range(
                length
            ):

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

        return len(
            self.frames_t
        )


    def preprocess(
        self,
        frame
    ):

        return preprocess_reacher_frame(
            frame
        )


    def __getitem__(
        self,
        idx
    ):

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


# ============================================================
# Frame-pair + simulator target dataset
#
# Every sample:
#
#   frame_t
#   frame_(t+1)
#   target_xy_world_t
#   target_xy_world_(t+1)
#
# Target coordinates are raw Reacher world coordinates.
# They are auxiliary labels only; they are NOT model inputs.
# ============================================================

class ReacherMotionTargetDataset(
    Dataset
):

    def __init__(
        self,
        npz_path,
        episode_indices
    ):

        data = np.load(
            npz_path
        )


        if (
            "target_xy_world"
            not in data.files
        ):

            raise KeyError(
                "target_xy_world is missing from the NPZ file. "
                "Re-run the modified collect_data.py first."
            )


        frames = data[
            "frames"
        ]

        target_xy_world = data[
            "target_xy_world"
        ]

        episode_lengths = data[
            "episode_lengths"
        ]

        frame_offsets = data[
            "frame_offsets"
        ]


        if (
            len(target_xy_world)
            !=
            len(frames)
        ):

            raise ValueError(
                "target_xy_world must be frame-aligned: "
                "len(target_xy_world) == len(frames)"
            )


        self.frames_t = []
        self.frames_next = []

        self.targets_t = []
        self.targets_next = []


        for ep in episode_indices:

            start = int(
                frame_offsets[ep]
            )

            length = int(
                episode_lengths[ep]
            )


            ep_frames = frames[
                start:
                start + length + 1
            ]


            ep_targets = (
                target_xy_world[
                    start:
                    start + length + 1
                ]
            )


            for t in range(
                length
            ):

                self.frames_t.append(
                    ep_frames[t]
                )

                self.frames_next.append(
                    ep_frames[t + 1]
                )

                self.targets_t.append(
                    ep_targets[t]
                )

                self.targets_next.append(
                    ep_targets[t + 1]
                )


        self.frames_t = np.stack(
            self.frames_t
        )

        self.frames_next = np.stack(
            self.frames_next
        )


        self.targets_t = np.stack(
            self.targets_t
        ).astype(
            np.float32
        )

        self.targets_next = np.stack(
            self.targets_next
        ).astype(
            np.float32
        )


    def __len__(self):

        return len(
            self.frames_t
        )


    def preprocess(
        self,
        frame
    ):

        return preprocess_reacher_frame(
            frame
        )


    def __getitem__(
        self,
        idx
    ):

        frame_t = self.preprocess(
            self.frames_t[idx]
        )

        frame_next = self.preprocess(
            self.frames_next[idx]
        )


        target_t = torch.from_numpy(
            self.targets_t[idx]
        ).float()


        target_next = torch.from_numpy(
            self.targets_next[idx]
        ).float()


        return (
            frame_t,
            frame_next,
            target_t,
            target_next
        )


# ============================================================
# Transition Dataset
#
# Each sample:
#
#   frame_t
#   action_t
#   frame_(t+1)
#
# Strictly constructed inside each episode.
# ============================================================

class ReacherTransitionDataset(Dataset):

    def __init__(
        self,
        npz_path,
        episode_indices
    ):

        data = np.load(
            npz_path
        )

        frames = data[
            "frames"
        ]

        actions = data[
            "actions"
        ]

        episode_lengths = data[
            "episode_lengths"
        ]

        frame_offsets = data[
            "frame_offsets"
        ]

        action_offsets = data[
            "action_offsets"
        ]


        self.frames_t = []
        self.actions_t = []
        self.frames_next = []


        for ep in episode_indices:

            frame_start = int(
                frame_offsets[ep]
            )

            action_start = int(
                action_offsets[ep]
            )

            length = int(
                episode_lengths[ep]
            )


            # ------------------------------------
            # Episode frames:
            #
            # o_0 ... o_L
            #
            # Episode actions:
            #
            # a_0 ... a_(L-1)
            # ------------------------------------

            ep_frames = frames[
                frame_start:
                frame_start + length + 1
            ]

            ep_actions = actions[
                action_start:
                action_start + length
            ]


            assert (
                len(ep_frames)
                ==
                len(ep_actions) + 1
            )


            for t in range(
                length
            ):

                self.frames_t.append(
                    ep_frames[t]
                )

                self.actions_t.append(
                    ep_actions[t]
                )

                self.frames_next.append(
                    ep_frames[t + 1]
                )


        self.frames_t = np.stack(
            self.frames_t,
            axis=0
        )

        self.actions_t = np.stack(
            self.actions_t,
            axis=0
        ).astype(
            np.float32
        )

        self.frames_next = np.stack(
            self.frames_next,
            axis=0
        )


        print(
            f"Loaded "
            f"{len(self.actions_t)} transitions "
            f"from "
            f"{len(list(episode_indices))} episodes"
        )


    def __len__(self):

        return len(
            self.actions_t
        )


    def __getitem__(
        self,
        idx
    ):

        frame_t = (
            preprocess_reacher_frame(
                self.frames_t[idx]
            )
        )

        frame_next = (
            preprocess_reacher_frame(
                self.frames_next[idx]
            )
        )

        action_t = torch.from_numpy(
            self.actions_t[idx]
        ).float()


        return (
            frame_t,
            action_t,
            frame_next
        )