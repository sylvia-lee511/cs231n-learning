import gymnasium as gym
import numpy as np
from pathlib import Path


# ============================================================
# Config
# ============================================================

ENV_NAME = "Reacher-v5"
NUM_EPISODES = 200
IMAGE_SIZE = 256
SEED = 42

OUTPUT_DIR = Path(
    "./data_visual_world_model"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "reacher_visual_random.npz"
)


# ============================================================
# Environment
# ============================================================

env = gym.make(
    ENV_NAME,
    render_mode="rgb_array",
    width=IMAGE_SIZE,
    height=IMAGE_SIZE,
)


# ============================================================
# Storage
#
# frames and target_xy_world are frame-aligned:
#
#   frame_0, frame_1, ..., frame_T
#   goal_0,  goal_1,  ..., goal_T
#
# actions are transition-aligned:
#
#   action_0, ..., action_(T-1)
#
# so for every episode:
#
#   N_frames = N_actions + 1
#   N_targets = N_frames
# ============================================================

all_frames = []
all_actions = []
all_target_xy_world = []

episode_lengths = []

frame_offsets = []
action_offsets = []

total_transitions = 0


# ============================================================
# Collect
# ============================================================

for episode in range(NUM_EPISODES):

    obs, info = env.reset(
        seed=SEED + episode
    )

    env.action_space.seed(
        SEED + episode
    )


    # ------------------------------------------
    # Reacher-v5 observation:
    #
    # obs[4] = target x-coordinate
    # obs[5] = target y-coordinate
    #
    # These are simulator/world coordinates.
    # ------------------------------------------

    target_xy = np.asarray(
        obs[4:6],
        dtype=np.float32
    )


    # ------------------------------------------
    # Initial frame o_0
    # ------------------------------------------

    frame = env.render()

    episode_frames = [
        frame.copy()
    ]

    episode_target_xy = [
        target_xy.copy()
    ]

    episode_actions = []


    terminated = False
    truncated = False


    while not (
        terminated or truncated
    ):

        # ======================================
        # Current:
        #
        # episode_frames[-1]   = o_t
        # episode_target_xy[-1] = goal_t
        # ======================================

        action = (
            env.action_space.sample()
        )


        # s_t --a_t--> s_(t+1)

        (
            obs,
            reward,
            terminated,
            truncated,
            info
        ) = env.step(
            action
        )


        next_frame = env.render()


        # Target coordinates belonging to o_(t+1)
        next_target_xy = np.asarray(
            obs[4:6],
            dtype=np.float32
        )


        episode_actions.append(
            action.copy()
        )

        episode_frames.append(
            next_frame.copy()
        )

        episode_target_xy.append(
            next_target_xy.copy()
        )


    # ==========================================
    # Alignment checks
    # ==========================================

    assert (
        len(episode_frames)
        ==
        len(episode_actions) + 1
    )

    assert (
        len(episode_target_xy)
        ==
        len(episode_frames)
    )


    # Reacher target is fixed inside one episode.
    episode_target_array = np.stack(
        episode_target_xy,
        axis=0
    )

    max_goal_drift = np.abs(
        episode_target_array
        -
        episode_target_array[0]
    ).max()

    assert (
        max_goal_drift < 1e-6
    ), (
        "Unexpected target drift inside episode: "
        f"{max_goal_drift}"
    )


    # Starting locations inside global arrays

    frame_offsets.append(
        len(all_frames)
    )

    action_offsets.append(
        len(all_actions)
    )


    all_frames.extend(
        episode_frames
    )

    all_target_xy_world.extend(
        episode_target_xy
    )

    all_actions.extend(
        episode_actions
    )

    episode_lengths.append(
        len(episode_actions)
    )


    total_transitions += (
        len(episode_actions)
    )


    print(
        f"Episode {episode + 1:03d}/"
        f"{NUM_EPISODES} | "
        f"steps={len(episode_actions):3d} | "
        f"target=({episode_target_xy[0][0]:+.4f}, "
        f"{episode_target_xy[0][1]:+.4f}) | "
        f"total transitions="
        f"{total_transitions}"
    )


env.close()


# ============================================================
# Convert
# ============================================================

frames = np.stack(
    all_frames,
    axis=0
).astype(
    np.uint8
)


actions = np.stack(
    all_actions,
    axis=0
).astype(
    np.float32
)


target_xy_world = np.stack(
    all_target_xy_world,
    axis=0
).astype(
    np.float32
)


episode_lengths = np.asarray(
    episode_lengths,
    dtype=np.int32
)


frame_offsets = np.asarray(
    frame_offsets,
    dtype=np.int64
)


action_offsets = np.asarray(
    action_offsets,
    dtype=np.int64
)


# ============================================================
# Diagnostics
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "Frames:",
    frames.shape,
    frames.dtype
)

print(
    "Actions:",
    actions.shape,
    actions.dtype
)

print(
    "Target XY world:",
    target_xy_world.shape,
    target_xy_world.dtype
)

print(
    "Target X range:",
    float(
        target_xy_world[:, 0].min()
    ),
    float(
        target_xy_world[:, 0].max()
    )
)

print(
    "Target Y range:",
    float(
        target_xy_world[:, 1].min()
    ),
    float(
        target_xy_world[:, 1].max()
    )
)

print(
    "Episodes:",
    len(episode_lengths)
)

print(
    "Transitions:",
    len(actions)
)

print(
    "Mean episode length:",
    episode_lengths.mean()
)

print(
    "Min episode length:",
    episode_lengths.min()
)

print(
    "Max episode length:",
    episode_lengths.max()
)

print(
    "=" * 70
)


assert len(frames) == (
    len(actions)
    +
    len(episode_lengths)
)

assert (
    len(target_xy_world)
    ==
    len(frames)
)


# ============================================================
# Save
# ============================================================

np.savez_compressed(
    OUTPUT_PATH,

    frames=frames,
    actions=actions,

    # One target coordinate for every frame.
    target_xy_world=target_xy_world,

    episode_lengths=episode_lengths,

    frame_offsets=frame_offsets,
    action_offsets=action_offsets,
)


print(
    "\nSaved to:",
    OUTPUT_PATH
)
