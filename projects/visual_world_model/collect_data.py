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
# ============================================================

all_frames = []
all_actions = []

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
    # Initial frame o_0
    # ------------------------------------------

    frame = env.render()

    episode_frames = [
        frame.copy()
    ]

    episode_actions = []


    terminated = False
    truncated = False

    step = 0


    while not (
        terminated or truncated
    ):

        # ======================================
        # Current:
        #
        # episode_frames[-1] = o_t
        # ======================================


        # TODO 1
        # sample random action a_t

        action = env.action_space.sample()


        # TODO 2
        # environment transition
        #
        # s_t --a_t--> s_(t+1)

        obs, reward, terminated, truncated, info = env.step(action)


        # TODO 3
        # render o_(t+1)

        next_frame = env.render()


        # ======================================
        # Save:
        #
        # o_t, a_t, o_(t+1)
        #
        # through sequential storage
        # ======================================

        episode_actions.append(
            action.copy()
        )

        episode_frames.append(
            next_frame.copy()
        )


        step += 1


    # ==========================================
    # Check temporal alignment
    # ==========================================

    assert (
        len(episode_frames)
        ==
        len(episode_actions) + 1
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
).astype(np.uint8)


actions = np.stack(
    all_actions,
    axis=0
).astype(np.float32)


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

print("\n" + "=" * 70)

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

print("=" * 70)

assert len(frames) == (
    len(actions)
    +
    len(episode_lengths)
)

np.savez_compressed(
    OUTPUT_PATH,

    frames=frames,
    actions=actions,

    episode_lengths=episode_lengths,

    frame_offsets=frame_offsets,
    action_offsets=action_offsets,
)


print(
    "\nSaved to:",
    OUTPUT_PATH
)