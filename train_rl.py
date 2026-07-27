# train_rl.py
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from env_gym import RCCarEnv

def main():
    print("🚀 Initializing RL Training Environment...")
    env = RCCarEnv(room_size=10.0, num_beams=120, max_range=5.0)

    # Save model checkpoints every 20,000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=20000,
        save_path="./models/",
        name_prefix="ppo_rccar_model"
    )

    # Configure PPO Agent
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log="./tensorboard_logs/"
    )

    print("🧠 Starting PPO Model Training (100,000 timesteps)...")
    model.learn(total_timesteps=100000, callback=checkpoint_callback)

    # Save final model
    os.makedirs("./models", exist_ok=True)
    model.save("./models/ppo_rccar_final")
    print("✅ Model trained and saved to './models/ppo_rccar_final.zip'")

if __name__ == "__main__":
    main()