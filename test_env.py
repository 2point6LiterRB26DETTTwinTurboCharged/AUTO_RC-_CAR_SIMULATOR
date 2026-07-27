# test_env.py
from env_gym import RCCarEnv

env = RCCarEnv(num_beams=120)
obs, info = env.reset()

print("✅ Gymnasium Environment Successfully Initialized!")
print(f"Observation Shape: {obs.shape}")
print(f"Action Space: {env.action_space}")

# Run 50 random test steps
for step in range(50):
    action = env.action_space.sample()  # Random steering and speed
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {step+1}: Reward = {reward:.2f} | Terminated = {terminated}")

    if terminated or truncated:
        print("Episode ended, resetting...")
        obs, info = env.reset()

env.close()