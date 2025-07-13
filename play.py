import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
import time
import warnings
warnings.filterwarnings("ignore")

def create_atari_env():
    """Create Atari environment with proper wrappers (same as train.py)"""
    try:
        import ale_py
        ale_py.ALEInterface.setLoggerMode(ale_py.LoggerMode.Error)
    except:
        pass
    env = gym.make("ALE/Breakout-v5", render_mode="human")
    print("Using environment: ALE/Breakout-v5")
    env = AtariWrapper(env)
    return env

def play_game():
    """Load trained model and play Atari Breakout with GreedyQPolicy"""
    print("Loading trained DQN model...")
    try:
        model = DQN.load("dqn_model")
        print("[SUCCESS] Model loaded successfully!")
    except FileNotFoundError:
        print("[ERROR] dqn_model.zip not found. Please run train.py first.")
        return
    except Exception as e:
        print(f"[ERROR] Error loading model: {e}")
        return
    env = create_atari_env()
    print("\nStarting Atari Breakout with GreedyQPolicy (deterministic actions)...")
    print("The agent will select actions with highest Q-values")
    print("Press Ctrl+C to stop\n")
    try:
        episodes = 3
        total_reward = 0
        for episode in range(episodes):
            obs, _ = env.reset()
            episode_reward = 0
            step_count = 0
            done = False
            print(f"Episode {episode + 1}/{episodes} starting...")
            while not done and step_count < 1000:
                action, _ = model.predict(obs, deterministic=True)  # GreedyQPolicy
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                step_count += 1
                time.sleep(0.05)
            total_reward += episode_reward
            print(f"Episode {episode + 1} completed: Reward = {episode_reward}, Steps = {step_count}")
        avg_reward = total_reward / episodes
        print("\n" + "="*50)
        print("ATARI BREAKOUT PERFORMANCE SUMMARY")
        print("="*50)
        print(f"Episodes played: {episodes}")
        print(f"Average reward: {avg_reward:.2f}")
        print("Agent used GreedyQPolicy (deterministic=True)")
    except KeyboardInterrupt:
        print("\nGame stopped by user")
    finally:
        env.close()

if __name__ == "__main__":
    play_game()