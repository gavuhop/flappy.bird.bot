from game.flappy_bird import FlappyBird
from agent.dqn_agent import DQNAgent
import os


def train():
    # Initialize environment and agent
    env = FlappyBird()
    state_size = 4  # [bird_y, bird_velocity, pipe_gap_y, pipe_distance]
    action_size = 2  # [do_nothing, flap]
    agent = DQNAgent(state_size, action_size)
    
    # Training parameters
    episodes = 1000
    target_update_frequency = 10
    
    # Create directory for saving weights
    if not os.path.exists('weights'):
        os.makedirs('weights')
    
    # Training loop
    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        
        while True:
            # For visualization
            env.render()
            
            # Get action
            action = agent.act(state)
            
            # Take action and get reward
            next_state, reward, done, score = env.step(action)
            
            # Remember the previous state, action, reward, and done
            agent.remember(state, action, reward, next_state, done)
            
            # Update state
            state = next_state
            
            # Accumulate reward
            total_reward += reward
            
            # Train the agent with experience replay
            agent.replay()
            
            if done:
                # Update target model periodically
                if e % target_update_frequency == 0:
                    agent.update_target_model()
                
                print(f"Episode: {e+1}/{episodes}, Score: {score}, "
                      f"Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")
                
                # Save weights periodically
                if (e + 1) % 100 == 0:
                    weights_file = f"weights/flappy_bird_dqn_{e+1}.h5"
                    agent.save(weights_file)
                    print(f"Weights saved to {weights_file}")
                break
    
    env.close()


def test(weights_path):
    # Check if weights file exists
    if not os.path.exists(weights_path):
        print(f"Error: Weights file '{weights_path}' not found!")
        print("Please make sure to:")
        print("1. Train the model first using 'train' mode")
        print("2. Provide the correct path to the weights file")
        print("3. Check if the weights file exists in the specified location")
        return
    
    # Initialize environment and agent
    env = FlappyBird()
    state_size = 4
    action_size = 2
    agent = DQNAgent(state_size, action_size)
    
    try:
        # Load trained weights
        agent.load(weights_path)
        print(f"Successfully loaded weights from {weights_path}")
        agent.epsilon = 0.0  # No exploration during testing
        
        # Test loop
        while True:
            state = env.reset()
            total_reward = 0
            
            while True:
                env.render()
                
                action = agent.act(state)
                next_state, reward, done, score = env.step(action)
                
                state = next_state
                total_reward += reward
                
                if done:
                    print(f"Score: {score}, Total Reward: {total_reward:.2f}")
                    break
    
    except Exception as e:
        print(f"Error loading weights: {str(e)}")
        print("Please make sure the weights file is valid and not corrupted.")
    finally:
        env.close()


if __name__ == "__main__":
    print("Flappy Bird AI - DQN Agent")
    print("==========================")
    print("Available modes:")
    print("1. train - Train a new agent")
    print("2. test  - Test a trained agent")
    print("==========================")
    
    mode = input("Enter mode (train/test): ").lower().strip()
    
    if mode == "train":
        print("\nStarting training mode...")
        print("Training will run for 1000 episodes")
        print("Weights will be saved every 100 episodes in the 'weights' folder")
        train()
    elif mode == "test":
        if not os.path.exists('weights'):
            print("\nError: No weights directory found!")
            print("Please train the agent first to generate weights.")
            exit(1)
            
        print("\nAvailable weight files:")
        weight_files = [f for f in os.listdir('weights') if f.endswith('.h5')]
        if not weight_files:
            print("No weight files found in 'weights' directory!")
            print("Please train the agent first to generate weights.")
            exit(1)
            
        for i, file in enumerate(weight_files, 1):
            print(f"{i}. {file}")
            
        weights_path = input("\nEnter path to weights file: ").strip()
        test(weights_path)
    else:
        print("Invalid mode. Please choose 'train' or 'test'.") 