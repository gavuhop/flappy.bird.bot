from agent.dqn_agent import DQNAgent
import numpy as np


def test_dqn_agent():
    print("Testing DQN Agent implementation...")

    # Initialize agent
    state_size = 4
    action_size = 2
    agent = DQNAgent(state_size, action_size)

    print(f"Agent initialized with state_size={state_size}, action_size={action_size}")
    print(f"Initial epsilon: {agent.epsilon}")

    # Test remember method
    state = np.random.rand(state_size)
    action = 1
    reward = 1.0
    next_state = np.random.rand(state_size)
    done = False

    agent.remember(state, action, reward, next_state, done)
    print(f"Memory size after adding one experience: {len(agent.memory)}")

    # Test act method
    action = agent.act(state)
    print(f"Action taken: {action}")

    # Add more experiences to test replay
    for _ in range(50):
        state = np.random.rand(state_size)
        action = np.random.randint(0, action_size)
        reward = np.random.rand()
        next_state = np.random.rand(state_size)
        done = np.random.choice([True, False])
        agent.remember(state, action, reward, next_state, done)

    print(f"Memory size after adding more experiences: {len(agent.memory)}")

    # Test replay method
    agent.replay()
    print(f"Epsilon after replay: {agent.epsilon}")

    # Test save and load
    agent.save("test_agent.pth")
    print("Agent saved to test_agent.pth")

    # Create a new agent and load the saved weights
    new_agent = DQNAgent(state_size, action_size)
    new_agent.load("test_agent.pth")
    print("Agent loaded from test_agent.pth")

    print("\nDQN Agent test completed successfully!")


if __name__ == "__main__":
    test_dqn_agent()
