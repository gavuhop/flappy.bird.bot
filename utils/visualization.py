import os
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("TkAgg")  # Use TkAgg backend for interactive display


class TrainingVisualizer:
    def __init__(self, save_dir="plots"):
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Initialize metrics storage
        self.scores = []
        self.rewards = []
        self.epsilons = []
        self.episodes = []
        self.avg_scores = []  # For moving average

        # Moving average window
        self.ma_window = 10

        # Initialize matplotlib figure
        plt.ion()  # Turn on interactive mode
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.fig.canvas.manager.set_window_title("Training Metrics")

        # Set up the plot
        self.ax.set_title("Training Metrics")
        self.ax.set_xlabel("Episode")
        self.ax.set_ylabel("Value")
        self.ax.grid(True, linestyle="--", alpha=0.7)

        # Initialize empty lines
        (self.score_line,) = self.ax.plot([], [], "b-", label="Score")
        (self.ma_line,) = self.ax.plot([], [], "b--", label="Moving Average")
        (self.reward_line,) = self.ax.plot([], [], "g-", label="Reward")
        (self.epsilon_line,) = self.ax.plot([], [], "r-", label="Epsilon")

        # Add legend
        self.ax.legend(loc="upper right")

        # Draw the initial plot
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update(self, episode, score, reward, epsilon):
        # Store metrics
        self.episodes.append(episode)
        self.scores.append(score)
        self.rewards.append(reward)
        self.epsilons.append(epsilon)

        # Calculate moving average of scores
        window_size = min(self.ma_window, len(self.scores))
        if window_size > 0:
            avg_score = sum(self.scores[-window_size:]) / window_size
            self.avg_scores.append(avg_score)

        # Print progress to console (all on one line)
        avg_score_str = ""
        if len(self.avg_scores) > 0:
            avg_score_str = f" | Avg: {self.avg_scores[-1]:.2f}"

        print(
            f"Episode {episode} | Score: {score} | "
            f"Reward: {reward:.2f} | Epsilon: {epsilon:.2f}"
            f"{avg_score_str}"
        )

        # Update visualization
        self._update_plot()

    def _update_plot(self):
        """Update the matplotlib plot"""
        # Update data
        self.score_line.set_data(self.episodes, self.scores)

        # Update moving average if available
        if len(self.avg_scores) > 0:
            ma_episodes = self.episodes[-len(self.avg_scores) :]
            self.ma_line.set_data(ma_episodes, self.avg_scores)
            self.ma_line.set_visible(True)
        else:
            self.ma_line.set_visible(False)

        self.reward_line.set_data(self.episodes, self.rewards)
        self.epsilon_line.set_data(self.episodes, self.epsilons)

        # Adjust axis limits
        if len(self.episodes) > 0:
            self.ax.set_xlim(0, max(self.episodes) * 1.05)

            # For y-axis, consider all metrics
            y_min = min(min(self.scores), min(self.rewards), min(self.epsilons))
            y_max = max(max(self.scores), max(self.rewards), max(self.epsilons))

            # Add some padding
            y_range = y_max - y_min
            if y_range == 0:
                y_range = 1
            self.ax.set_ylim(y_min - y_range * 0.1, y_max + y_range * 0.1)

        # Redraw the plot
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def save_plot(self, filename="training_metrics.png"):
        """Save the current plot as an image"""
        try:
            self.fig.savefig(os.path.join(self.save_dir, filename))
        except Exception as e:
            print(f"Error saving plot: {e}")

    def close(self):
        """Close the visualization"""
        # Save the final plot
        self._update_plot()
        self.save_plot("final_training_metrics.png")

        # Print summary statistics
        if self.scores:
            print("\nTraining Summary:")
            print(f"Final Score: {self.scores[-1]}")
            print(f"Best Score: {max(self.scores)}")
            print(f"Average Score: {sum(self.scores) / len(self.scores):.2f}")
            if len(self.avg_scores) > 0:
                print(f"Final Moving Average: {self.avg_scores[-1]:.2f}")
                print(f"Best Moving Average: {max(self.avg_scores):.2f}")

        # Close the figure
        plt.close(self.fig)
