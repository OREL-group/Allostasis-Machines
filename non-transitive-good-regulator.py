import numpy as np

class NonTransitiveSystem:
    """The System (S) being regulated. It follows a non-transitive circular logic."""
    def __init__(self):
        self.states = ['Rock', 'Paper', 'Scissors']
        self.current_state = 'Rock'
        # Non-transitive logic: beats[A] = B means A beats B
        self.beats = {'Rock': 'Scissors', 'Paper': 'Rock', 'Scissors': 'Paper'}

    def get_next_state(self):
        """Simulates the system's next move based on its internal circular dynamics."""
        # For this model, the system follows a predictable but circular pattern:
        # Rock -> Paper -> Scissors -> Rock...
        idx = (self.states.index(self.current_state) + 1) % 3
        self.current_state = self.states[idx]
        return self.current_state

class InternalModel:
    """The Internal Model (M) required by EGRT to approximate the system (S)."""
    def __init__(self):
        # Tracks transitions: observed state at t-1 -> count of states at t
        self.transitions = {
            'Rock': {'Rock': 0, 'Paper': 0, 'Scissors': 0},
            'Paper': {'Rock': 0, 'Paper': 0, 'Scissors': 0},
            'Scissors': {'Rock': 0, 'Paper': 0, 'Scissors': 0}
        }
        self.last_observed = None

    def update(self, current_observation):
        """Acquires state information from S to verify content via feedback."""
        if self.last_observed:
            self.transitions[self.last_observed][current_observation] += 1
        self.last_observed = current_observation

    def predict_next(self):
        """Uses the model to estimate future states of the system."""
        if not self.last_observed:
            return 'Rock'
        # Predict based on the most frequent transition observed so far
        possible_next = self.transitions[self.last_observed]
        return max(possible_next, key=possible_next.get)

class EGRT_Regulator:
    """The Regulator (R) that uses Model (M) to control outcomes (Z)."""
    def __init__(self):
        self.model = InternalModel()
        # Mapping for counter-moves: To beat X, play Y
        self.counter_moves = {'Rock': 'Paper', 'Paper': 'Scissors', 'Scissors': 'Rock'}

    def select_action(self):
        """Selects the best action (R) to reach target G (winning)."""
        predicted_s = self.model.predict_next()
        return self.counter_moves[predicted_s]

## --- Simulation of Good Regulation ---
system = NonTransitiveSystem()
regulator = EGRT_Regulator()

print(f"{'Step':<5} | {'System (S)':<10} | {'Prediction (M)':<15} | {'Regulator (R)':<15} | {'Outcome (Z)'}")
print("-" * 75)

wins = 0
for step in range(1, 11):
    # 1. Regulator predicts based on its internal model
    prediction = regulator.model.predict_next()

    # 2. Regulator selects action (Counter-move)
    action = regulator.select_action()

    # 3. System moves to its next state
    actual_s = system.get_next_state()

    # 4. Feedback: Regulator observes system to update its model
    regulator.model.update(actual_s)

    # 5. Outcome determination
    if system.beats[action] == actual_s:
        outcome = "WIN (Goal G)"
        wins += 1
    elif action == actual_s:
        outcome = "TIE"
    else:
        outcome = "LOSS"

    print(f"{step:<5} | {actual_s:<10} | {prediction:<15} | {action:<15} | {outcome}")

print(f"\nTotal wins (Success of Regulation): {wins}/10")

## Create the animation object
ani = FuncAnimation(fig, update, frames=300, interval=30, blit=True)

## Close the figure to prevent a static plot from being displayed alongside the animation
plt.close(fig)

## Display animation as HTML in the notebook
HTML(ani.to_jshtml())
