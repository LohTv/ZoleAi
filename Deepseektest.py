import numpy as np
import random

# ======================
# Game Constants & Setup
# ======================
SUITS = ['H', 'D', 'C', 'S']
RANKS = ['7', '8', '9', '10', 'J', 'Q', 'K', 'A']
CARD_POINTS = {rank: idx for idx, rank in enumerate(RANKS)}  # Higher index = better

# Card encoding/decoding
card_to_idx = {f"{suit}_{rank}": i for i, (suit, rank) in enumerate([(s, r) for s in SUITS for r in RANKS])}
idx_to_card = {v: k for k, v in card_to_idx.items()}


# =============
# Game Logic
# =============
class ZoleGame:
    def __init__(self):
        self.deck = list(card_to_idx.keys())
        self.trump_suit = None
        self.players = [Player(), Player(), Player()]
        self.current_trick = []
        self.bids = [0, 0, 0]
        self.soloist = None
        self.reset()

    def reset(self):
        random.shuffle(self.deck)
        self.trump_suit = None
        self.current_trick = []
        self.bids = [0, 0, 0]
        self.soloist = None

        # Deal cards (simplified: 10 cards each + 2 kitty)
        for p in self.players:
            p.hand = self.deck[:10]
            self.deck = self.deck[10:]
        self.kitty = self.deck[:2]

    def get_state(self, player_id):
        """Create numerical state representation for a player"""
        player = self.players[player_id]

        # Hand encoding (32-dim)
        hand_enc = np.zeros(32)
        for card in player.hand:
            hand_enc[card_to_idx[card]] = 1

        # Trump encoding (4-dim)
        trump_enc = np.zeros(4)
        if self.trump_suit:
            trump_enc[SUITS.index(self.trump_suit)] = 1

        # Trick encoding (32-dim)
        trick_enc = np.zeros(32)
        for card in self.current_trick:
            trick_enc[card_to_idx[card]] = 1

        return np.concatenate([hand_enc, trump_enc, trick_enc])

    def step(self, player_id, action_idx):
        """Execute one action (play a card)"""
        done = False
        reward = 0
        card = idx_to_card[action_idx]

        # Validate action
        if card not in self.players[player_id].hand:
            reward = -10  # Penalize invalid moves
            done = True
            return self.get_state(player_id), reward, done

        # Play card
        self.players[player_id].hand.remove(card)
        self.current_trick.append(card)

        # Complete trick
        if len(self.current_trick) == 3:
            winner = self.resolve_trick()
            reward = 5 if winner == player_id else 0
            self.current_trick = []

            # Check game end
            if all(len(p.hand) == 0 for p in self.players):
                done = True
                reward += 20 if self.soloist == player_id else -10
        else:
            reward = 0

        return self.get_state(player_id), reward, done

    def resolve_trick(self):
        """Determine trick winner (simplified)"""
        trick = self.current_trick
        lead_suit = trick[0].split('_')[0]
        trumps = [c for c in trick if c.startswith(self.trump_suit)] if self.trump_suit else []

        if trumps:
            winning_cards = trumps
            suit = self.trump_suit
        else:
            winning_cards = [c for c in trick if c.startswith(lead_suit)]
            suit = lead_suit

        # Find highest card in winning suit
        max_rank = -1
        winner = 0
        for i, card in enumerate(trick):
            c_suit, c_rank = card.split('_')
            if c_suit == suit and CARD_POINTS[c_rank] > max_rank:
                max_rank = CARD_POINTS[c_rank]
                winner = i

        return winner


# =============
# AI Components
# =============
class DQN:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros(output_size)

    def forward(self, x):
        x = x.reshape(1, -1)
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = np.maximum(0, self.z1)  # ReLU
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        return self.z2[0]

    def backward(self, x, target, lr=0.001):
        x = x.reshape(1, -1)
        output = self.forward(x)
        delta2 = (output - target).reshape(1, -1)
        delta1 = np.dot(delta2, self.W2.T) * (self.z1 > 0)

        self.W2 -= lr * np.dot(self.a1.T, delta2)
        self.b2 -= lr * delta2.mean(axis=0)
        self.W1 -= lr * np.dot(x.T, delta1)
        self.b1 -= lr * delta1.mean(axis=0)


class QLearningAgent:
    def __init__(self, state_size=72, action_size=32):
        self.q_net = DQN(state_size, 64, action_size)
        self.memory = []
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(32)
        else:
            q_values = self.q_net.forward(state)
            return np.argmax(q_values)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size=32, lr=0.001):
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                target = reward + self.gamma * np.max(self.q_net.forward(next_state))

            current_q = self.q_net.forward(state)
            current_q[action] = target
            self.q_net.backward(state, current_q, lr=lr)

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# =============
# Training Loop
# =============
class Player:
    def __init__(self):
        self.hand = []


if __name__ == "__main__":
    agent = QLearningAgent()
    episodes = 1000

    for ep in range(episodes):
        game = ZoleGame()
        state = game.get_state(0)  # Assume AI is player 0
        total_reward = 0
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, done = game.step(0, action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        agent.replay()
        print(f"Episode {ep + 1}/{episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")

    # Save weights (example)
    np.savez("zole_ai_weights.npz",
             W1=agent.q_net.W1, b1=agent.q_net.b1,
             W2=agent.q_net.W2, b2=agent.q_net.b2)


