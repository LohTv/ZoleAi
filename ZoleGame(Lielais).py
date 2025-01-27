import random

num_player = 3
ranks = ['7', '8', '9', '10', 'J', 'Q', 'K', 'A']
suits = ['hearts', 'diamonds', 'clubs', 'spades']
card_values = {'7': 0, '8': 1, '9': 2, '10': 3, 'J': 4, 'Q': 5, 'K': 6, 'A': 7}
players = {key: [] for key in range(num_player)}
# Create the deck
def create_deck():
    return [(rank, suit) for rank in ranks for suit in suits]

# Shuffle the deck
def shuffle_deck(deck):
    random.shuffle(deck)

deck = create_deck()
shuffle_deck(deck)
galds = [deck[0], deck[1]]
deck = deck[2::]

def deal_cards(deck, num_players, players):
    for i in range(len(deck)):
        players[i%num_players].append(deck[i])

deal_cards(deck, num_player, players)

for i in range(num_player):
    print(players[i])
    print(input(f'Player{i+1} choose: ' ))



