from itertools import combinations
import pandas as pd
import random

max_items = 3
nonthreat_dupes = 2
shuffle = True

nonthreat = [
    'wallet_and_keys',
    'phone',
    'laptop',
    'ipad',
    'handbag',
    'umbrella',
]

threat = [
    'ar15',
    'shotgun',
    'sw357',
    'glock',
    'beretta',
    'machete'
]

nonthreat_combs = []
for i in range(1, max_items + 1):
    nonthreat_combs.extend([list(c) for c in combinations(nonthreat, i)])
print('Nonthreat:', len(nonthreat_combs), '| dupes:', nonthreat_dupes)

nonthreat_combs_under = [c for c in nonthreat_combs if len(c) < max_items]

threat_combs = []
for t in threat:
    threat_combs.extend([c + [t] for c in nonthreat_combs_under])
threat_combs.extend([[t] for t in threat])
print('Threat:', len(threat_combs))

nonthreat_combs_duped = []
for i in range(nonthreat_dupes):
    nonthreat_combs_duped.extend(nonthreat_combs.copy())

combs = nonthreat_combs_duped + threat_combs
print('Total:', len(combs))

if shuffle:
    random.shuffle(combs)

combs = pd.DataFrame(combs)
print(combs)

combs.to_csv('data/test_matrix.csv')