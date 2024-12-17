import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Simulated historical Vietlott data
# Replace this with actual historical lottery data
# Columns: ['draw_date', 'winning_numbers']
data = {
    "draw_date": pd.date_range(start="2023-01-01", periods=100, freq="W"),
    "winning_numbers": [
        sorted(np.random.choice(range(1, 46), 6, replace=False)) for _ in range(100)
    ]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Split the winning numbers into separate columns for analysis
for i in range(6):
    df[f"number_{i+1}"] = df["winning_numbers"].apply(lambda x: x[i])

# Display the first few rows of the dataset
print("Historical Lottery Data:")
print(df.head())

# Frequency Analysis
all_numbers = [num for numbers in df["winning_numbers"] for num in numbers]
number_counts = Counter(all_numbers)

# Plot the frequency of each number
plt.figure(figsize=(10, 6))
plt.bar(number_counts.keys(), number_counts.values(), color='skyblue')
plt.title("Frequency of Winning Numbers")
plt.xlabel("Numbers")
plt.ylabel("Frequency")
plt.xticks(range(1, 46))
plt.show()

# Statistical Analysis
print("\nStatistical Summary:")
for i in range(1, 7):
    print(f"Number {i} Summary:\n{df[f'number_{i}'].describe()}")

# Analyze most common pairs of numbers
from itertools import combinations

pair_counts = Counter()
for numbers in df["winning_numbers"]:
    for pair in combinations(numbers, 2):
        pair_counts[pair] += 1

most_common_pairs = pair_counts.most_common(10)
print("\nMost Common Pairs of Numbers:")
for pair, count in most_common_pairs:
    print(f"Pair {pair} appeared {count} times")

# Placeholder for "prediction" (random selection)
predicted_numbers = sorted(np.random.choice(range(1, 46), 6, replace=False))
print("\nRandomly Selected Numbers (for fun, NOT a prediction):")
print(predicted_numbers)
