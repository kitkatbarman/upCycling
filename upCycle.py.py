import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

###############################################################################
# 1) CONFIGURATION
###############################################################################

plt.style.use('dark_background')


# Base probabilities to test:
# We'll map each of these to a custom label in `custom_labels`.
base_probs_list = [0.10, 0.128, 0.16, 0.188, 0.248]

# Assign a word (label) for each base probability to appear in the legend
custom_labels = {
    0.10:  "Normal",
    0.128: "Uncommon",
    0.16:  "Rare",
    0.188: "Epic",
    0.248: "Legendary"
}

# Number of items initially in Normal
NUM_ITEMS = 100000

# 75% loss each round (below Legendary)
LOSS_PROB = 0.75

# Safety cap on number of rounds
MAX_ROUNDS = 8

# Rarity order and colors
RARITY_ORDER = ["Normal", "Uncommon", "Rare", "Epic", "Legendary"]
RARITY_COLORS = {
    "Normal":    '#BBBBBB',
    "Uncommon":  '#5ec197',
    "Rare":      '#57b2f2',
    "Epic":      '#d487ff',
    "Legendary": '#f7d97e'
}

###############################################################################
# 2) NESTED PROBABILITIES
###############################################################################

def build_nested_probs(base_p):

    p1 = base_p
    p2 = p1 * 0.10
    p3 = p2 * 0.10
    p4 = p3 * 0.10
    return [p1, p2, p3, p4]


def single_roll_distribution(num_in_bin, rarity_index, base_p):

    if rarity_index == 4:
        return {4: num_in_bin}

    tiers_above = 4 - rarity_index
    nested = build_nested_probs(base_p)[:tiers_above]
    if not nested:
        # no higher tier, remain where you are
        return {rarity_index: num_in_bin}

    thresholds = np.cumsum(nested)
    out_dict = {}
    cum_prev = 0.0
    for i, th in enumerate(thresholds):
        frac = th - cum_prev
        count_here = int(round(num_in_bin * frac))
        new_index = rarity_index + (i + 1)
        out_dict[new_index] = out_dict.get(new_index, 0) + count_here
        cum_prev = th

    # remainder remain in current rarity
    remain_frac = 1.0 - thresholds[-1]
    if remain_frac < 0:
        remain_frac = 0
    remain_count = int(round(num_in_bin * remain_frac))
    out_dict[rarity_index] = out_dict.get(rarity_index, 0) + remain_count
    return out_dict


###############################################################################
# 3) ONE ROUND OF PROCESSING (WITH 75% LOSS BELOW LEGENDARY)
###############################################################################

def apply_round(distribution, base_p, loss_prob=0.75):

    new_dist = np.zeros(5, dtype=int)
    
    # Legendary stays
    new_dist[4] = distribution[4]
    
    # process bins 0..3
    for rarity_idx in [0,1,2,3]:
        count = distribution[rarity_idx]
        if count <= 0:
            continue
        
        # remove 75%
        lost = int(round(count * loss_prob))
        survivors = count - lost
        if survivors <= 0:
            continue

        outcome = single_roll_distribution(survivors, rarity_idx, base_p)
        for r_i, c_i in outcome.items():
            new_dist[r_i] += c_i
    
    return new_dist


###############################################################################
# 4) RUN THE SIM UNTIL ALL LEGENDARY OR STABLE, RECORD HISTORY
###############################################################################

def run_sim_and_get_history(base_p, num_items, loss_prob=0.75, max_rounds=50):
    dist = np.zeros(5, dtype=int)
    dist[0] = num_items

    history = [dist.copy()]  # store the initial distribution (round 0)
    
    for round_i in range(max_rounds):
        new_dist = apply_round(dist, base_p, loss_prob)
        
        # store
        history.append(new_dist.copy())

        # check stability
        if np.array_equal(new_dist, dist):
            break
        
        dist = new_dist
        
        # check if all are Legendary
        total = dist.sum()
        if total == dist[4]:
            break

    return history  # list of arrays, each array length=5


###############################################################################
# 5) MAIN: RUN FOR MULTIPLE PROBS, PLOT "ROUND VS NUMBER OF ITEMS"
###############################################################################

def main():
    # We'll store the round-by-round history for each base probability
    all_histories = {}
    
    for bp in base_probs_list:
        label_name = custom_labels.get(bp, f"{bp*100:.2f}%")  # fallback if not in dict
        print(f"Running for base_p={bp*100:.2f}% => label='{label_name}'")
        hist = run_sim_and_get_history(bp, NUM_ITEMS, LOSS_PROB, MAX_ROUNDS)
        all_histories[bp] = hist
    
    # We'll create 5 subplots, one for each rarity
    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(18,5), sharey=True)
    
    # We'll define alpha range for base probabilities
    alpha_values = np.linspace(0.4, 1.0, len(base_probs_list))

    for cat_idx, rarity in enumerate(RARITY_ORDER):
        ax = axes[cat_idx]
        color = RARITY_COLORS[rarity]

        # We'll plot one line per base probability
        for i, bp in enumerate(base_probs_list):
            alpha_val = alpha_values[i]
            # get the "word" label for this base probability
            label_name = custom_labels.get(bp, f"{bp*100:.2f}%")

            # extract the time series for this category
            hist_list = all_histories[bp]
            y_vals = [dist[cat_idx] for dist in hist_list]
            x_vals = range(len(hist_list))  # rounds

            # Only put legend labels on the first subplot for clarity
            label = None
            if cat_idx == 0:
                label = label_name

            ax.plot(
                x_vals,
                y_vals,
                color=color,
                alpha=alpha_val,
                label=label,
                linewidth=2
            )
        
        ax.set_title(rarity, color='white', fontsize=12)
        ax.set_xlabel("Round #", fontsize=10, color='white')
        if cat_idx == 0:
            ax.set_ylabel("Number of Items (Log Scale)", fontsize=10, color='white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, color='white', linestyle='--')

        # Add legend only to the first subplot
        if cat_idx == 0:
            ax.legend(
                title="4x Quality Module",
                fontsize=8,
                title_fontsize=9,
                facecolor='#333333',
                framealpha=0.5,
                loc='best'
            )
    
    # Adjust layout
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
