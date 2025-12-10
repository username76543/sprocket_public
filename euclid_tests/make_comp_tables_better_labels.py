import os
import ast
import json
import numpy as np
from scipy.stats import rankdata, studentized_range
from math import sqrt
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import LabelEncoder



def rank_floats(values):
    """
    Ranks a list of floats such that:
      - The highest value gets rank 0
      - The second highest gets rank 1, etc.
      - Tied values receive the same rank.
    Returns a list of ranks corresponding to the input order.
    """
    # Get the unique values sorted descending
    sorted_unique = sorted(set(values), reverse=True)
    
    # Create a mapping of value -> rank
    rank_map = {val: rank for rank, val in enumerate(sorted_unique)}
    
    # Map each original value to its rank
    return [rank_map[v] for v in values]


def scatter_grid(pairs, titles=None, max_per_fig=9, save_prefix="scatter_page_accs", fig_title="Comparison Plots"):
    """
    Plot multiple x–y scatterplots in grid layout, adding y=x line.
    Automatically splits into multiple figures if there are more than max_per_fig.

    Args:
        pairs (list of tuples): Each element is a (x, y) pair of equal-length lists or arrays.
        titles (list of str, optional): Titles for each scatterplot.
        max_per_fig (int, optional): Maximum number of subplots per figure (default: 9).
        save_prefix (str, optional): Prefix for saved figure filenames (default: 'scatter_page').
    """
    n = len(pairs)
    titles = titles or [f"Pair {i+1}" for i in range(n)]

    # Determine how many figures are needed
    n_figs = math.ceil(n / max_per_fig)
    
    MIN_LIMIT = .001
    MAX_LIMIT = 1.01

    for fig_idx in range(n_figs):
        start = fig_idx * max_per_fig
        end = min(start + max_per_fig, n)
        subset = pairs[start:end]
        subset_titles = titles[start:end]

        num_subplots = len(subset)
        cols = math.ceil(math.sqrt(num_subplots))
        rows = math.ceil(num_subplots / cols)
        
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        axes = np.array(axes).flatten() if num_subplots > 1 else np.array([axes])
        fig.suptitle(fig_title, fontsize=16, fontweight='bold', y=.97)

        for i, (x, y) in enumerate(subset):
            ax = axes[i]
            ax.scatter(x, y, alpha=0.7)
            ax.set_title(subset_titles[i])
            ax.set_xlabel(subset_titles[i].split('/')[0])
            ax.set_ylabel(subset_titles[i].split('/')[1])

            '''# Draw y=x line
            min_val = min(min(x), min(y))
            max_val = max(max(x), max(y))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=1)
            ax.set_yscale('log')
            ax.set_xscale('log')'''
            
            # 1. Set the fixed axes limits (must be done before log scale)
            ax.set_xlim(MIN_LIMIT, MAX_LIMIT)
            ax.set_ylim(MIN_LIMIT, MAX_LIMIT)
            
            # 2. Apply the log scale
            ax.set_yscale('log')
            ax.set_xscale('log')
            
            # 3. Draw y=x line using the new fixed limits
            # The line should span the entire visible area
            ax.plot([MIN_LIMIT, MAX_LIMIT], [MIN_LIMIT, MAX_LIMIT], 'r--', lw=1)

        # Hide unused subplots
        for ax in axes[num_subplots:]:
            ax.axis('off')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Save and show
        filename = f"{save_prefix}_{fig_idx+1}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"✅ Saved: {filename}")

        #plt.show()

def nemenyi_cd(num_methods, num_datasets, alpha=0.05):
    """
    Compute Critical Difference (CD) for the Nemenyi post-hoc test
    based on Demšar (2006).
    """
    q_alpha = {
        0.10: [0, 1.65, 1.96, 2.10, 2.22, 2.33, 2.39, 2.44, 2.49, 2.52, 2.56],
        0.05: [0, 1.96, 2.24, 2.34, 2.43, 2.52, 2.56, 2.61, 2.65, 2.68, 2.72],
        0.01: [0, 2.57, 2.91, 3.05, 3.17, 3.27, 3.33, 3.39, 3.43, 3.47, 3.50],
    }
    q = q_alpha.get(alpha, q_alpha[0.05])[min(num_methods, 10)]
    return q * sqrt(num_methods * (num_methods + 1) / (6.0 * num_datasets))


def visualize_ranking_results_with_cd(results, num_datasets, alpha=0.05, save_path=None, suptitle='CD and Best Performances'):
    """
    Visualize ranking results using:
      1. A Critical Difference (CD) diagram (single line layout with evenly spaced labels)
      2. A bar chart showing frequency of best performance
    
    Parameters
    ----------
    results : dict
        {
            'approach_name': {
                'mean_rank': float,
                'rank_counts': {0: int, 1: int, ...}
            },
            ...
        }
    num_datasets : int
        Number of datasets used for ranking.
    alpha : float
        Significance level for CD computation.
    save_path : str or None
        Optional path to save the figure.
    """
    # Extract data
    methods = list(results.keys())
    mean_ranks = np.array([results[m]['mean_rank'] for m in methods])
    best_counts = np.array([results[m]['rank_counts'].get(0, 0) for m in methods])
    
    # Sort by mean rank (ascending = better)
    order = np.argsort(mean_ranks)
    methods = [methods[i] for i in order]
    mean_ranks = mean_ranks[order]
    best_counts = best_counts[order]
    
    num_methods = len(methods)
    cd = nemenyi_cd(num_methods, num_datasets, alpha)
    
    # ---- Create figure ----
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    plt.subplots_adjust(wspace=0.4)
    fig.suptitle(suptitle, fontsize=16, fontweight='bold', y=.97)
    
    # --- (1) Critical Difference Diagram ---
    ax = axs[0]
    min_rank = np.floor(mean_ranks.min() - 0.5)
    max_rank = np.ceil(mean_ranks.max() + 0.5)
    
    # Draw number line
    ax.hlines(0, min_rank, max_rank, color='black', lw=1.2)
    ax.set_xlim(max_rank, min_rank)  # Standard orientation (lower rank on left = better)
    ax.set_ylim(-2.5, 2.5)
    ax.set_yticks([])
    ax.set_xlabel("Mean Rank (lower is better)", fontsize=10)
    ax.set_title(f"Critical Difference Diagram (α={alpha})", fontsize=11)
    
    # Evenly spaced label positions (left to right, matching sorted order)
    label_spacing = (max_rank - min_rank) / (num_methods + 1)
    label_positions = [min_rank + label_spacing * (i + 1) for i in range(num_methods)]
    
    # Alternate labels above and below the line
    for i, (r, method, label_x) in enumerate(zip(mean_ranks, methods, label_positions)):
        # Draw vertical line from rank to label position
        if i % 2 == 0:  # Above the line
            label_y = 1.5
            line_y = 0.15
            va = 'bottom'
        else:  # Below the line
            label_y = -1.5
            line_y = -0.15
            va = 'top'
        
        # Vertical line from rank point to label height
        ax.vlines(r, 0, line_y, color='gray', lw=1, linestyle='--', alpha=0.5)
        
        # Horizontal connector from rank to label position
        ax.plot([r, label_x], [line_y, label_y], 'gray', lw=0.8, linestyle='--', alpha=0.5)
        
        # Plot point at actual rank
        ax.plot(r, 0, 'o', color='C0', markersize=8, zorder=3)
        
        # Label at evenly-spaced position
        ax.text(label_x, label_y, f"{method}\n({r:.2f})",
                ha='center', va=va, fontsize=8.5, 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='lightgray', alpha=0.8))
    
    # CD reference bar (top left)
    cd_bar_start = min_rank + 0.1
    cd_bar_end = cd_bar_start + cd
    ax.hlines(2.0, cd_bar_start, cd_bar_end, lw=2.5, color='black')
    ax.vlines([cd_bar_start, cd_bar_end], 1.95, 2.05, lw=2.5, color='black')
    ax.text((cd_bar_start + cd_bar_end) / 2, 2.2, f"CD = {cd:.2f}", 
            ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Find groups within CD
    groups = []
    i = 0
    while i < num_methods:
        group = [i]
        j = i + 1
        while j < num_methods and abs(mean_ranks[j] - mean_ranks[i]) <= cd:
            group.append(j)
            j += 1
        if len(group) > 1:
            groups.append(group)
        i += 1
    
    # Draw CD group bars (thick horizontal lines connecting non-significantly different methods)
    y_group_start = -0.3
    for idx, group in enumerate(groups):
        x_start = mean_ranks[group[0]]
        x_end = mean_ranks[group[-1]]
        y_group = y_group_start - (idx * 0.2)
        ax.hlines(y_group, x_start, x_end, lw=4, color='red', alpha=0.6, zorder=2)
    
    # --- (2) Bar chart of best counts ---
    ax2 = axs[1]
    ax2.barh(methods, best_counts, color='C1', alpha=0.7, edgecolor='black')
    for y, val in enumerate(best_counts):
        ax2.text(val + 0.3, y, str(val), va='center', fontsize=9, fontweight='bold')
    ax2.set_xlabel("Times ranked #0 (best)", fontsize=10)
    ax2.set_title("Frequency of Best Performance", fontsize=11)
    ax2.invert_yaxis()  # Keep same order as CD diagram
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✅ Saved CD diagram to: {save_path}")
    
    plt.show()
    
def disagreement_stats(pred1, pred2, y_true):
    """
    Q-statistic for classifier diversity
    Q = (N11*N00 - N01*N10) / (N11*N00 + N01*N10)
    where Nij = number of samples where classifier 1 is i, classifier 2 is j
    (1=correct, 0=incorrect)
    """
    n11 = 0
    n10 = 0
    n01 = 0
    n00 = 0
    for i in range(0, len(y_true)):
        #print(i, pred1[i], pred2[i], y_true[i])
        if int(pred1[i]) == int(float(y_true[i])):
            #print('first true')
            if int(pred2[i]) == int(float(y_true[i])):
                #print('11')
                n11+=1
            else:
                n10+=1
        else:
            if int(pred2[i]) == int(float(y_true[i])):
                n01+=1
            else:
                n00+=1
    #print(n11, n00, n10, n01)
    #exit()
    if n11*n00+n01*n10 < 1:
        q_statistic = 0
    else:
        q_statistic = (n11*n00-n01*n10)/(n11*n00+n01*n10)
    disagreement = (n10+n01)/len(y_true)
    double_fault = n00/len(y_true)
    return q_statistic, disagreement, double_fault
    
import matplotlib.pyplot as plt
import numpy as np

def visualize_four_matrices(mats, titles=None, labels=None, save_prefix=None, cmap='viridis', suptitle="Ensemble Analysis Grid"):
    """
    Visualize 4 square matrices (values in [0,1]) in a 2x2 grid of subplots.
    Higher numbers appear with more intense colors, and each cell shows its value.

    Parameters
    ----------
    mats : list of 4 array-like
        List containing 4 square matrices (NumPy arrays or lists of lists).
    titles : list of str, optional
        List of 4 titles for each subplot.
    labels : list of str, optional
        List of labels for the matrix axes (same for all subplots).
    cmap : str, optional
        Matplotlib colormap name (e.g., 'viridis', 'plasma', 'hot').
    """
    if len(mats) != 4:
        raise ValueError("Expected a list of 4 matrices.")

    fig, axes = plt.subplots(2, 2, figsize=(9, 9))
    axes = axes.ravel()
    fig.suptitle(suptitle, fontsize=16, fontweight='bold', y=.97)

    for i, mat in enumerate(mats):
        mat = np.array(mat)
        n = mat.shape[0]
        if mat.shape[0] != mat.shape[1]:
            raise ValueError(f"Matrix {i} is not square: shape {mat.shape}")

        im = axes[i].imshow(mat, cmap=cmap, vmin=0, vmax=1)

        # Title
        if titles and i < len(titles):
            axes[i].set_title(titles[i], fontsize=12)

        # Tick labels
        if labels:
            if len(labels) != n:
                raise ValueError(f"Label list length ({len(labels)}) must match matrix size ({n})")
            axes[i].set_xticks(range(n))
            axes[i].set_yticks(range(n))
            axes[i].set_xticklabels(labels, rotation=45, ha='right')
            axes[i].set_yticklabels(labels)
        else:
            axes[i].set_xticks([])
            axes[i].set_yticks([])

        # Write each cell value (two decimal places)
        for r in range(n):
            for c in range(n):
                value = mat[r, c]
                text_color = 'white' if value > 0.5 else 'black'
                axes[i].text(c, r, f"{value:.2f}",
                             ha='center', va='center', color=text_color, fontsize=10)

        # Add colorbar
        fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    filename = f"{save_prefix}_ensemble_analysis.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"✅ Saved: {filename}")
    plt.show()

    
def create_bar_graph_with_labels(data_dict, save_prefix=None, suptitle_prefix=None):
    """
    Creates a bar graph from a dictionary and displays the float value 
    on top of each bar.

    Args:
        data_dict (dict): The input dictionary (e.g., {'A': 5.5, 'B': 10.2, 'C': 3.8}).
    """
    if not isinstance(data_dict, dict) or not data_dict:
        print("Error: Input must be a non-empty dictionary.")
        return

    # 1. Prepare data
    try:
        keys = list(data_dict.keys())
        values = [float(v) for v in data_dict.values()]
    except ValueError:
        print("Error: All dictionary values must be convertible to a float.")
        return

    # 2. Create the bar chart
    plt.figure(figsize=(10, 6))
    # We store the bar objects returned by plt.bar to use their properties later
    bars = plt.bar(keys, values, color='teal')

    # 3. Add labels (the core change)
    for bar in bars:
        # Get the height (value) of the bar
        height = bar.get_height()
        
        # Annotate the bar with the height value, formatted to one decimal place
        plt.text(
            bar.get_x() + bar.get_width() / 2.,  # x-position (center of the bar)
            height,                             # y-position (top of the bar)
            f'{height:.1f}',                    # The text to display (e.g., '10.2')
            ha='center',                        # Horizontal alignment (center the text)
            va='bottom'                         # Vertical alignment (place text slightly above the bar)
        )

    # 4. Add labels and title
    plt.xlabel("Transformation")
    plt.ylabel("Time (Seconds)")
    title_string = "Total Data Transformation Times"
    plt.title(suptitle_prefix+title_string)
    
    plt.xticks(rotation=45, ha='right') 
    plt.yscale('log')
    # Optional: Increase the y-axis limit slightly to give space for the labels
    plt.ylim(0, max(values) * 1.4) 
    # 5. Display the plot
    plt.tight_layout()
    filename = f"{save_prefix}_transformation_time_totals.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"✅ Saved: {filename}")
    plt.show()

def get_x_val(test_metadata):
    #'RacketSports':{'train':151, 'test':152, 'length':30, 'channel':6}, 
    return_val = 1
    return_val = return_val*(test_metadata['train']+test_metadata['test'])
    return_val = return_val*(test_metadata['channel'])
    return_val = return_val*(test_metadata['length'])#*int(np.emath.logn(4, test_metadata['length'])))
    return return_val

from sklearn.metrics import cohen_kappa_score
from aeon.datasets import load_classification, load_regression

dir_list = os.listdir()  
print(dir_list)
all_test_accuracies = {}
all_test_times = {}
all_test_preds = {}

id_string = 'sprocket_euclid_big_compare'
suptitle_prefix = 'Euclidean Sprocket Other Algorithms Comparison '

file_name_list = ['test_batch_1', 'test_batch_0', 'test_batch_2', 'test_batch_3', 'test_batch_4']

#Ensemble Stats
all_corcoeffs = []
all_Q_stats = []
all_disagreements = []
all_double_faults = []

test_accuracies_to_aggregate = {}
test_times_to_aggregate = {}
test_prep_times_to_aggregate = {}

for target_file_name in file_name_list:
    print(target_file_name)
    for filename in dir_list:
        if target_file_name in filename:
            with open(filename, 'r') as file:
                data = file.read()
            file.close()
            #print(data)
            if 'time' or 'acc' in filename:
                data = eval(data)
            
            for test_set in data.keys():
                #data = eval(data)
                if "time" in filename:
                    all_test_times[test_set] = data[test_set]
                if 'acc' in filename:
                    all_test_accuracies[test_set] = data[test_set]
                if 'preds' in filename:
                    all_test_preds[test_set] = data[test_set]

    for test_set in all_test_preds.keys():
        for transform_type in all_test_preds[test_set].keys():
            first_element = all_test_preds[test_set][transform_type][0]

            if isinstance(first_element, int):
                # Already an int, convert remaining elements to int for consistency
                all_test_preds[test_set][transform_type] = [int(x) for x in all_test_preds[test_set][transform_type]]

            elif isinstance(first_element, str):
                # Only proceed with string methods if it's actually a string
                if first_element.endswith('.0'):
                    # Convert '1.0' -> 1
                    all_test_preds[test_set][transform_type] = [int(float(x)) for x in all_test_preds[test_set][transform_type]]
                
                else:
                    # Fallback for other strings (e.g., 'A', 'B', 'C')
                    all_test_preds[test_set][transform_type] = LabelEncoder().fit_transform(all_test_preds[test_set][transform_type])
        
            else:
                # Final fallback if the data is not int or str (e.g., float, bool, etc.)
                all_test_preds[test_set][transform_type] = [int(x) for x in all_test_preds[test_set][transform_type]]
                
    for test_set in all_test_preds.keys():
        if test_set not in test_accuracies_to_aggregate.keys():
            test_accuracies_to_aggregate[test_set] = {}
            for transform_type in all_test_preds[test_set].keys():
                if transform_type not in test_accuracies_to_aggregate[test_set].keys():
                    test_accuracies_to_aggregate[test_set][transform_type] = []
        for test_set in all_test_times.keys():
            if test_set not in test_times_to_aggregate.keys():
                test_times_to_aggregate[test_set] = {}
                test_prep_times_to_aggregate[test_set] = {}
                for transform_type in all_test_times[test_set].keys():
                    if transform_type not in test_times_to_aggregate[test_set].keys():
                        test_times_to_aggregate[test_set][transform_type] = []
                        test_prep_times_to_aggregate[test_set][transform_type] = []
    all_corrcoef = {}

    for test_set in all_test_preds.keys():
        pred_matrix = []
        for transform_type in all_test_preds[test_set].keys():
            pred_matrix.append(all_test_preds[test_set][transform_type])
        corrcoef_matrix = np.corrcoef(pred_matrix)
        for i in range(0, len(corrcoef_matrix)):
            for j in range(0, len(corrcoef_matrix)):
                #print(corrcoef_matrix[i][j])
                if math.isnan(corrcoef_matrix[i][j]):
                    corrcoef_matrix[i][j] = 0.0
        all_corrcoef[test_set] = corrcoef_matrix
        '''if test_set == 'MindReading':
            exit()'''


    all_q_matrices = {}
    all_disagreement_matrices = {}
    all_double_fault_matrices = {}
    len_pred_list = 0
    label_list = []
    for test_set in all_test_preds.keys():
        pred_list = []
        label_list = []
        _, test_set_truths = load_classification(test_set, split='test')
        if len(test_set_truths) > 0 and isinstance(test_set_truths[0], str):
            test_set_truths = LabelEncoder().fit_transform(test_set_truths)    
        else:
            test_set_truths = [int(x) for x in test_set_truths]
        for transform_type in all_test_preds[test_set].keys():
            '''if transform_type == 'quant_rf':
                print(all_test_preds[test_set][transform_type])
                exit()'''
            pred_list.append(all_test_preds[test_set][transform_type])
            #label_list.append(transform_type.split("_")[-2])
            new_label = ''
            for i in range(0, len(transform_type.split("_"))-1):
                new_label += transform_type.split("_")[i]
                if i != len(transform_type.split("_"))-2:
                    new_label += "-"
            label_list.append(new_label)
        len_pred_list = len(pred_list)
        q_matrix = np.zeros((len(pred_matrix), len(pred_matrix)))
        disagreement_matrix = np.zeros((len(pred_matrix), len(pred_matrix)))
        double_fault_matrix = np.zeros((len(pred_matrix), len(pred_matrix)))
        for i in range(0, len(pred_list)):
            for j in range(0, len(pred_list)):
                if i == j:
                    q_matrix[i][j] = 1
                else:
                    q_matrix[i][j], disagreement_matrix[i][j], double_fault_matrix[i][j] = disagreement_stats(pred_list[i], pred_list[j], test_set_truths)
        all_q_matrices[test_set] = q_matrix
        all_disagreement_matrices[test_set] = disagreement_matrix
        all_double_fault_matrices[test_set] = double_fault_matrix

    mean_corrcoeff = np.zeros((len_pred_list, len_pred_list))
    mean_q = disagreement_matrix = np.zeros((len_pred_list, len_pred_list))
    mean_disagreement = disagreement_matrix = np.zeros((len_pred_list, len_pred_list))
    mean_double_fault = disagreement_matrix = np.zeros((len_pred_list, len_pred_list))

    for test_set in all_test_preds.keys():
        for i in range(0, len_pred_list):
            for j in range(0, len_pred_list):
                mean_q[i][j] += all_q_matrices[test_set][i][j]
                mean_disagreement[i][j] += all_disagreement_matrices[test_set][i][j]
                mean_double_fault[i][j] += all_double_fault_matrices[test_set][i][j]
                mean_corrcoeff[i][j] += all_corrcoef[test_set][i][j]
                
    mean_corrcoeff = mean_corrcoeff*1/len(all_test_preds)
    mean_q = mean_q*1/len(all_test_preds)
    mean_disagreement = mean_disagreement*1/len(all_test_preds)
    mean_double_fault = mean_double_fault*1/len(all_test_preds)
    
    all_corcoeffs.append(mean_corrcoeff)
    all_Q_stats.append(mean_q)
    all_disagreements.append(mean_disagreement)
    all_double_faults.append(mean_double_fault)
                
    #print(mean_corrcoeff)

    matrices_for_visual = [mean_corrcoeff, mean_q, mean_disagreement, mean_double_fault]
    plot_titles = ['Correlations', 'Q Statistics', 'Disagreements', 'Double Faults']
    axes_labels = set()

    for key in all_test_preds.keys():
        for alg_type in all_test_preds[key].keys():
            axes_labels.add(alg_type)

    #visualize_four_matrices(matrices_for_visual, plot_titles, save_prefix=id_string, labels=label_list, suptitle=suptitle_prefix + "Ensemble Analysis Grids")
    all_classifiers = {}

    for test_set in all_test_accuracies.keys():
        for transform_type in all_test_accuracies[test_set].keys():
            for classifier_type in all_test_accuracies[test_set][transform_type].keys():
                all_classifiers[classifier_type] = []
           
    classifier_list = list(all_classifiers.keys())

    classifier_acc_compare_xy = {}
    for test_set in all_test_accuracies.keys():
        for transform_type in all_test_accuracies[test_set].keys():
            #print(all_test_accuracies[test_set][transform_type])
            for classifier_type in all_test_accuracies[test_set][transform_type].keys():
                #print(test_accuracies_to_aggregate[test_set])
                test_accuracies_to_aggregate[test_set][classifier_type].append(all_test_accuracies[test_set][transform_type][classifier_type]['zero_one'])
    for test_set in all_test_times.keys():
        for transform_type in all_test_times[test_set].keys():
            data_prepared_time = 0.0
            all_time = 0.0
            for subtime in all_test_times[test_set][transform_type].keys():
                all_time += all_test_times[test_set][transform_type][subtime]
                if subtime == 'data_prepared':
                    data_prepared_time = all_test_times[test_set][transform_type][subtime]
            test_times_to_aggregate[test_set][transform_type].append(all_time)
            test_prep_times_to_aggregate[test_set][transform_type].append(data_prepared_time)

    
    
agg_corcoeff = np.zeros((len(all_corcoeffs[0]), len(all_corcoeffs[0])), dtype=float)
agg_q_stats = np.zeros((len(all_corcoeffs[0]), len(all_corcoeffs[0])), dtype=float)
agg_disagreements = np.zeros((len(all_corcoeffs[0]), len(all_corcoeffs[0])), dtype=float)
agg_double_faults = np.zeros((len(all_corcoeffs[0]), len(all_corcoeffs[0])), dtype=float)
#print(all_corcoeffs)
for i in range(len(all_corcoeffs)):
    agg_corcoeff = np.add(agg_corcoeff, all_corcoeffs[i])
for i in range(len(all_corcoeffs)):
    agg_q_stats = np.add(agg_q_stats, all_Q_stats[i])
for i in range(len(all_corcoeffs)):
    agg_disagreements = np.add(agg_disagreements, all_disagreements[i])
for i in range(len(all_corcoeffs)):
    agg_double_faults = np.add(agg_double_faults, all_double_faults[i])
agg_corcoeff = agg_corcoeff*(1/len(all_corcoeffs))
agg_q_stats = agg_q_stats*(1/len(all_corcoeffs))
agg_disagreements = agg_disagreements*(1/len(all_corcoeffs))
agg_double_faults = agg_double_faults*(1/len(all_corcoeffs))
matrices_for_visual = [agg_corcoeff, agg_q_stats, agg_disagreements, agg_double_faults]
suptitle_prefix = 'Aggregate '
visualize_four_matrices(matrices_for_visual, plot_titles, save_prefix='agg_corr_stats', labels=label_list, suptitle=suptitle_prefix + "Euclidean SPROCKET Ensemble Analysis Grids")
all_ag_type_avgs = {}
all_alg_types = {}
for test_set in test_accuracies_to_aggregate:
    all_ag_type_avgs[test_set] = {}
    print(test_set)
    for alg_type in test_accuracies_to_aggregate[test_set]:
        if alg_type not in all_alg_types:
            all_alg_types[alg_type] = []
        print(test_accuracies_to_aggregate[test_set][alg_type])
        all_ag_type_avgs[test_set][alg_type] = sum(test_accuracies_to_aggregate[test_set][alg_type])/len(test_accuracies_to_aggregate[test_set][alg_type])
        
all_classifier_ranks = {}

for key in all_alg_types.keys():
    all_classifier_ranks[key] = []

for test_set in all_ag_type_avgs.keys():
    element_acc_pairs = []
    accuracies = []
    for transform_type in all_ag_type_avgs[test_set].keys():
        print(transform_type)
        element_acc_pairs.append((transform_type, 1-all_ag_type_avgs[test_set][transform_type]))
        accuracies.append(1-all_ag_type_avgs[test_set][transform_type])
    element_acc_pairs = np.array(element_acc_pairs)
    sorted_indices = rank_floats(accuracies)
    print(element_acc_pairs)
    print(sorted_indices)
    for i in range(0, len(element_acc_pairs)):
        all_classifier_ranks[element_acc_pairs[i][0]].append(sorted_indices[i]) 
print(all_classifier_ranks)

for_visualizer = {}
for key in all_classifier_ranks.keys():
    mean_rank = sum(all_classifier_ranks[key])/len(all_classifier_ranks[key])
    rank_dict = {}
    for i in range(0, len(all_classifier_ranks)):
        rank_dict[i] = all_classifier_ranks[key].count(i)
    new_label = ''
    for i in range(0, len(key.split("_"))-1):
        new_label += key.split("_")[i]
        if i != len(key.split("_"))-2:
            new_label += "-"
        #label_list.append(new_label)
    for_visualizer[new_label] = {'mean_rank': mean_rank, 'rank_counts':rank_dict}


suptitle='Euclidean SPROCKET CD and Best Performances'

visualize_ranking_results_with_cd(for_visualizer, save_path=suptitle_prefix+suptitle, num_datasets=len(all_ag_type_avgs.keys()), suptitle=suptitle_prefix+suptitle)

classifier_list = list(all_alg_types.keys())
classifier_acc_compare_xy = {}
for i in range(0, len(classifier_list)):
    classifier_acc_compare_xy[classifier_list[i]] = {}
    for j in range(i+1, len(classifier_list)):
        classifier_acc_compare_xy[classifier_list[i]][classifier_list[j]] = [[],[]]
        for test_set in all_ag_type_avgs.keys():
            x_val = 0
            y_val = 0
            for transform_type in all_ag_type_avgs[test_set].keys():
                if transform_type == classifier_list[i]:
                    x_val = all_ag_type_avgs[test_set][transform_type]
                if transform_type == classifier_list[j]:
                    y_val = all_ag_type_avgs[test_set][transform_type]
            classifier_acc_compare_xy[classifier_list[i]][classifier_list[j]][0].append(x_val)
            classifier_acc_compare_xy[classifier_list[i]][classifier_list[j]][1].append(y_val)
x_labels = []
y_labels = []
titles = []
x_y_pairs = []
for x_key in classifier_acc_compare_xy.keys():
    for y_key in classifier_acc_compare_xy[x_key].keys():
        new_x_label = ''
        print(x_key, y_key)
        for i in range(0, len(x_key.split("_"))-1):
            new_x_label += x_key.split("_")[i]
            if i != len(x_key.split("_"))-2:
                new_x_label += "-"
        x_label = new_x_label
        new_y_label = ''
        for i in range(0, len(y_key.split("_"))-1):
            new_y_label += y_key.split("_")[i]
            if i != len(y_key.split("_"))-2:
                new_y_label += "-"
        y_label = new_y_label
        print(x_label, y_label)
        x_labels.append(x_label)
        y_labels.append(y_label)
        titles.append(x_label + "/" + y_label)
        x_y_pairs.append(classifier_acc_compare_xy[x_key][y_key])
'''for i in range(0, len(x_labels)):
    print(x_labels[i], y_labels[i], x_y_pairs[i])
    print(x_key, classifier_acc_compare_xy[x_key].keys())'''
    
scatter_id = id_string+'_scatter_grid_accs'
scatter_title = "Agg Acc Comparisons"
scatter_grid(x_y_pairs, titles, save_prefix=scatter_id, fig_title=suptitle_prefix+scatter_title)

all_ag_type_time_avgs = {}
all_alg_types = {}
for test_set in test_times_to_aggregate:
    all_ag_type_time_avgs[test_set] = {}
    print(test_set)
    for alg_type in test_times_to_aggregate[test_set]:
        if alg_type not in all_alg_types:
            all_ag_type_time_avgs[alg_type] = []
        print(test_times_to_aggregate[test_set][alg_type])
        all_ag_type_time_avgs[test_set][alg_type] = sum(test_times_to_aggregate[test_set][alg_type])/len(test_times_to_aggregate[test_set][alg_type])
        
all_ag_type_prep_avgs = {}
all_alg_types = {}
for test_set in test_prep_times_to_aggregate:
    all_ag_type_prep_avgs[test_set] = {}
    print(test_set)
    for alg_type in test_prep_times_to_aggregate[test_set]:
        if alg_type not in all_alg_types:
            all_ag_type_prep_avgs[alg_type] = []
        print(test_prep_times_to_aggregate[test_set][alg_type])
        all_ag_type_prep_avgs[test_set][alg_type] = sum(test_prep_times_to_aggregate[test_set][alg_type])/len(test_prep_times_to_aggregate[test_set][alg_type])
        
#add in prep times for ensembles
for test_set in all_ag_type_time_avgs:
    for alg_type in all_ag_type_time_avgs[test_set]:
        print(alg_type)
        if '_' in alg_type:
            if "mr" in alg_type:
                all_ag_type_time_avgs[test_set][alg_type] += all_ag_type_prep_avgs[test_set]['multirocket']
            if "hy" in alg_type:
                all_ag_type_time_avgs[test_set][alg_type] += all_ag_type_prep_avgs[test_set]['hydra']
            if "sp" in alg_type:
                all_ag_type_time_avgs[test_set][alg_type] += all_ag_type_prep_avgs[test_set]['sprocket']
            print(alg_type)
for test_set in all_ag_type_time_avgs:
    print(test_set, all_ag_type_time_avgs[test_set])
    
all_wall_clock_summable = {}    

for test_set in all_ag_type_time_avgs:
    for alg_type in all_ag_type_time_avgs[test_set]:
        if alg_type not in all_wall_clock_summable.keys():
            all_wall_clock_summable[alg_type] = []
        all_wall_clock_summable[alg_type].append(all_ag_type_time_avgs[test_set][alg_type])

for key in all_wall_clock_summable:
    print(key, sum(all_wall_clock_summable[key]))
    

classification_metadata = {'Coffee':{'train':28, 'test':28, 'length':286, 'channel':1},
                         'DuckDuckGeese':{'train':60, 'test':40, 'length':270, 'channel':1345}, 
                         'PEMS-SF':{'train':267, 'test':173, 'length':144, 'channel':963},
                         'MindReading':{'train':727, 'test':653, 'length':200, 'channel':204},
                         'PhonemeSpectra':{'train':3315, 'test':3353, 'length':217, 'channel':11},
                         'ShapeletSim':{'train':20, 'test':180, 'length':500, 'channel':1}, 
                         'EMOPain':{'train':1093, 'test':50, 'length':180, 'channel':30},
                         'SmoothSubspace':{'train':150, 'test':150, 'length':15, 'channel':1},
                         'MelbournePedestrian':{'train':1194, 'test':2439, 'length':24, 'channel':1}, 
                         'ItalyPowerDemand':{'train':67, 'test':1029, 'length':24, 'channel':1},
                         'Chinatown':{'train':20, 'test':345, 'length':24, 'channel':1},
                         'JapaneseVowels':{'train':270, 'test':370, 'length':29, 'channel':12},
                         'RacketSports':{'train':151, 'test':152, 'length':30, 'channel':6}, 
                         'LSST':{'train':2459, 'test':2466, 'length':36, 'channel':6}, 
                         'Libras':{'train':180, 'test':180, 'length':45, 'channel':2},
                         'FingerMovements':{'train':316, 'test':100, 'length':50, 'channel':28},
                         'NATOPS':{'train':180, 'test':180, 'length':51, 'channel':24},
                         'SharePriceIncrease':{'train':965, 'test':965, 'length':60, 'channel':1},
                         'SyntheticControl':{'train':300, 'test':300, 'length':60, 'channel':1},
                         'SonyAIBORobotSurface2':{'train':27, 'test':953, 'length':65, 'channel':1},
                         'ERing':{'train':30, 'test':270, 'length':65, 'channel':4},
                         'SonyAIBORobotSurface1':{'train':20, 'test':601, 'length':70, 'channel':1},
                         'PhalangesOutlinesCorrect':{'train':1800, 'test':858, 'length':80, 'channel':1},
                         'ProximalPhalanxOutlineCorrect':{'train':600, 'test':291, 'length':80, 'channel':1},
                         'MiddlePhalanxOutlineCorrect':{'train':600, 'test':291, 'length':80, 'channel':1},
                         'DistalPhalanxOutlineCorrect':{'train':600, 'test':291, 'length':80, 'channel':1},
                         'ProximalPhalanxTW':{'train':399, 'test':154, 'length':80, 'channel':1},
                         'ProximalPhalanxOutlineAgeGroup':{'train':400, 'test':154, 'length':80, 'channel':1},
                         'MiddlePhalanxOutlineAgeGroup':{'train':400, 'test':154, 'length':80, 'channel':1},
                         'MiddlePhalanxTW':{'train':399, 'test':154, 'length':80, 'channel':1},
                         'DistalPhalanxTW':{'train':399, 'test':154, 'length':80, 'channel':1},
                         'DistalPhalanxOutlineAgeGroup':{'train':400, 'test':154, 'length':80, 'channel':1},
                         'TwoLeadECG':{'train':23, 'test':1139, 'length':82, 'channel':1},
                         'MoteStrain':{'train':20, 'test':1252, 'length':84, 'channel':1},
                         'ECG200':{'train':100, 'test':100, 'length':96, 'channel':1}, 
                         'MedicalImages':{'train':381, 'test':760, 'length':99, 'channel':1},
                         'BasicMotions':{'train':40, 'test':40, 'length':100, 'channel':6},
                         'TwoPatterns':{'train':1000, 'test':4000, 'length':128, 'channel':1},
                         'CBF':{'train':30, 'test':900, 'length':128, 'channel':1},
                         'SwedishLeaf':{'train':500, 'test':625, 'length':128, 'channel':1},
                         'BME':{'train':30, 'test':150, 'length':128, 'channel':1},
                         'EyesOpenShut':{'train':56, 'test':42, 'length':128, 'channel':14},
                         'FacesUCR':{'train':200, 'test':2050, 'length':131, 'channel':1},
                         'FaceAll':{'train':560, 'test':1690, 'length':131, 'channel':1},
                         'ECGFiveDays':{'train':23, 'test':861, 'length':136, 'channel':1},
                         'ECG5000':{'train':500, 'test':4500, 'length':140, 'channel':1},
                         'ArticularyWordRecognition':{'train':275, 'test':300, 'length':144, 'channel':9}, 
                         'PowerCons':{'train':180, 'test':180, 'length':144, 'channel':1},
                         'Plane':{'train':105, 'test':105, 'length':144, 'channel':1},
                         'GunPointOldVersusYoung':{'train':135, 'test':316, 'length':150, 'channel':1},
                         'GunPointMaleVersusFemale':{'train':135, 'test':316, 'length':150, 'channel':1},
                         'GunPointAgeSpan':{'train':135, 'test':316, 'length':150, 'channel':1},
                         'GunPoint':{'train':50, 'test':150, 'length':150, 'channel':1},
                         'UMD':{'train':36, 'test':144, 'length':150, 'channel':1},
                         'Wafer':{'train':1000, 'test':6164, 'length':152, 'channel':1}, 
                         'Handwriting':{'train':150, 'test':850, 'length':152, 'channel':3},
                         'ChlorineConcentration':{'train':467, 'test':3840, 'length':166, 'channel':1},
                         'Adiac':{'train':390, 'test':391, 'length':176, 'channel':1},
                         'Epilepsy2':{'train':80, 'test':11420, 'length':178, 'channel':1},
                         'Colposcopy':{'train':100, 'test':100, 'length':180, 'channel':1},
                         'Fungi':{'train':18, 'test':186, 'length':201, 'channel':1},
                         'Epilepsy':{'train':137, 'test':138, 'length':207, 'channel':1},
                         'Wine':{'train':57, 'test':54, 'length':234, 'channel':1},
                         'Strawberry':{'train':613, 'test':370, 'length':235, 'channel':1},
                         'ArrowHead':{'train':36, 'test':175, 'length':251, 'channel':1},
                         'ElectricDeviceDetection':{'train':623, 'test':3767, 'length':256, 'channel':1},
                         'WordSynonyms':{'train':267, 'test':638, 'length':270, 'channel':1},
                         'FiftyWords':{'train':450, 'test':455, 'length':270, 'channel':1},
                         'Trace':{'train':100, 'test':100, 'length':275, 'channel':1},
                         'ToeSegmentation1':{'train':40, 'test':228, 'length':277, 'channel':1},
                         'DodgerLoopWeekend':{'train':78, 'test':80, 'length':288, 'channel':1},
                         'DodgerLoopGame':{'train':20, 'test':138, 'length':288, 'channel':1},
                         'DodgerLoopDay':{'train':20, 'test':138, 'length':288, 'channel':1},
                         'CricketZ':{'train':390, 'test':390, 'length':300, 'channel':1}, 
                         'CricketY':{'train':390, 'test':390, 'length':300, 'channel':1}, 
                         'CricketX':{'train':390, 'test':390, 'length':300, 'channel':1},
                         'FreezerRegularTrain':{'train':150, 'test':2850, 'length':301, 'channel':1},
                         'FreezerSmallTrain':{'train':28, 'test':2850, 'length':301, 'channel':1},
                         'UWaveGestureLibraryZ':{'train':896, 'test':3582, 'length':315, 'channel':1},
                         'UWaveGestureLibraryY':{'train':896, 'test':3582, 'length':315, 'channel':1},
                         'UWaveGestureLibraryX':{'train':896, 'test':3582, 'length':315, 'channel':1}, 
                         'UWaveGestureLibrary':{'train':2238, 'test':3582, 'length':315, 'channel':3},
                         'Lightning7':{'train':70, 'test':73, 'length':319, 'channel':1},
                         'ToeSegmentation2':{'train':36, 'test':130, 'length':343, 'channel':1},
                         'DiatomSizeReduction':{'train':16, 'test':306, 'length':345, 'channel':1},
                         'FaceFour':{'train':24, 'test':88, 'length':350, 'channel':1},
                         'GestureMidAirD3':{'train':208, 'test':130, 'length':360, 'channel':1},
                         'GestureMidAirD2':{'train':208, 'test':130, 'length':360, 'channel':1},
                         'GestureMidAirD1':{'train':208, 'test':130, 'length':360, 'channel':1},
                         'Symbols':{'train':25, 'test':995, 'length':398, 'channel':1},
                         'HandMovementDirection':{'train':160, 'test':74, 'length':400, 'channel':10},
                         'Heartbeat':{'train':204, 'test':205, 'length':405, 'channel':61},
                          'Yoga':{'train':300, 'test':3000, 'length':426, 'channel':1}, 
                         'OSULeaf':{'train':200, 'test':242, 'length':427, 'channel':1},
                         'Meat':{'train':60, 'test':60, 'length':448, 'channel':1},
                         'Fish':{'train':175, 'test':175, 'length':463, 'channel':1},
                         'FordA':{'train':3601, 'test':1320, 'length':500, 'channel':1}, 
                         'FordB':{'train':3636, 'test':810, 'length':500, 'channel':1},
                         'Ham':{'train':109, 'test':105, 'length':431, 'channel':1}
                         }
graph_x_axis = []
graph_y_axis = []
for test_set in all_ag_type_prep_avgs:
    if test_set in ['sprocket', 'multirocket', 'hydra', 'quant', 'mr_hy', 'mr_sp', 'hy_sp', 'mr_hy_sp']:
        continue
    #print(all_ag_type_prep_avgs.keys())
    #print('test_set', test_set)
    test_metadata = classification_metadata[test_set]
    test_x_val = get_x_val(test_metadata)
    graph_x_axis.append(test_x_val)
    graph_y_axis.append(all_ag_type_prep_avgs[test_set]['sprocket'])
    
import matplotlib.pyplot as plt
import numpy as np

# Create the plot
plt.figure(figsize=(10, 8))

# Plot with log scales on both axes
plt.scatter(graph_x_axis, graph_y_axis, 
           marker='o',          # shows data points
           color='blue',
           label='Time Comparisons Vs Predicted')

plt.yscale('log')
plt.xscale('log')
# Customize the plot
plt.xlabel('Unitless Time Predicted = (|Train|+|Test|)*Channels*Length')
plt.ylabel('Time Observed (Seconds)')
plt.title('Predicted VS Observed Euclidean Sprocket Transform Times')
plt.grid(True, which="both", ls="--", alpha=0.5)  # grid helpful on log plots

# Make sure all data is visible
plt.tight_layout()

plt.show()
exit()

all_data_prep_times = {}
data_prep_time_defeats = []

for key in all_classifiers.keys():
    if key in data_prep_time_defeats:
        continue
    all_data_prep_times[key] = []

time_defeats = []

for test_set in all_test_times.keys():
    for transform_type in all_test_times[test_set].keys():
        #print(all_test_times[test_set])
        all_data_prep_times[transform_type].append(all_test_times[test_set][transform_type]['data_prepared'])

for_visualizer = {}
for key in all_data_prep_times.keys():
    for_visualizer[key] = sum(all_data_prep_times[key])

create_bar_graph_with_labels(for_visualizer, save_prefix=id_string, suptitle_prefix=suptitle_prefix)


