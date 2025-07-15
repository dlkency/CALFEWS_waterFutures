# %%
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import datetime

def plot_water_purchase_costs(dataframes, filename_suffix, directory='Figure/index_insurance'):
    # Define colors for the bars
    # below_median_colors = ['white','#b2182b','#d6604d','#f4a582',  '#fddbc7']  # Light to dark red
    # above_median_colors = ['#d1e5f0', '#92c5de', '#4393c3', '#2166ac',]  # Light to dark blue  ]  # Ensure a color for each segment
    below_median_colors = ['white','#8c510a','#bf812d','#dfc27d',  '#f6e8c3']  # Light to dark red
    above_median_colors = ['#c7eae5', '#80cdc1', '#35978f', '#01665e',]  # Light to dark blue  ]  # Ensure a color for each segment
    fig, ax = plt.subplots(figsize=(12, 6))
    bar_width = 0.4

    for x_pos, (label, data) in enumerate(dataframes.items()):
        median_cost = data.median()
        mean_cost = data.mean()
        # print(f"Mean cost for {label}: ${mean_cost:.2f} M")

        percentiles = data.quantile([0.05, 0.20, 0.80, 0.95]).values
        p5, p20, p80, p95= sorted(percentiles)

        # Define starting and ending points
        min_value = data.min()
        max_value = data.max()

        # Calculate segment heights starting from 0
        segments = [
            min_value,  # Segment from 0 to min_value
            max(0, p5 - min_value),     
            # max(0, p10 - p1),
            max(0, p20 - p5),
            max(0, p80 - p20),
            max(0, p95 - p80),
            # max(0, p99 - p90),
            max(0, max_value - p95)
        ]

        # Split below and above median segments
        below_segments, above_segments = [], []
        bottom = 0  # Start from 0 for the initial segment
        for height in segments:
            if bottom + height <= median_cost:
                below_segments.append(height)
            elif bottom >= median_cost:
                above_segments.append(height)
            else:  # Crosses the median
                below_segments.append(median_cost - bottom)
                above_segments.append((bottom + height) - median_cost)

            if height > 0:
                bottom += height

        # Ensure both lists are the same length for color assignment
        max_colors = max(len(above_segments), len(below_segments))
        while len(above_segments) < max_colors:
            above_segments.append(0)
        while len(below_segments) < max_colors:
            below_segments.append(0)

        # Plot below-median segments
        bottom = 0  # Start again from 0 for plotting
        for idx, height in enumerate(below_segments):
            if height > 0:
                ax.bar(x_pos, height, bottom=bottom, color=below_median_colors[idx % len(below_median_colors)], width=bar_width, alpha=1)
                bottom += height

        # Plot above-median segments
        above_segments = sorted(above_segments, reverse=True)
        for idx, height in enumerate(above_segments):
            if height > 0:
                ax.bar(x_pos, height, bottom=bottom, color=above_median_colors[idx % len(above_median_colors)], width=bar_width, alpha=0.8)
                bottom += height

        # Draw Mean line and annotate
        ax.axhline(y=mean_cost, xmin=(x_pos - bar_width / 2 + 0.3) / len(dataframes),
                   xmax=(x_pos + bar_width / 2 + 0.7) / len(dataframes),
                   color='black', linestyle='--', linewidth=2)
        ax.text(x_pos, mean_cost + 0.02 * max_value, f'${mean_cost:.2f} M', color='black', ha='center', va='bottom', fontsize=14)

    legend_patches = [
        mpatches.Patch(color=above_median_colors[2], label='95th Percentile to Max'),
        mpatches.Patch(color=above_median_colors[1], label='80th to 95th Percentile'),
        # mpatches.Patch(color=above_median_colors[1], label='80th to 95th Percentile'),
        mpatches.Patch(color=above_median_colors[0], label='50th to 80th Percentile'),
        mpatches.Patch(color=below_median_colors[3], label='20th to 50th Percentile'),
        mpatches.Patch(color=below_median_colors[2], label='5th to 20th Percentile'),
        # mpatches.Patch(color=below_median_colors[2], label='1th to 5th Percentile'),
        mpatches.Patch(color=below_median_colors[1], label='Min to 5th Percentile'),
        mpatches.Patch(color=below_median_colors[0], label=' '),
    ]

    ax.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)

    # Set labels and grid
    ax.set_ylabel('District Revenue (Million $)', fontsize=14)
    ax.set_xticks(range(len(dataframes)))
    ax.set_xticklabels(dataframes.keys(), fontsize=14)
    plt.tight_layout()
    plt.yticks(fontsize=14)
    plt.grid(False)

    # Create a unique filename using the provided suffix
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{directory}/strategy_{timestamp}_{filename_suffix}.png"

    # Save the figure to a file
    plt.savefig(filename, dpi=300)
    plt.show()

    return filename  # Return the filename for further use if needed


