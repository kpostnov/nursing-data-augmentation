import pandas as pd
from matplotlib import pyplot as plt


def plot_distribution_pie_chart(distribution: "dict[str, int]", display_n: int = 22):
    activity_dict = dict(sorted(distribution.items(), key=lambda item: item[1], reverse=True))
    # Convert to minutes & round to 2 decimal places
    steps_per_minute = 60 * 60
    activity_dict = {k: round(v / steps_per_minute, 2) for k, v in activity_dict.items()}

    data = list(activity_dict.values())
    labels = list(activity_dict.keys())

    # Take only the first n important labels
    others = sum(data[display_n:])
    data = data[:display_n]
    labels = labels[:display_n]
    if others > 0:
        data.append(others)
        labels.append('others')

    plt.rcParams.update({'font.size': 25})
    cmap = plt.get_cmap('jet')
    colors = [cmap(1. * (i+1) / len(labels)) for i in range(len(labels))]

    plt.figure(figsize=(18, 14))
    plt.pie(data, colors=colors, labels=labels, startangle=180, labeldistance=1.05, pctdistance=0.92, 
        autopct='%1.1f%%', wedgeprops={"edgecolor":"1", 'linewidth': 1, 'linestyle': 'solid', 'antialiased': True})
    plt.suptitle("People Distribution")
    plt.title(f"{len(activity_dict)} people – {round(sum(data))} minutes")
    plt.axis('equal')
    plt.savefig("distribution_pie_chart.png")


def plot_distribution_bar_chart(distribution: "dict[str, int]"):
    activity_dict = dict(sorted(distribution.items(), key=lambda item: item[1], reverse=True))
    # Convert to minutes & round to 2 decimal places
    steps_per_minute = 60 * 60
    activity_dict = {k: round(v / steps_per_minute, 2) for k, v in activity_dict.items()}

    plt.rcParams.update({'font.size': 22})
    plt.grid(linewidth=0.5, axis='y', alpha=0.8, zorder=0)
    plt.figure(figsize=(18, 12))
    pd.Series(activity_dict).plot(kind="bar", zorder=3, color='royalblue')
    
    # Remove borders
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)    
    ax.spines['left'].set_visible(False)

    #plt.suptitle("Activity Distribution")
    plt.ylabel("Total recording time in minutes")
    plt.xlabel("Subjects")
    plt.xticks(rotation=45, ha="right")
    plt.subplots_adjust(bottom=0.2)
    plt.savefig("distribution_bar_chart.png")
