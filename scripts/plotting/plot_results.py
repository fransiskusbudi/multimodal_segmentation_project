# dice_boxplot_excel.py
# Creates a grouped box plot of organ Dice scores by experiment from Excel.
# Shortens experiment names to only last two parts (e.g., ct_n100, mri_n100).

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


xlsx_file = os.path.join("..", "results", "dann.xlsx")
# Read the Excel file
df = pd.read_excel(xlsx_file)

def shorten_experiment(name: str) -> str:
    parts = str(name).split("_")
    if name.startswith("dann"):
        # For dann experiments, look for ct_add pattern
        for i, part in enumerate(parts):
            if part == "ct" and i + 1 < len(parts) and parts[i + 1].startswith("add"):
                if parts[i + 1] == "add0":
                    # For add0, look for ns pattern
                    for j, ns_part in enumerate(parts):
                        if ns_part.startswith("n"):
                            return f"{part}_{ns_part}"
                    # If no ns found, return ct
                    return part
                else:
                    return f"{part}_{parts[i + 1]}"
        # If no ct_add pattern found, keep everything after 'dann'
        return "_".join(parts[1:])
    elif name.startswith("kd"):
        # For kd experiments, look for ct_n pattern
        for i, part in enumerate(parts):
            if part == "ct" and i + 1 < len(parts) and parts[i + 1].startswith("n"):
                return f"{part}_{parts[i + 1]}"
        # If no ct_n pattern found, keep everything after 'kd'
        return "_".join(parts[1:])
    elif name.startswith("finetune"):
        # Keep last two parts
        return "_".join(parts[-2:])
    elif "baseline" in parts:
        # Remove 'baseline' from the name
        parts = [p for p in parts if p != "baseline"]
        return "_".join(parts)
    else:
        # Fallback: return original
        return name

df["experiment"] = df["experiment"].apply(shorten_experiment)

df = df[df["experiment"].str.contains("ct", case=False)]

# Reshape to long format: one row per (experiment, filename, organ)
melted = df.melt(
    id_vars=["experiment", "filename"],
    value_vars=["dice_spleen", "dice_liver", "dice_kidneys"],
    var_name="organ",
    value_name="dice",
)

# Clean organ names for nicer labels
melted["organ"] = (
    melted["organ"]
    .str.replace("dice_", "", regex=False)
    .str.capitalize()
)

# Set colorblind-friendly palette
plt.style.use('default')
sns.set_palette("colorblind")  # Colorblind-friendly palette

# Set larger font sizes
plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'legend.fontsize': 14,
    'figure.titlesize': 20
})

# Plot
plt.figure(figsize=(10, 6))
ax = sns.boxplot(
    data=melted,
    x="experiment",
    y="dice",
    hue="organ",
    showmeans=True,
    meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black"},
)

# ax.set_title("Organ Dice Score by Experiment")
ax.set_xlabel("Experiment", fontsize=16, fontweight='bold')
ax.set_ylabel("Dice Score", fontsize=16, fontweight='bold')
ax.set_ylim(0, 1)  # Dice is in [0, 1]



# Place legend outside to the right with larger text
plt.legend(title="Organ", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0., 
          title_fontsize=16, fontsize=14)

fig_filename = f"dice_boxplot_{os.path.splitext(os.path.basename(xlsx_file))[0]}.png"

plt.tight_layout()
plt.savefig(fig_filename, dpi=300, bbox_inches="tight")
plt.show()
