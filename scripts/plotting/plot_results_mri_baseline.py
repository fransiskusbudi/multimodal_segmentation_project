# dice_boxplot_excel.py
# Creates a grouped box plot of organ Dice scores by experiment from Excel.
# Shortens experiment names to only last two parts (e.g., ct_n100, mri_n100).

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


xlsx_file = os.path.join("..", "results", "mri_baseline.xlsx")
# Read the Excel file
df = pd.read_excel(xlsx_file)

def shorten_experiment(name: str) -> str:
    parts = str(name).split("_")
    if name.startswith("baseline_mri_on_mri"):
        # For MRI on MRI experiments
        if "unoriented" in name:
            return "mri_unoriented"
        elif "oriented" in name:
            return "mri_oriented"
    elif name.startswith("baseline_mri_on_ct"):
        # For MRI on CT experiments
        if "unoriented" in name:
            return "ct_unoriented"
        elif "oriented" in name:
            return "ct_oriented"
    else:
        # Fallback: return original
        return name

df["experiment"] = df["experiment"].apply(shorten_experiment)

# Filter for specific experiments
target_experiments = ["mri_unoriented", "mri_oriented", "ct_unoriented", "ct_oriented"]
df = df[df["experiment"].isin(target_experiments)]

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
