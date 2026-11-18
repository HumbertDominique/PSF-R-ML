import numpy as np
import matplotlib.pyplot as plt

test_means = np.array([0.944,0.944,0.944,0.944,0.944, 0.943, 0.943])
test_std_dev = np.array([0.0173,0.0173,0.0173,0.0173,0.0173, 0.0145, 0.0145])
means = np.array([0.845, 0.845, 0.845, 0.845, 0.845, 0.844, 0.844])
std_dev = np.array([0.1203, 0.0417, 0.0623, 0.0333, 0.0125, 0.0114, 0.089])


x = np.arange(len(means))

plt.figure(figsize=(6, 5))

plt.errorbar(
    x,
    means,
    yerr=2*std_dev,
    fmt="o",
    capsize=5,
    color="#000000",
    ecolor="#000000",
    label= "Prediction mean ± 2std"
)
plt.scatter(
    x, means,
    color="black", s=35,
    zorder=3, label="Prediction mean"
)

plt.errorbar(
    x, test_means, yerr=2*test_std_dev,
    fmt="none", capsize=5,
    ecolor="#646464",
    label="Test mean ± 2std"
)
plt.scatter(
    x, test_means,
    color="#646464", s=35,
    zorder=3, label="Test mean"
)


plt.xticks(x, [f"Inputs {i + 1}" for i in x])
plt.ylabel("Score")
plt.title(r"Mean ± 2 standard deviation")
plt.grid(axis="y", alpha=0.3)
plt.ylim([0, 1.1])
plt.legend()
plt.tight_layout()
# plt.show()


test_means = np.array([0.592,0.592,0.592,0.592,0.592, 0.590, 0.590])
test_std_dev = np.array([0.1095,0.1095,0.1095,0.1095,0.1095, 0.1059, 0.1059])
means = np.array([0.698, 0.698, 0.698, 0.698, 0.697, 0.696, 0.696])
std_dev = np.array([0.0856, 0.0734, 0.0695, 0.0644, 0.0530, 0.0497, 0.0470])


x = np.arange(len(means))

plt.figure(figsize=(6, 5))

plt.errorbar(
    x,
    means,
    yerr=2*std_dev,
    fmt="o",
    capsize=5,
    color="#000000",
    ecolor="#000000",
    label= "Prediction mean ± 2std"
)
plt.scatter(
    x, means,
    color="black", s=35,
    zorder=3, label="Prediction mean"
)

plt.errorbar(
    x, test_means, yerr=2*test_std_dev,
    fmt="none", capsize=5,
    ecolor="#646464",
    label="Test mean ± 2std"
)
plt.scatter(
    x, test_means,
    color="#646464", s=35,
    zorder=3, label="Test mean"
)


plt.xticks(x, [f"Inputs {i + 1}" for i in x])
plt.ylabel("Score")
plt.title(r"Mean ± 2 standard deviation")
plt.grid(axis="y", alpha=0.3)
plt.ylim([0, 1.1])
plt.legend()
plt.tight_layout()
plt.show()