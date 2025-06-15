import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def generate_and_plot_datasets():
    plt.figure(figsize=(15, 5))

    configs = [
        {
            "weights": [1/6, 5/6],
            "flip_y": 0.0,
            "title": "Dataset 1: proporcja 1:5"
        },
        {
            "weights": [1/100, 99/100],
            "flip_y": 0.0,
            "title": "Dataset 2: proporcja 1:99"
        },
        {
            "weights": [1/10, 9/10],     #
            "flip_y": 0.05,
            "title": "Dataset 3: proporcja 1:9 + 5% szumu"
        }
    ]

    for i, cfg in enumerate(configs):
        X, y = make_classification(
            n_samples=5000,
            n_features=8,
            n_informative=2,
            n_redundant=2,
            n_repeated=0,
            n_clusters_per_class=2,
            weights=cfg["weights"],
            flip_y=cfg["flip_y"],
            class_sep=0.8,
            random_state=42 + i
        )


        X_scaled = StandardScaler().fit_transform(X)


        X_pca = PCA(n_components=2).fit_transform(X_scaled)


        plt.subplot(1, 3, i + 1)
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, alpha=0.3, s=5)
        plt.title(cfg["title"])
        plt.grid(True)
        plt.xlim(-8, 8)
        plt.ylim(-6, 6)


    plt.show()


generate_and_plot_datasets()
