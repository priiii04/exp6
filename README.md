
Experiment 6 — Principal Component Analysis (PCA) on Iris


AIM:

To reduce 4D Iris data to 2 principal components and visualize species separation.

ALGORITHM:

1. Start and import PCA, load_iris, matplotlib.pyplot, and numpy.
2. Load the dataset using load_iris() and extract feature matrix X (4D) and target labels y.
3. Initialize PCA: Create a PCA(n_components=2) model to reduce dimensionality to 2 components via Singular Value Decomposition (SVD).
4. Fit and transform: Apply pca.fit_transform(X) to center the data and project it onto the top two principal components (Z).
5. Inspect variance: Print pca.explained_variance_ratio_ rounded to 4 decimal places to see the variance captured by each component.
6. Visualize: Plot a 2D scatter plot of PC1 (Z[:, 0]) vs PC2 (Z[:, 1]), colored by species (y), add axis labels and title, and display the plot.
7. Stop.

THEORY:

Principal Component Analysis (PCA) is an unsupervised linear dimensionality reduction technique that transforms high-dimensional correlated features into a smaller set of uncorrelated variables called principal components. Mathematically, PCA finds orthogonal directions of maximal variance in the data—which correspond to the eigenvectors of the covariance matrix (computed efficiently via SVD)—allowing high-dimensional data to be projected into 2D or 3D space while retaining most of the original information.

PROGRAM:

python
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np

iris = load_iris()
X = iris.data
y = iris.target

pca = PCA(n_components=2)
Z = pca.fit_transform(X)

print('Explained variance ratio:', np.round(pca.explained_variance_ratio_, 4))

plt.scatter(Z[:, 0], Z[:, 1], c=y, cmap='viridis')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA of Iris')
plt.show()
Use code with caution.

DETAILED EXPLANATION:

1. Load Iris dataset: Loads the 150-sample Iris dataset with 4 numeric features (X) and 3 species classes (y).
2. Initialize PCA: PCA(n_components=2) configures the transformer to retain the top two orthogonal principal components.
3. Center and project: fit_transform(X) centers each feature around zero and projects the 4D data onto the 2D principal component subspace (Z).
4. Explained variance: explained_variance_ratio_ reports the proportion of total dataset variance captured by PC1 (92.46%) and PC2 (5.31%).
5. Plot projection: A scatter plot maps PC1 against PC2 and colors points by species (c=y), visually showing clear clustering and separation among species.

SAMPLE OUTPUT:

text
Explained variance ratio: [0.9246 0.0531]
(Scatter plot shows separation among species)
Use code with caution.

RESULT:

Thus, 4D Iris data was reduced to 2 principal components using PCA, demonstrating that PC1 captures the vast majority of variance and effectively separates species for visualization and dimensionality reduction.
