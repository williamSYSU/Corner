from sklearn.cluster import KMeans
import numpy as np

final_embedding = np.array(np.load("../embed/Vector_word_embedding_all.npy"))
k_means = KMeans(n_clusters=24)
k_means.fit(final_embedding)
k_means_cluster_centers = k_means.cluster_centers_
np.save("../initAspect.npy", k_means_cluster_centers)

print(k_means_cluster_centers.shape)
print(k_means_cluster_centers)

