import cv2
import numpy as np
import matplotlib.pyplot as plt




def initialize_centroids(pixels, k):
    # Khởi tạo các tâm cụm ngẫu nhiên từ dữ liệu
    np.random.seed(42)
    random_indices = np.random.choice(len(pixels), size=k, replace=False)
    centroids = pixels[random_indices]
    return centroids


def assign_clusters(pixels, centroids):
    # Tính toàn bộ khoảng cách giữa các pixel và tất cả các tâm cụm cùng lúc
    distances = np.linalg.norm(pixels[:, np.newaxis] - centroids, axis=2)
    # Gán mỗi pixel vào cụm gần nhất
    clusters = np.argmin(distances, axis=1)
    return clusters


def update_centroids(pixels, clusters, k):
    # Cập nhật lại các tâm cụm
    new_centroids = np.array(
        [pixels[clusters == i].mean(axis=0) if len(pixels[clusters == i]) > 0 else pixels[np.random.choice(len(pixels))]
         for i in range(k)])
    return new_centroids


def kmeans(pixels, k, max_iters=100, tolerance=1e-4):
    centroids = initialize_centroids(pixels, k)
    for i in range(max_iters):
        clusters = assign_clusters(pixels, centroids)
        new_centroids = update_centroids(pixels, clusters, k)
        # Kiểm tra hội tụ
        if np.all(np.abs(new_centroids - centroids) < tolerance):
            break
        centroids = new_centroids
    return clusters, centroids


# Đọc ảnh và chuyển đổi không gian màu
image = cv2.imread('meo1.jpg')
image = cv2.resize(image, (512, 512))  # Giảm kích thước ảnh xuống 512x512
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Chuyển sang không gian RGB
pixel_values = image.reshape((-1, 3))
pixel_values = np.float32(pixel_values)

# Danh sách các giá trị k
k_values = [2, 3, 4, 5]

# Tạo lưới hiển thị với số ô là số giá trị k cộng thêm một ô cho ảnh gốc
plt.figure(figsize=(15, 10))

# Hiển thị ảnh gốc
plt.subplot(2, 3, 1)
plt.imshow(image)
plt.title('Ảnh gốc')
plt.axis('off')

# Hiển thị các kết quả phân cụm KMeans
for i, k in enumerate(k_values, 2):  # Bắt đầu từ 2 để ảnh gốc là ô 1
    clusters, centroids = kmeans(pixel_values, k)
    segmented_image = centroids[clusters].astype(np.uint8)
    segmented_image = segmented_image.reshape(image.shape)

    plt.subplot(2, 3, i)
    plt.imshow(segmented_image)
    plt.title(f'KMeans với k={k}')
    plt.axis('off')

plt.tight_layout()
plt.show()
