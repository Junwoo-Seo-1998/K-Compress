import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import math


def kmeanCompute(data: np.array, ks: np.array):
    # time0 = time.time()
    cluster = [0 for i in range(data.shape[0])]
    """
    #slow
    dist_list = [0 for i in range(ks.shape[0])]
    for i in range(data.shape[0]):
        for k in range(ks.shape[0]):
            dist_list[k]=np.sum(np.square(data[i] - ks[k]))
        cluster[i]=np.argmin(dist_list)
    """
    labels = []
    for k in range(ks.shape[0]):
        # print(np.sum(np.square((data - ks[k])), axis=1).shape)
        labels.append(np.sum(np.square(data - ks[k]), axis=1))
    for i in range(data.shape[0]):
        min_k = -1
        min_val = math.inf
        for k in range(ks.shape[0]):
            if min_val > labels[k][i]:
                min_k = k
                min_val = labels[k][i]
        cluster[i] = min_k
    # print("cluster time took: ", time.time() - time0)
    return cluster


def recomputeMeans(data: np.array, cluster: list, ks: np.array):
    result = np.zeros(shape=ks.shape)
    # time0 = time.time()
    for i in range(ks.shape[0]):
        result[i] = np.average(data[np.array(cluster) == i], axis=0)
    # print("recomputeMeans time took: ", time.time() - time0)
    return result


def kmean(data: np.array, ks: np.array):
    cluster = kmeanCompute(data, ks)
    # print("cluster: ",cluster)
    ks = recomputeMeans(data, cluster, ks)
    # print("recomputed centroids: ", ks)
    prev = cluster
    while True:
        cluster = kmeanCompute(data, ks)
        # print("cluster: ", cluster)
        if cluster == prev:
            # print("final centroids: ", ks)
            break
        ks = recomputeMeans(data, cluster, ks)
        # print("recomputed centroids: ", ks)
        prev = cluster

    return ks


def GenImage(image_file, k, prefix):
    ks = np.array([[i, i, i] for i in range(0, 256, 255 // (k - 1))])
    image = cv2.imread(image_file, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    new_image = image.copy()
    flat_data = new_image.reshape(image.shape[0] * image.shape[1], 3)
    # print(kmeanCompute(flat_data,ks))
    time0 = time.time()
    centorid = kmean(flat_data, ks)
    # print("final: ", centorid)
    print("time took: ", time.time() - time0)
    cluster_labels = kmeanCompute(flat_data, centorid)
    centorid = centorid.astype(np.uint8)
    print("k=", k)
    print(centorid)

    for i in range(flat_data.shape[0]):
        flat_data[i] = centorid[cluster_labels[i]]

    cv2.imwrite(
        "{}_k_{}.jpg".format(prefix, str(k)), cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    )


def GenImages(filename, maxK, prefix):
    for i in range(3, maxK + 1):
        GenImage(filename, i, prefix)


def ShowImg(fileName):
    plt.rcParams["figure.figsize"] = [8, 6]
    fig = plt.figure()  # rows*cols
    rows = 3
    cols = 3
    i = 1
    for index in range(rows * cols):
        ax = fig.add_subplot(rows, cols, i)
        img = None
        if i == 1:
            img = cv2.imread("{}.jpg".format(fileName))
            ax.set_xlabel("original")
        else:
            print("{}_k_{}.jpg".format(fileName, i + 1))
            img = cv2.imread("{}_k_{}.jpg".format(fileName, i + 1))
            ax.set_xlabel("k = {}".format(i + 1))

        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        ax.set_xticks([]), ax.set_yticks([])
        i += 1
    plt.show()


def main():

    print("img1 result: ")
    GenImages("img1.jpg", 10, "img1")
    print()

    print("img2 result: ")
    GenImages("img2.jpg", 10, "img2")
    print()

    print("img3 result: ")
    GenImages("img3.jpg", 10, "img3")
    print()

    ShowImg("img1")
    ShowImg("img2")
    ShowImg("img3")

    """
    #just to check
    k=10
    ks=np.array([[i,i,i] for i in range(0, 256, 255//(k-1))])
    print(ks)
    image = cv2.imread("img2.jpg", cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print(image.shape)

    new_image=image.copy()
    flat_data=new_image.reshape(image.shape[0]*image.shape[1], 3)
    #print(kmeanCompute(flat_data,ks))
    time0 = time.time()
    centorid = kmean(flat_data, ks)
    print("final: ", centorid)
    print("time took: ",time.time() - time0)
    cluster_labels=kmeanCompute(flat_data, centorid)
    centorid=centorid.astype(np.uint8)
    print(centorid)

    for i in range(flat_data.shape[0]):
        flat_data[i]=centorid[cluster_labels[i]]

    fig, ax = plt.subplots(1, 2)

    ax[0].imshow(image)
    ax[1].imshow(new_image)
    cv2.imwrite('k_{}.jpg'.format(str(k)), cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR))
    plt.show()
    """


if __name__ == "__main__":
    main()
