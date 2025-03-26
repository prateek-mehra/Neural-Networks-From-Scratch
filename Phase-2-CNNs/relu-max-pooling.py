import numpy as np

def relu(x):
    return np.maximum(0, x)

def max_pooling(img, pool_len = 2, stride = 2):

    img_h, img_w = img.shape
    output_h = (img_h - pool_len) // stride + 1
    output_w = (img_w - pool_len) // stride + 1

    output = np.zeros((output_h, output_w))

    for i in range(0, img_h - pool_len + 1, stride):
        for j in range(0, img_w - pool_len + 1, stride):

            region = img[i:i+pool_len, j:j+pool_len]
            output[i // stride, j // stride] = np.max(region)
    
    return output

if __name__ == "__main__":

    image = np.array([
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20],
            [21, 22, 23, 24, 25],
        ])

    print("relu result: ", relu(image))

    print("max pooling result: ", max_pooling(image))



