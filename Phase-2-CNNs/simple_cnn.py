import numpy as np

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

def conv2D(img, kernel):

    img_h, img_w = img.shape
    kernel_h, kernel_w = kernel.shape

    output_h = (img_h - kernel_h) + 1
    output_w = (img_w - kernel_w) + 1

    output = np.zeros((output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):

            filter = img[i:i+kernel_h, j:j+kernel_w]
            output[i, j] = np.sum(filter * kernel)
    
    return output


def relu(x):
    return np.maximum(0, x)

def fully_connected_layer(flattened_output, weights, bias):
    return np.dot(flattened_output, weights) + bias

def simple_cnn(img, kernel, weights, bias):

    convoluted_output = conv2D(img, kernel)
    relu_output = relu(convoluted_output)
    pooled_output = max_pooling(relu_output)

    flattened_output = pooled_output.flatten()

    fc_output = fully_connected_layer(flattened_output, weights, bias)

    return fc_output

if __name__ == "__main__":

    image = np.array([
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15],
        [16, 17, 18, 19, 20],
        [21, 22, 23, 24, 25],
    ])

    kernel = np.array([
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1],
    ])


    neurons = 1
    flattened_size = (((image.shape[0] - 2) // 2) * ((image.shape[1] - 2) // 2))
    weights = np.random.randn(flattened_size, neurons)
    bias = np.random.randn(neurons)

    print("CNN output: ", simple_cnn(image, kernel, weights, bias))