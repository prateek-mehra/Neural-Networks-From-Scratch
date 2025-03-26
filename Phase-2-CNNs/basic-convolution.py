import numpy as np

def conv2D(image, kernel):

    image_h, image_w = image.shape
    kernel_h, kernel_w = kernel.shape

    # Output size after convolution (assuming no padding, stride = 1)
    output_h = image_h - kernel_h + 1
    output_w = image_w - kernel_w + 1

    output = np.zeros((output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):

            filter = image[i : i + kernel_h, j : j + kernel_w]
            output[i, j] = np.sum(filter * kernel)
    
    return output

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

    convoluted_output = conv2D(image, kernel)

    print("convoluted_output: ", convoluted_output)

