import tensorflow as tf

# Load the dataset
(img_train, result_train), (img_test, result_test) = tf.keras.datasets.mnist.load_data()

# Normalize the pixel values to [0, 1]
img_train, img_test = img_train / 255.0, img_test / 255.0

def main():
    return

if __name__ == "__main__":
    main()