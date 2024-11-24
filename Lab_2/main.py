import numpy
import tensorflow as tf
import sys

def rotate_point(point, angle):
    rotation_matrix = tf.stack(
        [(tf.cos(angle),
        -tf.sin(angle)),
        (tf.sin(angle),
        tf.cos(angle))],
        axis=0
    )
    rotation_matrix = tf.reshape(rotation_matrix, (2, 2))
    return tf.matmul(rotation_matrix, tf.reshape(point, (2, 1)).numpy())

def solve_linear_functions(ab_matrix, c_matrix):
    ab_matrix = tf.stack(ab_matrix, axis=0).numpy().astype(numpy.float32)
    ab_matrix = tf.linalg.inv(ab_matrix)
    c_matrix = tf.stack(c_matrix, axis=0).numpy().astype(numpy.float32)

    return tf.matmul(ab_matrix, tf.reshape(c_matrix, (2, 1)))

def main():
    # print(rotate_point([2,2], math.pi / 2))
    ab_matrix = sys.argv[1].split(" ")
    c_matrix = sys.argv[2].split(" ")

    ab_matrix = [int(x) for x in ab_matrix]
    c_matrix = [int(x) for x in c_matrix]

    top = ab_matrix[:len(ab_matrix) // 2]
    bottom = ab_matrix[len(ab_matrix) // 2:]
    ab_matrix = [top, bottom]

    try:
        solution = solve_linear_functions(
            ab_matrix,
            c_matrix
        )
    except Exception:
        print("Solution not available")
        return

    print(f"{solution[0][0]}, {solution[1][0]}")

if __name__ == "__main__":
    main()