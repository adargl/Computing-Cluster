import threading
import random


def multiply_chunk(matrix1, matrix2, start, end, result):
    """Multiplies a chunk of two matrices and stores the result in a shared result matrix"""
    for i in range(start, end):
        for j in range(len(matrix2[0])):
            result[i][j] = sum(matrix1[i][k] * matrix2[k][j] for k in range(len(matrix1[0])))


def threaded_matrix_multiply(matrix1, matrix2):
    """Multiplies two matrices using threads"""
    # Create a shared result matrix
    result = [[0 for _ in range(len(matrix2[0]))] for _ in range(len(matrix1))]

    # Calculate the size of each chunk to be multiplied
    num_threads = 5
    chunk_size = len(matrix1) // num_threads

    var1 = 0
    var2 = 2 * chunk_size
    var3 = 3 * chunk_size
    var4 = 4 * chunk_size
    var5 = len(matrix1)

    # Create thread objects
    thread1 = threading.Thread(target=multiply_chunk, args=(matrix1, matrix2, var1, chunk_size, result))
    thread2 = threading.Thread(target=multiply_chunk, args=(matrix1, matrix2, chunk_size, var2, result))
    thread3 = threading.Thread(target=multiply_chunk, args=(matrix1, matrix2, var2, var3, result))
    thread4 = threading.Thread(target=multiply_chunk, args=(matrix1, matrix2, var3, var4, result))
    thread5 = threading.Thread(target=multiply_chunk, args=(matrix1, matrix2, var4, var5, result))

    # Start the threads
    thread1.start()
    thread2.start()
    thread3.start()
    thread4.start()
    thread5.start()

    # Wait for all threads to finish
    thread1.join()
    thread2.join()
    thread3.join()
    thread4.join()
    thread5.join()

    return result


# Create two matrices to multiply
mat1 = [[random.random() for _ in range(10)] for _ in range(20)]
mat2 = [[random.random() for _ in range(20)] for _ in range(10)]

# Multiply the matrices using threads
multiplied = threaded_matrix_multiply(mat1, mat2)

# Print the result
for row in multiplied:
    print(row)




