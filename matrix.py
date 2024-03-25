from random import shuffle
import numpy as np
from time import time

timeout = time() + 10  # seconds


def random_vector(length, weight):
    arr = [True] * weight + [False] * (length - weight)
    shuffle(arr)
    return np.array(arr, dtype='bool')


def generate_matrix_h(rows, columns, weight):
    # initialize matrix with 0 binary
    matrix = np.zeros((rows, columns), dtype='bool')
    for index in range(0, rows):
        new_row = [True] * weight + [False] * (columns - weight)
        shuffle(new_row)
        matrix[index] = list(new_row)
    print("Matrix before remove")
    print_matrix(matrix)
    return matrix


def get_rank(m):
    return np.linalg.matrix_rank(m)


def print_matrix(matrix, binary=True):
    if binary:
        matrix = matrix * 1
        for row in matrix:
            print(row)
    else:
        for row in matrix:
            print(row)
    print()


def print_rank(matrix):
    print("The rank is ", np.linalg.matrix_rank(matrix))


def remove_duplicates_rows(matrix, n, k):
    rank = get_rank(matrix)
    h_final = np.copy(matrix)
    delta = n - k - get_rank(matrix)
    if delta != 0:
        h_temp = np.copy(matrix)
        while delta > 0:
            for row in range(0, len(h_temp)):
                h_temp = np.delete(h_temp, row, 0)
                if rank == get_rank(h_temp):
                    matrix = np.copy(h_temp)
                    h_final = np.copy(h_temp)
                break
            else:
                h_temp = np.copy(matrix)
        delta = delta - 1
    else:
        print('Rank full')
    print("Matrix without duplicate rows")
    print_matrix(h_final)
    return h_final


def remove_zero_columns(matrix):
    false_columns = np.all(matrix == False, axis=0)
    filtered_matrix = matrix[:, ~false_columns]
    print("matrix without 0 columns")
    print_matrix(filtered_matrix)
    num_columns = filtered_matrix.shape[1]
    num_rows = filtered_matrix.shape[0]
    return filtered_matrix, num_columns, num_rows
    

def calculate_score(matrix, syndrome, n):
    ls, = syndrome.nonzero()
    l_score = []
    for index in range(0, n):
        nonzero, = matrix[:, index].nonzero()
        intersect = np.intersect1d(nonzero, ls)
        l_score.append(len(intersect))
    # print("score", l_score)
    return l_score


def decode(matrix, syndrome, n, k):
    decode_syndrome = np.copy(syndrome)
    y = np.zeros(n, dtype="bool")
    length, = decode_syndrome.nonzero()
    print("length", length)
    while len(length) != 0:
        score = calculate_score(matrix, decode_syndrome, n)
        print("score = ", score)
        m = max(score)
        list_max, = (np.array(score) >= m).nonzero()
        print("list max = ", list_max)
        v = np.zeros(k, dtype="bool")
        x = np.zeros(n, dtype="bool")
        for column in list_max:
            v = np.logical_xor(v,np.transpose(matrix)[:][column])
            x[column] = 1
        print("v = ", v*1)   
        print("syndrome before= ", decode_syndrome*1)   
        decode_syndrome = np.logical_xor(decode_syndrome, v)
        print("x", x*1)
        print("syndrome", decode_syndrome*1)
        print("y", y*1)
        y = np.logical_xor(y,x)
        length, = decode_syndrome.nonzero()
        print("updated length", length*1)
        if time() > timeout:
            return False
    return len(length) == 0


if __name__ == '__main__':
    validity = []
    n = 2000  # columns
    k = 1000  # n-k rows
    w = 5  # number of 1 value on a row
    # errorWeight = 3  # complexity of error vector
    generated_matrix = generate_matrix_h(n - k, n, w)  # generating random matrix
    zero_columns_matrix, new_n, new_k = remove_zero_columns(generated_matrix)
    striped_matrix = remove_duplicates_rows(zero_columns_matrix, new_n,new_k)  # cleaning matrix => generator matrix H
    start_error_weight = 2
    stop_error_weight = 4
    number_of_retries = 100
    for errorWeight in range(start_error_weight, stop_error_weight):
        true_count = 0
        for i in range(1, number_of_retries):
            print(f"Weight: {errorWeight}, i: {i}")
            error_vector = random_vector(new_n, errorWeight)  # error vector
            print("error vector", error_vector*1)
            S = striped_matrix @ error_vector  # calculating syndrome
            calculated_syndrome = np.copy(S)
            print("decode")
            print(decode(striped_matrix, calculated_syndrome, new_n, new_k))
            if (decode(striped_matrix, calculated_syndrome, new_n, new_k)) is True:
                true_count += 1
        result = true_count / (number_of_retries-1)
        validity.append(result)
    print(validity)
