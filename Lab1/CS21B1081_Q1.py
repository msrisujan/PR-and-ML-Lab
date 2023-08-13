import numpy as np
    

def matrix_addition(matrix1, matrix2):
    return np.add(matrix1, matrix2)

def matrix_subtraction(matrix1, matrix2):
    return np.subtract(matrix1, matrix2)

def scalar_matrix_multiplication(matrix, scalar):
    return np.multiply(matrix, scalar)

def elementwise_matrix_multiplication(matrix1, matrix2):
    return np.multiply(matrix1, matrix2)

def matrix_multiplication(matrix1, matrix2):
    return np.dot(matrix1, matrix2)

def matrix_transpose(matrix):
    return np.transpose(matrix)

def trace_of_matrix(matrix):
    return np.trace(matrix)

def solve_system_of_linear_equations(matrix1, matrix2):
    return np.linalg.solve(matrix1, matrix2)

def determinant(matrix):
    return np.linalg.det(matrix)

def inverse(matrix):
    return np.linalg.inv(matrix)

def eigen_value_and_eigen_vector(matrix):
    return np.linalg.eig(matrix)

if __name__ == "__main__":
    print(" 1. Matrix Addition")
    print(" 2. Matrix Subtraction")
    print(" 3. Scalar Matrix Multiplication")
    print(" 4. Elementwise Matrix Multiplication")
    print(" 5. Matrix Multiplication")
    print(" 6. Matrix Transpose")
    print(" 7. Trace of a Matrix")
    print(" 8. Solve System of Linear Equations")
    print(" 9. Determinant")
    print("10. Inverse")
    print("11. Eigen Value and Eigen Vector")
    print("12. Exit")
    print("MATRIX 1")
    rows1 = int(input("Enter the number of rows: "))
    columns1 = int(input("Enter the number of columns: "))
    matrix = []
    print("Enter the matrix")
    for i in range(rows1):
        row = list(map(int, input().split()))
        if len(row) != columns1:
            print("Invalid input")
            exit()
        matrix.append(row)
    matrix1 = np.array(matrix)
    print("MATRIX 2")
    rows2 = int(input("Enter the number of rows: "))
    columns2 = int(input("Enter the number of columns: "))
    matrix = []
    print("Enter the matrix")
    for i in range(rows2):
        row = list(map(int, input().split()))
        if len(row) != columns2:
            print("Invalid input")
            exit()
        matrix.append(row)
    matrix2 = np.array(matrix)
    while True:
        choice = int(input("Enter your choice: "))
        if choice == 1:
            if rows1 != rows2 or columns1 != columns2:
                print("Invalid operation")
            else:
                print(matrix_addition(matrix1, matrix2))
        elif choice == 2:
            if rows1 != rows2 or columns1 != columns2:
                print("Invalid operation")
            else:
                print(matrix_subtraction(matrix1, matrix2))
        elif choice == 3:
            scalar = int(input("Enter the scalar: "))
            option = int(input("Enter 1 for matrix1 and 2 for matrix2: "))
            if option == 1:
                print(scalar_matrix_multiplication(matrix1, scalar))
            elif option == 2:
                print(scalar_matrix_multiplication(matrix2, scalar))
            else:
                print("Invalid option")
        elif choice == 4:
            if rows1 != rows2 or columns1 != columns2:
                print("Invalid operation")
            else:
                print(elementwise_matrix_multiplication(matrix1, matrix2))
        elif choice == 5:
            if columns1 != rows2 or rows1 != columns2:
                print("Invalid operation")
            else:
                print(matrix_multiplication(matrix1, matrix2))
        elif choice == 6:
            option = int(input("Enter 1 for matrix1 and 2 for matrix2: "))
            if option == 1:
                print(matrix_transpose(matrix1))
            elif option == 2:
                print(matrix_transpose(matrix2))
            else:
                print("Invalid option")
        elif choice == 7:
            option = int(input("Enter 1 for matrix1 and 2 for matrix2: "))
            if option == 1:
                if rows1 != columns1:
                    print("Invalid operation")
                else:
                    print(trace_of_matrix(matrix1))
            elif option == 2:
                if rows2 != columns2:
                    print("Invalid operation")
                else:
                    print(trace_of_matrix(matrix2))
            else:
                print("Invalid option")
        elif choice == 8:
            b=list(map(int,input("Enter the matrix b: ").split()))
            b=np.array(b)
            option = int(input("Enter 1 for matrix1 and 2 for matrix2: "))
            if option == 1:
                if columns1 != len(b):
                    print("Invalid operation")
                else:
                    print(solve_system_of_linear_equations(matrix1, b))
            elif option == 2:
                if columns2 != len(b):
                    print("Invalid operation")
                else:
                    print(solve_system_of_linear_equations(matrix2, b))
            else:
                print("Invalid option")
        elif choice == 9:
            option = int(input("Enter 1 for matrix1 and 2 for matrix2: "))
            if option == 1:
                if rows1 != columns1:
                    print("Invalid operation")
                else:
                    print(determinant(matrix1))
            elif option == 2:
                if rows2 != columns2:
                    print("Invalid operation")
                else:
                    print(determinant(matrix2))
            else:
                print("Invalid option")
        elif choice == 10:
            option = int(input("Enter 1 for matrix1 and 2 for matrix2: "))
            if option == 1:
                if rows1 != columns1:
                    print("Invalid operation")
                elif determinant(matrix1) == 0:
                    print("Inverse does not exist")
                else:
                    print(inverse(matrix1))
            elif option == 2:
                if rows2 != columns2:
                    print("Invalid operation")
                elif determinant(matrix2) == 0:
                    print("Inverse does not exist")
                else:
                    print(inverse(matrix2))
            else:
                print("Invalid option")
        elif choice == 11:
            option = int(input("Enter 1 for matrix1 and 2 for matrix2: "))
            if option == 1:
                if rows1 != columns1:
                    print("Invalid operation")
                else:
                    print(eigen_value_and_eigen_vector(matrix1))
            elif option == 2:
                if rows2 != columns2:
                    print("Invalid operation")
                else:
                    print(eigen_value_and_eigen_vector(matrix2))
            else:
                print("Invalid option")
        elif choice == 12:
            exit()
        else:
            print("Invalid choice")

