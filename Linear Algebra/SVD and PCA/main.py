'''
1. Implement SVD for randomly generated 100-D vectors
2. Find some principal component
3. Represent 100-D vectors with the selected principal component vectors
4 Discuss the errors in representing vectors with partial basis (less than 20~30 basis vectors)
'''

import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    A = np.empty((0,100))
    for i in range(1000):
        arr = np.array([])
        # arr = np.append(arr, np.random.randint(-100, 100, size=10))
        a = np.random.randint(-100, 100, size=10)
        # Define 5 relations between 2 elements => Total 10 elements are selected
        a[0] = a[1]; a[2] = 3*a[8]; a[4] = -2*a[6]; a[7] = -a[9]; a[3] = 2*a[5];
        arr = np.append(arr, a)
        arr = np.append(arr, np.random.randint(-20, 20, size=45))
        arr = np.append(arr, np.random.randint(-5, 5, size=45))
        A = np.vstack([A, arr])
    np.set_printoptions(threshold=10000000, linewidth=np.inf)
    AT = np.transpose(A)
    # print(A.shape)
    # print(AT.shape)
    ATA = AT.dot(A) # 100 X 100 matrix = AT * A
    # print(AT.dot(A))
    U, S, VT = np.linalg.svd(ATA, full_matrices=True)

    # print("U:" + str(U.shape))
    # print("S:" + str(S.shape))
    # print("VT:" + str(VT.shape))
    # print(U)
    rankOfA = np.linalg.matrix_rank(A)
    # print(np.linalg.matrix_rank(U))
    # print(S)
    # A에서 100개 column가져와서 100*100D
    randomVector = np.array([])
    randomVector = AT[:, :100].copy()
    n = 5
    xAxis = []
    vectorDist = np.array([])
    while True:
        xAxis.append(n)
        basisVector = U[:, :n].copy()
        coefficient = np.array([])
        lstart = 0; lend = 1; rstart = 0; rend = 1;
        approximationVector = np.zeros((100, 0), float)
        # randomVector에서 하나의 열 뽑아내기 = x1, x2, ... , x100
        for idx in range(100):
            # randomVector에서 하나의 열 고정
            leftVal = randomVector[:, lstart:lend].copy()
            
            # 총 계수 획득 : randomVector에서 고른 하나의 열과 en 내적한 계수들 구하기, n만큼 반복
            for i in range(n):
                # x = (x1T@e1)e1 + (x1T@e2)e2 + (x1T@e3)e3 + (x1T@e4)e4 + (x1T@e5)e5
                rightVal = basisVector[:, rstart:rend].copy()
                # 계수 획득
                coefficient = np.append(coefficient, np.dot(np.transpose(leftVal), rightVal))
                rstart += 1; rend += 1;

            # approximationVector의 한 column 획득
            approximationColumn = np.zeros(shape=(100, 1))
            # 새로운 x'의 column 추가하기
            start = 0; end = 1;
            for i in range(n):
                approximationColumn += coefficient[i] * basisVector[:, start:end]
                start += 1; end += 1
            approximationVector = np.append(approximationVector, approximationColumn, axis=1)

        # get vector distance between randomVector & approximationVector
        dist = (randomVector-approximationVector)**2
        dist = dist.sum(axis=-1)
        dist = np.sqrt(dist)
        # print(np.mean(dist))
        vectorDist = np.append(vectorDist, np.mean(dist))
        lstart += 1; lend += 1;
        n += 5
        if n > rankOfA:
            break
    #print(vectorDist)
    #print(xAxis)
    plt.plot(xAxis, vectorDist, 'ro')
    plt.axis([0,100,0,250])
    plt.xticks(np.arange(0,100,5))
    plt.xlabel('Number of basis vectors')
    plt.ylabel('Average distance for each basis')
    plt.show()