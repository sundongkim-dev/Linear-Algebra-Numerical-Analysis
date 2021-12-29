/* mxn 행렬 A를 QR 분해한다. Q행렬은 곧 m*m 행렬이며, orthonormal vector basis를 뜻한다.
Given a matrix A of dimension m by n, this algorithm
computes a QR decomposition of A, where Q is a unitary
m by n matrixand R is a n by n upper triangular matrix
and A = QR.
*/
#include <iostream>
#include <cmath>
#include <vector>

#define EPS 1.0e-13
using namespace std;

vector<vector<double>> a(10, vector<double> (10));  // 과제의 벡터 범위: 2 <= # <= 10
vector<vector<double>> q(10, vector<double>(100));

void printVector(vector<vector<double>> v, int n, int numOfVecters)
{
    for (int i = 0; i < numOfVecters; i++)
    {
        cout << "q" << i+1 << " vector: ";
        for (int j = 0; j < n; j++)
        {
            cout << v[i][j] << " ";
        }
        cout << "\n";
    }
}
void normalize(vector<vector<double>>& v, int idx) // normalize the vector
{
    double sum = 0;
    for (int i = 0; i < v.size(); i++)
    {
        sum += pow(v[idx][i], 2);
    }
    double lengthOfVector = sqrt(sum);
    /*if (lengthOfVector < 1.1e-14)
    {
        for (int i = 0; i < v.size(); i++)
            v[idx][i] = 0;
    }
    else
    {
        for (int i = 0; i < v.size(); i++)
        {
            v[idx][i] /= lengthOfVector;
        }
    }*/
    
    for (int i = 0; i < v.size(); i++)
    {
        v[idx][i] /= lengthOfVector;
    }
}

double dotProduct(vector<double> v1, vector<double> v2, int n) // v1T * v2
{
    double val = 0;
    for (int i = 0; i < n; i++)
    {
        val += (v1[i] * v2[i]);
    }
    return val;
}

void gramSchmidt(vector<vector<double>> &q, const vector<vector<double>> A, int n, int numOfVecters)
{
    // q0 얻음
    normalize(q, 0);
    // q1부터 qn구해야 함
    for (int i = 1; i < numOfVecters; i++)
    {
        //qi에서 뺼값 계산
        for (int j = 0; j < i; j++)
        {
            double coefficient = dotProduct(q[j], A[i], n);
            vector<double> tmp;
            for (int k = 0; k < n; k++)
            {
                tmp.push_back(q[j][k] * coefficient);
            }
            for (int j = 0; j < n; j++)
            {
                q[i][j] -= tmp[j];
            }      
        }
        normalize(q, i);
    }    
}

bool verifyOrthonormal(vector<vector<double>>& q, int n, int numOfVectors)
{
    bool flag = true;
    double result;
    for (int i = 0; i < numOfVectors; i++)
    {
        for (int j = 1; j < numOfVectors; j++)
        {
            if (i == j || i > j)
                continue;
            result = dotProduct(q[i], q[j], n);
            cout << result << "\n";
            if (abs(result) > EPS)
                flag = false;
        }
    }
    return flag;
}

int main()
{
    int n, numOfVecters;
    cout << "Input the number of dimension of vector and the number of vectors" << endl;
    cin >> n >> numOfVecters;               // numOfVectors >= n
    cout << "Input the elements of the vectors" << endl;
    for (int i = 0; i < numOfVecters; i++)
    {
        for (int j = 0; j < n; j++)
        {
            cin >> a[i][j];
        }
    }
    q = a;
    cout << "\n";
    gramSchmidt(q, a, n, numOfVecters);
    printVector(q, n, numOfVecters);
    cout << "Verify that Q vectors are orthonormal" << endl;
    if (verifyOrthonormal(q, n, numOfVecters))
    {
        cout << "Verified" << endl;
    }
    else
    {
        cout << "Wrong answer" << endl;
    }
    return 0;
}

