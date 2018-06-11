#pragma once
#include <bits/stdc++.h>
#include <regex>
using namespace std;
struct TreeNode {
	int val;
	struct TreeNode *left;
	struct TreeNode *right;
	TreeNode(int x) :
		val(x), left(NULL), right(NULL) {
	}
};

struct ListNode {
	int val;
	ListNode *next;
	ListNode(int x) : val(x), next(NULL) {}
};

struct TreeLinkNode {
	int val;
	struct TreeLinkNode *left;
	struct TreeLinkNode *right;
	struct TreeLinkNode *next;
	TreeLinkNode(int x) :val(x), left(NULL), right(NULL), next(NULL) {

	}
};

struct RandomListNode {
	int label;
	struct RandomListNode *next, *random;
	RandomListNode(int x) :
		label(x), next(NULL), random(NULL) {
	}
};

/*********************************************************/
/*------------------------Array����-----------------------*/
/**********************************************************/

//��ά�����еĲ���
bool Find(int target, vector<vector<int> > array);

//��ת�������С����
int minNumberInRotateArray(vector<int> rotateArray);

//��������˳��ʹ����λ��ż��ǰ��
void reOrderArray(vector<int> &array);

//�����г��ִ�������һ�������
int MoreThanHalfNum_Solution(vector<int> numbers);

//���������������
int FindGreatestSumOfSubArray(vector<int> array);

//�������ų���С����
string PrintMinNumber(vector<int> numbers);

//�����е������
long mergeCount(vector<int>& data, int lo, int hi);
int InversePairs(vector<int> data);

//���������������г��ֵĴ���
int GetNumberOfK(vector<int> data, int k);

//������ֻ����һ�ε�����
void FindNumsAppearOnce(vector<int> data, int* num1, int *num2);

//�������ظ�������
bool duplicate(int numbers[], int length, int* duplication);

//˳ʱ���ӡ����
vector<int> printMatrix(vector<vector<int> > matrix);

//��ΪS��������������
vector<vector<int> > FindContinuousSequence(int sum);

//����
int GetUglyNumber_Solution(int index);

//��ΪS����������
vector<int> FindNumbersWithSum(vector<int> array, int sum);

//�˿���˳��
bool IsContinuous(vector<int> numbers);

//�������е���λ��
void Insert(int num);
double GetMedian();

//��С��k����
vector<int> GetLeastNumbers_Solution(vector<int> input, int k);

/************************************************************/
/*----------------------------�ַ���------------------------*/
/************************************************************/