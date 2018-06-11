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
/*------------------------Array数组-----------------------*/
/**********************************************************/

//二维数组中的查找
bool Find(int target, vector<vector<int> > array);

//旋转数组的最小数字
int minNumberInRotateArray(vector<int> rotateArray);

//调整数组顺序使奇数位于偶数前面
void reOrderArray(vector<int> &array);

//数组中出现次数超过一半的数字
int MoreThanHalfNum_Solution(vector<int> numbers);

//连续子数组的最大和
int FindGreatestSumOfSubArray(vector<int> array);

//把数组排成最小的数
string PrintMinNumber(vector<int> numbers);

//数组中的逆序对
long mergeCount(vector<int>& data, int lo, int hi);
int InversePairs(vector<int> data);

//数字在排序数组中出现的次数
int GetNumberOfK(vector<int> data, int k);

//数组中只出现一次的数字
void FindNumsAppearOnce(vector<int> data, int* num1, int *num2);

//数组中重复的数字
bool duplicate(int numbers[], int length, int* duplication);

//顺时针打印矩阵
vector<int> printMatrix(vector<vector<int> > matrix);

//和为S的连续正数序列
vector<vector<int> > FindContinuousSequence(int sum);

//丑数
int GetUglyNumber_Solution(int index);

//和为S的两个数字
vector<int> FindNumbersWithSum(vector<int> array, int sum);

//扑克牌顺子
bool IsContinuous(vector<int> numbers);

//数据流中的中位数
void Insert(int num);
double GetMedian();

//最小的k个数
vector<int> GetLeastNumbers_Solution(vector<int> input, int k);

/************************************************************/
/*----------------------------字符串------------------------*/
/************************************************************/