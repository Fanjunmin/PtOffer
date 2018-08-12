// CopyRight 2018, FanJunmin.
// All rights reserved.
// Author: junminfan@126.com (FanJunmin)
// Update: 2018/8/5

#pragma once
#ifndef POINTTOOFFER_H_
#define POINTTOOFFER_H_
//#include <bits/stdc++.h>
#include <iostream>
#include <assert.h>  //for assert
#include <algorithm>
#include <list>
#include <map>
#include <numeric>  //for iota()
#include <queue>
#include <regex>
#include <set>
#include <stack>
#include <string>
#include <vector>

using namespace std;

//��Ҫ������һЩ���ݽṹ�Լ���ʼ��/�ͷź���
struct TreeNode {
  int val;
  struct TreeNode* left;
  struct TreeNode* right;
  TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};

struct ListNode {
  int val;
  ListNode* next;
  ListNode(int x) : val(x), next(NULL) {}
};

struct TreeLinkNode {
  int val;
  struct TreeLinkNode* left;
  struct TreeLinkNode* right;
  struct TreeLinkNode* next;
  TreeLinkNode(int x) : val(x), left(NULL), right(NULL), next(NULL) {}
};

struct RandomListNode {
  int label;
  struct RandomListNode *next, *random;
  RandomListNode(int x) : label(x), next(NULL), random(NULL) {}
};

//------------------------Array����-----------------------

//��ά�����еĲ���
bool Find(int target, vector<vector<int>> array);

//��ת�������С����
int minNumberInRotateArray(vector<int> rotateArray);

//��������˳��ʹ����λ��ż��ǰ��
void reOrderArray(vector<int>& array);

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
void FindNumsAppearOnce(vector<int> data, int* num1, int* num2);

//�������ظ�������
bool duplicate(int numbers[], int length, int* duplication);

//˳ʱ���ӡ����
vector<int> printMatrix(vector<vector<int>> matrix);

//��ΪS��������������
vector<vector<int>> FindContinuousSequence(int sum);

//����
int GetUglyNumber_Solution(int index);

//��ΪS����������
vector<int> FindNumbersWithSum(vector<int> array, int sum);

//�˿���˳��
bool IsContinuous(vector<int> numbers);

//�������е���λ��
static priority_queue<int> g_small_que;	//��С���ȶ���
static priority_queue<int, vector<int>, greater<int>> g_big_que;		//������ȶ���
static int g_insert_count = 0;		//�������
void Insert(int num);
double GetMedian();

//��С��k����
vector<int> GetLeastNumbers_Solution(vector<int> input, int k);

//------------------------�ַ���-----------------------

//����ת�ַ���
string LeftRotateString(string str, int n);

//��ʾ��ֵ���ַ���
bool isNumeric(char* string);

//���ַ���ת��������
long power(int e, int m);
int StrToInt(string str);

//�ַ���������
void pBackTracking(set<string>& strSet, string& str, int beg);
vector<string> Permutation(string str);

//�滻�ո�
void replaceSpace(char* str, int length);

//��һ��ֻ����һ�ε��ַ�
int FirstNotRepeatingChar(string str);

//������ʽƥ��
bool match(char* str, char* pattern);

//��ת����˳����
string ReverseSentence(string str);

//�ַ����е�һ�����ظ����ַ�
static list<char> dataList;
static map<char, int> countMap;
void Insert(char ch);
char FirstAppearingOnce();

//------------------------��-----------------------

//�ؽ�������
TreeNode* reConstructBinaryTree(vector<int> pre, vector<int> vin);

//�����ӽṹ
bool HasSubtree(TreeNode* pRoot1, TreeNode* pRoot2);
bool isSubtree(TreeNode* pRoot1, TreeNode* pRoot2);

//�������ľ���
void Mirror(TreeNode* pRoot);

//�������´�ӡ������
vector<int> PrintFromTopToBottom(TreeNode* root);

//�����������ĺ����������
bool judgeBST(vector<int>::iterator first, vector<int>::iterator last);
bool VerifySquenceOfBST(vector<int> sequence);

//�������к�Ϊĳһֵ��·��
void FindPath(vector<vector<int>>& vec_store,
							vector<int> store,
							TreeNode* root,
              int expNumber);
vector<vector<int>> FindPath(TreeNode* root, int expectNumber);

//������������˫������
void recurConvert(TreeNode* root, TreeNode*& pre);
TreeNode* Convert(TreeNode* pRootOfTree);

//���������
int TreeDepth(TreeNode* pRoot);

//ƽ�������
bool IsBalanced(TreeNode* pRoot, int& pDepth);
bool IsBalanced_Solution(TreeNode* pRoot);

//����������һ�����
TreeLinkNode* GetNext(TreeLinkNode* pNode);

//�ԳƵĶ�����
bool isSymmetrical(TreeNode* lChild, TreeNode* rChild);
bool isSymmetrical(TreeNode* pRoot);

//�Ѷ�������ӡ�ɶ���
vector<vector<int>> Print1(TreeNode* pRoot);

//��֮����˳���ӡ������
vector<vector<int>> Print2(TreeNode* pRoot);

//�����������ĵ�k�����
void inOrderTraversal(TreeNode* pRoot, vector<TreeNode *>& store);
TreeNode* KthNode(TreeNode* pRoot, int k);

//���л�������
//���������������vector�洢
void Serialize(vector<int>& sto, TreeNode* root);
char* Serialize(TreeNode* root);
TreeNode* Deserialize(int*& str);
TreeNode* Deserialize(char* str);

//---------------------------����-------------------------------

//�����е�·��
bool hasPath(char* matrix, int rows, int cols, char* str);
bool hasPath(char* matrix, int rows, int cols, char* str, bool* flag, int x,
             int y, int index);

//�����˵��˶���Χ
bool IsVaildVal(int x, int y, int threshold);
int Find(vector<int> &id, int p);
void Union(vector<int>& id, vector<int>& sz, int p, int q);
int movingCount(int threshold, int rows, int cols);

//---------------------------ջ�Ͷ���--------------------------

//�������ڵ����ֵ
vector<int> maxInWindows(const vector<int>& num, unsigned int size);

//����min������ջ
static stack<int> g_dataStack{}, g_minStack{};
void push(int value);
void pop();
int top();
int min();

//ջ��ѹ�롢��������
bool IsPopOrder(vector<int> pushV, vector<int> popV);

//-----------------------------����-------------------------------
//��β��ͷ��ӡ����
vector<int> printListFromTailToHead(ListNode* head);

//�����е�����k�����
ListNode* FindKthToTail(ListNode* pListHead, unsigned int k);

//��ת����
ListNode* ReverseList(ListNode* pHead);

//�ϲ��������������
ListNode* Merge(ListNode* pHead1, ListNode* pHead2);

//��������ĸ���
RandomListNode* Clone(RandomListNode* pHead);

//��������ĵ�һ���������
ListNode* FindFirstCommonNode(ListNode* pHead1, ListNode* pHead2);

//�����л�����ڽ��
ListNode* EntryNodeOfLoop(ListNode* pHead);

//ɾ���������ظ��Ľ��
ListNode* deleteDuplication(ListNode* pHead);

//---------------------------��ѧ---------------------------

//쳲���������
int Fibonacci(int n);

//��ֵ�������η�
double Power(double base, int exponent);

//�����ǵ���Ϸ(ԲȦ�����ʣ�µ���)
int LastRemaining_Solution(int n, int m);

#endif  // POINTTOOFFER_H_
