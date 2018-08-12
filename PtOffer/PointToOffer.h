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

//需要声明的一些数据结构以及初始化/释放函数
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

//------------------------Array数组-----------------------

//二维数组中的查找
bool Find(int target, vector<vector<int>> array);

//旋转数组的最小数字
int minNumberInRotateArray(vector<int> rotateArray);

//调整数组顺序使奇数位于偶数前面
void reOrderArray(vector<int>& array);

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
void FindNumsAppearOnce(vector<int> data, int* num1, int* num2);

//数组中重复的数字
bool duplicate(int numbers[], int length, int* duplication);

//顺时针打印矩阵
vector<int> printMatrix(vector<vector<int>> matrix);

//和为S的连续正数序列
vector<vector<int>> FindContinuousSequence(int sum);

//丑数
int GetUglyNumber_Solution(int index);

//和为S的两个数字
vector<int> FindNumbersWithSum(vector<int> array, int sum);

//扑克牌顺子
bool IsContinuous(vector<int> numbers);

//数据流中的中位数
static priority_queue<int> g_small_que;	//最小优先队列
static priority_queue<int, vector<int>, greater<int>> g_big_que;		//最大优先队列
static int g_insert_count = 0;		//插入个数
void Insert(int num);
double GetMedian();

//最小的k个数
vector<int> GetLeastNumbers_Solution(vector<int> input, int k);

//------------------------字符串-----------------------

//左旋转字符串
string LeftRotateString(string str, int n);

//表示数值的字符串
bool isNumeric(char* string);

//把字符串转换成整数
long power(int e, int m);
int StrToInt(string str);

//字符串的排列
void pBackTracking(set<string>& strSet, string& str, int beg);
vector<string> Permutation(string str);

//替换空格
void replaceSpace(char* str, int length);

//第一个只出现一次的字符
int FirstNotRepeatingChar(string str);

//正则表达式匹配
bool match(char* str, char* pattern);

//翻转单词顺序列
string ReverseSentence(string str);

//字符流中第一个不重复的字符
static list<char> dataList;
static map<char, int> countMap;
void Insert(char ch);
char FirstAppearingOnce();

//------------------------树-----------------------

//重建二叉树
TreeNode* reConstructBinaryTree(vector<int> pre, vector<int> vin);

//树的子结构
bool HasSubtree(TreeNode* pRoot1, TreeNode* pRoot2);
bool isSubtree(TreeNode* pRoot1, TreeNode* pRoot2);

//二叉树的镜像
void Mirror(TreeNode* pRoot);

//从上往下打印二叉树
vector<int> PrintFromTopToBottom(TreeNode* root);

//二叉搜索树的后序遍历序列
bool judgeBST(vector<int>::iterator first, vector<int>::iterator last);
bool VerifySquenceOfBST(vector<int> sequence);

//二叉树中和为某一值的路径
void FindPath(vector<vector<int>>& vec_store,
							vector<int> store,
							TreeNode* root,
              int expNumber);
vector<vector<int>> FindPath(TreeNode* root, int expectNumber);

//二叉搜索树与双向链表
void recurConvert(TreeNode* root, TreeNode*& pre);
TreeNode* Convert(TreeNode* pRootOfTree);

//二叉树深度
int TreeDepth(TreeNode* pRoot);

//平衡二叉树
bool IsBalanced(TreeNode* pRoot, int& pDepth);
bool IsBalanced_Solution(TreeNode* pRoot);

//二叉树的下一个结点
TreeLinkNode* GetNext(TreeLinkNode* pNode);

//对称的二叉树
bool isSymmetrical(TreeNode* lChild, TreeNode* rChild);
bool isSymmetrical(TreeNode* pRoot);

//把二叉树打印成多行
vector<vector<int>> Print1(TreeNode* pRoot);

//按之字形顺序打印二叉树
vector<vector<int>> Print2(TreeNode* pRoot);

//二叉搜索树的第k个结点
void inOrderTraversal(TreeNode* pRoot, vector<TreeNode *>& store);
TreeNode* KthNode(TreeNode* pRoot, int k);

//序列化二叉树
//采用先序遍历，用vector存储
void Serialize(vector<int>& sto, TreeNode* root);
char* Serialize(TreeNode* root);
TreeNode* Deserialize(int*& str);
TreeNode* Deserialize(char* str);

//---------------------------回溯-------------------------------

//矩阵中的路径
bool hasPath(char* matrix, int rows, int cols, char* str);
bool hasPath(char* matrix, int rows, int cols, char* str, bool* flag, int x,
             int y, int index);

//机器人的运动范围
bool IsVaildVal(int x, int y, int threshold);
int Find(vector<int> &id, int p);
void Union(vector<int>& id, vector<int>& sz, int p, int q);
int movingCount(int threshold, int rows, int cols);

//---------------------------栈和队列--------------------------

//滑动窗口的最大值
vector<int> maxInWindows(const vector<int>& num, unsigned int size);

//包含min函数的栈
static stack<int> g_dataStack{}, g_minStack{};
void push(int value);
void pop();
int top();
int min();

//栈的压入、弹出序列
bool IsPopOrder(vector<int> pushV, vector<int> popV);

//-----------------------------链表-------------------------------
//从尾到头打印链表
vector<int> printListFromTailToHead(ListNode* head);

//链表中倒数第k个结点
ListNode* FindKthToTail(ListNode* pListHead, unsigned int k);

//反转链表
ListNode* ReverseList(ListNode* pHead);

//合并两个排序的链表
ListNode* Merge(ListNode* pHead1, ListNode* pHead2);

//复杂链表的复制
RandomListNode* Clone(RandomListNode* pHead);

//两个链表的第一个公共结点
ListNode* FindFirstCommonNode(ListNode* pHead1, ListNode* pHead2);

//链表中环的入口结点
ListNode* EntryNodeOfLoop(ListNode* pHead);

//删除链表中重复的结点
ListNode* deleteDuplication(ListNode* pHead);

//---------------------------数学---------------------------

//斐波那契数列
int Fibonacci(int n);

//数值的整数次方
double Power(double base, int exponent);

//孩子们的游戏(圆圈中最后剩下的数)
int LastRemaining_Solution(int n, int m);

#endif  // POINTTOOFFER_H_
