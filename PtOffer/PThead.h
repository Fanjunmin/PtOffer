#pragma once
#include <bits/stdc++.h>
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
