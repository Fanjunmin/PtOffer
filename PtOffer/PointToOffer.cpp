#include "stdafx.h"
#include "PointToOffer.h"

//------------------------Array数组-----------------------

//二维数组中的查找
bool Find(int target, vector<vector<int>> array) {
  //从左下角开始查找，小于target上移，大于target右移
	if (array.empty() || array.at(0).empty()) return false;
	int row_len = array.size(), col_len = array[0].size();
	int i = row_len - 1, j = 0;
	while (i >= 0 && j < col_len) {
		if (array[i][j] == target) {
			return true;
		}
		else if (array[i][j] < target) {
			++j;		
		}
		else {
			--i;
		}
	}
	return false;
}

//旋转数组的最小数字
int minNumberInRotateArray(vector<int> rotateArray) {
	if (rotateArray.empty()) return 0;
	int index = 1;
	while (index < rotateArray.size()) {
		if (rotateArray[index - 1]  > rotateArray[index]) {
			break;
		}
		++index;
	}
	return rotateArray[index];
}

//调整数组顺序使奇数位于偶数前面
void reOrderArray(vector<int> &array) {
	//插入排序的思路
	if (array.empty() || array.size() == 1) return;
	for (int i = 1; i < array.size(); ++i) {
		int get = array[i];
		if ((get & 1) == 0) continue;
		int j = i - 1;
		while (j >= 0 && (array[j] & 1) == 0) {
			array[j + 1] = array[j];
			--j;
		}
		array[j + 1] = get;
	}
	return;
}

//数组中出现次数超过一半的数字
int MoreThanHalfNum_Solution(vector<int> numbers) {
	map<int, int> map_count;
	for (auto num : numbers) {
		++map_count[num];
	}
	for (auto num : numbers) {
		if (map_count[num] > numbers.size() / 2) {
			return num;
		}
	}
	return 0;
}

//连续子数组的最大和
int FindGreatestSumOfSubArray(vector<int> array) {
  //采用动态规划策略,F[i]表示array[0, i]中连续子数组的最大和
  // F[i] = max(array[i], F[i] + array[i])
	if (array.empty()) return 0;
	int max_sum = array[0], temp = array[0];
	for (int i = 1; i < array.size(); ++i) {
		temp = temp >= 0 ? temp + array[i] : array[i];
		max_sum = max(max_sum, temp);
	}
	return max_sum;
}

//把数组排成最小的数
string PrintMinNumber(vector<int> numbers) {
	if (numbers.empty())	return "";
	sort(numbers.begin(), numbers.end(), 
			[](int x, int y){ return to_string(x) + to_string(y) < to_string(y) + to_string(x); });
	string result = "";
	for (auto num : numbers) {
		result += to_string(num);
	}
	return result;
}

//数组中的逆序对
long mergeCount(vector<int> &data, int lo, int hi) {
  // data[lo, hi]
  if (hi == lo) return 0;
  int mid = (lo + hi) / 2;
  long leftCount = mergeCount(data, lo, mid); 
	long rightCount = mergeCount(data, mid + 1, hi);
	long countSum = leftCount + rightCount;
  int *temp = new int[hi - lo + 1];  //临时数组
  int i = mid, j = hi, k = hi - lo;
  while (i >= lo && j >= mid + 1) {
    if (data[i] > data[j]) {
			//分组的数据都是有序的
      //从末尾进行比较，如果前一个区间元素大于后一个区间，则存在(j-mid)个逆序对
      countSum += (j - mid);
      temp[k--] = data[i--];
    } else {
      temp[k--] = data[j--];
    }
  }
  //将剩余元素放入临时数组中,[lo, mid]和[mid+1, hi]中至少有一个区间为空
  for (; j >= mid + 1; --j) temp[k--] = data[j];
  for (; i >= lo; --i) temp[k--] = data[i];
  for (int l = lo; l <= hi; ++l) data[l] = temp[l - lo];
  delete[] temp;
  return countSum % 1000000007;
}
int InversePairs(vector<int> data) {
  //采用归并排序的原理：O(nlog n)
  if (data.empty()) return 0;
  return mergeCount(data, 0, data.size() - 1);
}

//数字在排序数组中出现的次数
int GetNumberOfK(vector<int> data, int k) {
  //利用STL标注库函数lower_bound()和upper_bound()
	auto first = lower_bound(data.begin(), data.end(), k);
	auto last = upper_bound(data.begin(), data.end(), k);
	return last - first;
}

//数组中只出现一次的数字
void FindNumsAppearOnce(vector<int> data, int *num1, int *num2) {
	//利用位运算性质:b^b = 0 a^0 = a
	//a = a^b^b = b^a^b = b^b^a
	if (data.empty()) return;
	int result = 0;
	for (auto d : data) {
		result ^= d;		//最终的结果result=num1^num2
	}
	result -= result & (result - 1); //分组依据：result中最低位的1
	//num1和num2在这最低位必然一个为1，一个为0这就是分组依据
	*num1 = 0;
	*num2 = 0;
	for (auto d : data) {
		if (d & result) *num1 ^= d;
		else *num2 ^= d;
	}
}

//数组中重复的数字
bool duplicate(int numbers[], int length, int *duplication) {
	map<int, int> dup_map;
	for (int i = 0; i < length; ++i) {
		++dup_map[numbers[i]];
		if (dup_map[numbers[i]] == 2) {
			*duplication = numbers[i];
			return true;
		}
	}
	return false;
}

//顺时针打印矩阵
vector<int> printMatrix(vector<vector<int>> matrix) {
	if (matrix.empty()) return {};
	int m = matrix.size(), n = matrix[0].size();
	vector<int> result(m * n);
	int i = 0, j = 0, index = 0;
	while (true) {
		for (int k = j; k < n; ++k) result[index++] = matrix[i][k];
		if (++i >= m) break;
		for (int k = i; k < m; ++k) result[index++] = matrix[k][n - 1];
		if (j >= --n) break;
		for (int k = n - 1; k >= j; --k) result[index++] = matrix[m - 1][k];
		if (i >= --m) break;
		for (int k = m - 1; k >= i; --k) result[index++] = matrix[k][j];
		if (++j >= n) break;
	}
	return result;
}

//和为S的连续正数序列
vector<vector<int>> FindContinuousSequence(int sum) {
	if (sum < 3) return {};
	vector<vector<int>> result;
	int lo = 1, hi = 2, S = lo + hi;
	int mid = (sum + 1) >> 1;
	while (lo <= mid && hi <= sum) {
		while (S < sum) {
			++hi;
			S += hi;
		}
		if (S == sum) {
			vector<int> vec(hi - lo + 1);
			iota(vec.begin(), vec.end(), lo);
			result.push_back(vec);
		}
		S -= lo;
		++lo;
	}
	return result;
}

//丑数
int GetUglyNumber_Solution(int index) {	
	if (index < 1) return 0;
	vector<int> ugly(index, 1);
	int i = 0, j = 0, k = 0;
	for (int pos = 1; pos < index; ++pos) {
		ugly[pos] = min(ugly[i] * 2, min(ugly[j] * 3, ugly[k] * 5));
		if (ugly[pos] == ugly[i] * 2) ++i;
		if (ugly[pos] == ugly[j] * 3) ++j;
		if (ugly[pos] == ugly[k] * 5) ++k;
	}
	return ugly[index - 1];
}

//和为S的两个数字
vector<int> FindNumbersWithSum(vector<int> array, int sum) {
	if (array.size() < 2) return {};
	vector<int> result;
	map<int, int> sum_map;
	int multi_value = INT_MAX;
	for (int i = 0; i < array.size(); ++i) {
		if (sum_map[sum - array[i]] >= 1 && (sum - array[i]) * array[i] <= multi_value) {
			result.resize(2);
			multi_value = (sum - array[i]) * array[i];
			result[0] = sum - array[i];
			result[1] = array[i];
		}
		++sum_map[array[i]];
	}
	return result;
}

//扑克牌顺子
bool IsContinuous(vector<int> numbers) {
	if (numbers.empty()) return false;
	if (numbers.size() == 1) return true;
	sort(numbers.begin(), numbers.end());
	int queen_count = 0, distance = 0;	//记录大小王的个数和非连续派之间的距离和
	for(int i = 0; i < numbers.size(); ++i) {
		if (numbers[i] == 0) {
			++queen_count;
		}
		else {
			if (i + 1 < numbers.size()) {
				if (numbers[i + 1] == numbers[i]) {
					return false;
				}
				else {
					distance += numbers[i + 1] - numbers[i] - 1;
				}
			}
		}
	}
	return distance <= queen_count;
}

//数据流中的中位数
void Insert(int num) {
  //必须满足：最大优先队列的top元素小于等于最小优先队列top元素
	++g_insert_count;
	if (g_insert_count & 1) {
		g_big_que.push(num);
		g_small_que.push(g_big_que.top());
		g_big_que.pop();
	}
	else {
    g_small_que.push(num);
    g_big_que.push(g_small_que.top());
    g_small_que.pop();
	}
}
double GetMedian() {
	if (g_big_que.size() == g_small_que.size()) {
		return (g_big_que.top() + g_small_que.top()) / 2.0;
	}
	else {
		return g_big_que.size() > g_small_que.size() ? g_big_que.top() : g_small_que.top();
	}
}

//最小的k个数
vector<int> GetLeastNumbers_Solution(vector<int> input, int k) {
	//使用STL库函数make_heap();
	int length = input.size();
	if (length < k) return {};
  auto iter = input.begin();
	for(; iter != input.begin() + k; ++iter) {
		make_heap(iter, input.end(), greater<int>());		//构建最小堆
	}
	return vector<int>(input.begin(), iter);
}


//---------------------------树--------------------------------

//重建二叉树
TreeNode *reConstructBinaryTree(vector<int>::iterator pre_first, 
																vector<int>::iterator pre_last,
                                vector<int>::iterator vin_first,
                                vector<int>::iterator vin_last) {
	if (vin_last - vin_first != pre_last - pre_first ||
			pre_last == pre_first || vin_first == vin_last) {
		return nullptr;
	}
	TreeNode *curr_node = new TreeNode(*pre_first);
	if (pre_last == pre_first + 1) return curr_node;
	auto iter = vin_first;
	while (iter++ < vin_last) {
		if (*iter == *pre_first) break;
	}
	int len = iter - vin_first;
	curr_node->left = reConstructBinaryTree(pre_first + 1, 
																					pre_first + len + 1, vin_first, iter);
	curr_node->right = reConstructBinaryTree(pre_first + len + 1, 
																					 pre_last, iter + 1, vin_last);
	return curr_node;
}
TreeNode *reConstructBinaryTree(vector<int> pre, vector<int> vin) {
  return reConstructBinaryTree(pre.begin(), pre.end(), vin.begin(), vin.end());
}

//树的子结构
bool isSubtree(TreeNode *pRoot1, TreeNode *pRoot2) {
  //判断以pRoot2为根节点的树是否是以pRoot1为根节点的树的子树
	if (!pRoot2) return true;
	if (!pRoot1) return false;
	return pRoot1->val != pRoot2->val ? false :
				 isSubtree(pRoot1->left, pRoot2->left) &&
				 isSubtree(pRoot1->right, pRoot2->right);
}
bool HasSubtree(TreeNode *pRoot1, TreeNode *pRoot2) {
	if(!pRoot1 || !pRoot2) return false;
	return isSubtree(pRoot1, pRoot2) ||
				 isSubtree(pRoot1->left, pRoot2) ||
				 isSubtree(pRoot1->right, pRoot2);
}

//二叉树的镜像
void Mirror(TreeNode *pRoot) {
	if (pRoot && (pRoot->left || pRoot->right)) {
    std::swap(pRoot->left, pRoot->right);
		Mirror(pRoot->left);
    Mirror(pRoot->right);
	}
	return;
}

//从上往下打印二叉树
vector<int> PrintFromTopToBottom(TreeNode *root) {
	queue<TreeNode *> Q;
	vector<int> store;
	Q.push(root);
	while (!Q.empty()) {
		TreeNode *curr_node = Q.front();
		Q.pop();
		if (curr_node) {
			store.push_back(curr_node->val);
			Q.push(curr_node->left);
			Q.push(curr_node->right);
		}
	}
	return store;
}

//二叉搜索树的后序遍历序列
bool judgeBST(vector<int>::iterator first, vector<int>::iterator last) {
	if (last == first) return true;
	int root_val = *(last - 1);
	auto iter = first;
	while (iter < last - 1) {
		if (*iter > root_val) break;
		++iter;
	}
	auto temp = iter;
	while (iter < last - 1) {
		if (*iter  <= root_val) return false;
		++iter;
	}
	return judgeBST(first, temp) && judgeBST(temp, last - 1);
}
bool VerifySquenceOfBST(vector<int> sequence) {
	if (sequence.empty()) return false;
	return judgeBST(sequence.begin(), sequence.end());
}

//二叉树中和为某一值的路径
void FindPath(vector<vector<int>> &vec_store, 
							vector<int> store,
							TreeNode *root,
              int expNumber) {
	store.push_back(root->val);
	if (!root->left && !root->right) {
		if (expNumber == root->val) vec_store.push_back(store);
		return;
	}
	if (root->left) FindPath(vec_store, store, root->left, expNumber - root->val);
	if (root->right) FindPath(vec_store, store, root->right, expNumber - root->val);
	store.pop_back();		//回溯
}
vector<vector<int>> FindPath(TreeNode *root, int expectNumber) {
  vector<vector<int>> vec_store;
	vector<int> store;
	if(root) FindPath(vec_store, store, root, expectNumber);
	return vec_store;
}

//二叉搜索树与双向链表
void recurConvert(TreeNode *root, TreeNode *&pre) {
  if (!root) return;
  recurConvert(root->left, pre);
  root->left = pre;
  if (pre) pre->right = root;
  pre = root;
  recurConvert(root->right, pre);
}
TreeNode *Convert(TreeNode *pRootOfTree) {
  if (!pRootOfTree) return pRootOfTree;
  TreeNode *pre = 0;
  recurConvert(pRootOfTree, pre);
  TreeNode *res = pRootOfTree;
  while (res->left) res = res->left;
  return res;
}

//二叉树深度
int TreeDepth(TreeNode *pRoot) {
	//递归版本
  return pRoot ? 1 + max(TreeDepth(pRoot->left), 
												 TreeDepth(pRoot->right)) : 0;
}
int TreeDepth2(TreeNode *pRoot) {
	//迭代版本
  queue<TreeNode *> Q;
	int depth = 0;
  Q.push(pRoot);
	while (!Q.empty()) {
    int len = Q.size();
		++depth;
		while (len--) {
			TreeNode *curr_node = Q.front();
			Q.pop();
			if (curr_node) {
				Q.push(curr_node->left);
				Q.push(curr_node->right);
			}
		}
	}
	return depth - 1; //将叶节点的空孩子节点也算作一层了，所以减1
}

//平衡二叉树
bool IsBalanced(TreeNode *pRoot, int &pDepth) {
	if (!pRoot) {
		pDepth = 0;
		return true;
	}
	int left_depth, right_depth;	//记录左右子树的高度
	if (IsBalanced(pRoot->left, left_depth) && 
			IsBalanced(pRoot->right, right_depth)) {
		int diff = left_depth - right_depth;
		if (diff <= 1 && diff >= -1) {
			pDepth = 1 + (left_depth > right_depth ? left_depth : right_depth);
			return true;
		}
	}
	return false;
}
bool IsBalanced_Solution(TreeNode *pRoot) {
	int pDepth = 0;
	return IsBalanced(pRoot, pDepth);
}

//二叉树的下一个结点
TreeLinkNode *GetNext(TreeLinkNode *pNode) {
	if (!pNode) return pNode;
	if (pNode->right) {		//有右孩子的情况
		pNode = pNode->right;
		while (pNode->left) {
			pNode =  pNode->left;
		}
		return pNode;
	} 
	else {		//没有右孩子的情况
		TreeLinkNode *parent = pNode->next;
		if (!parent) {
			return parent;
		} 
		else {
			if (pNode == parent->left) {	//该节点是其父节点的左孩子
				return parent;
			} 
			else {	//该节点是其父节点的右孩子，沿左侧链上升
				while (parent->next && parent == parent->next->right) {
					parent = parent->next;
				}
				return parent->next;
			}
		}
	}
}

//对称的二叉树
bool isSymmetrical(TreeNode *leftChild, TreeNode *rightChild) {
	if (!leftChild && !rightChild) {
		//左右子树同时为空
		return true;
	}
	else if (leftChild && rightChild) {
		//左右子树都不为空
		return leftChild->val == rightChild->val &&
					 isSymmetrical(leftChild->left, rightChild->right) &&
					 isSymmetrical(leftChild->right, rightChild->left);
	}
	else {
		return false;
	}
}
bool isSymmetrical(TreeNode *pRoot) {
	if (!pRoot) return true; 
	return isSymmetrical(pRoot->left, pRoot->right);
}

//把二叉树打印成多行
vector<vector<int>> Print1(TreeNode *pRoot) {
	vector<vector<int>> store;
	queue<TreeNode *> Q;
	Q.push(pRoot);
	int index = 0;
	while (!Q.empty()) {
		int length = Q.size();
		store.push_back(vector<int>());
		while (length--) {
			TreeNode *curr_node = Q.front();
			Q.pop();
			if (curr_node) {
				store[index].push_back(curr_node->val);
				Q.push(curr_node->left);
				Q.push(curr_node->right);
			}
		}
		++index;
	}
	store.pop_back();
	return store;
}

//按之字形顺序打印二叉树
vector<vector<int>> Print2(TreeNode *pRoot) {
	if (!pRoot) return {};
	vector<vector<int>> result;
	stack<TreeNode *> odd, even;
	even.push(pRoot);		//从第零行开始
	while (!even.empty() || !odd.empty()) {
		vector<int> line;
		if (odd.empty()) {
			while (!even.empty()) {
				TreeNode *curr_node = even.top();
				even.pop();
				line.push_back(curr_node->val);
				if (curr_node->left) odd.push(curr_node->left);
				if (curr_node->right) odd.push(curr_node->right);	//注意，先左后右
			}
		}
		else {
			while (!odd.empty()) {
				TreeNode *curr_node = odd.top();
				odd.pop();
				line.push_back(curr_node->val);
				if (curr_node->right) even.push(curr_node->right);
				if (curr_node->left) even.push(curr_node->left);		//注意，先右后左
			}			
		}
		result.push_back(line);
	}
	return result;
}

//二叉搜索树的第k个结点
void inOrderTraversal(TreeNode *pRoot, vector<TreeNode *> &store) {
  if (!pRoot) return;
  inOrderTraversal(pRoot->left, store);
  store.push_back(pRoot);
  inOrderTraversal(pRoot->right, store);
}
TreeNode *KthNode(TreeNode *pRoot, int k) {
  vector<TreeNode *> store;
  inOrderTraversal(pRoot, store);
  return k > 0 && k <= store.size() ? store[k - 1] : nullptr;
}

//序列化二叉树
//采用先序遍历，用vector存储
void Serialize(vector<int> &sto, TreeNode *root) {
  if (!root)
    sto.push_back(0xffff);
  else {
    sto.push_back(root->val);
    Serialize(sto, root->left);
    Serialize(sto, root->right);
  }
}
char *Serialize(TreeNode *root) {
  vector<int> sto;
  Serialize(sto, root);
  int *res = new int[sto.size()];
  for (int i = 0; i < sto.size(); ++i) res[i] = sto[i];
  return (char *)res;
}
TreeNode *Deserialize(int *&str) {
  if (*str == 0xffff) {
    ++str;
    return NULL;
  }
  TreeNode *res = new TreeNode(*str);
  ++str;
  res->left = Deserialize(str);
  res->right = Deserialize(str);
  return res;
}
TreeNode *Deserialize(char *str) {
  int *p = (int *)str;
  return Deserialize(p);
}


//---------------------------回溯-------------------------------

//矩阵中的路径
bool hasPath(char *matrix, int rows, int cols, char *str);
bool hasPath(char *matrix, int rows, int cols, char *str, bool *flag, int x,
             int y, int index);

//机器人的运动范围
bool IsVaildVal(int x, int y, int threshold) {
  int result = 0;
  while (x) {
    result += x % 10;
    x /= 10;
  }
  while (y) {
    result += y % 10;
    y /= 10;
  }
  return result <= threshold;
}
int Find(vector<int> &id, int p) {
  if (p == -1 || id[p] == -1) return -1;
  int temp = p;
  while (p != id[p]) p = id[p];
  while (p != id[temp]) {
    int q = id[temp];
    id[temp] = p;
    temp = q;
  }
  return p;
}
void Union(vector<int> &id, vector<int> &sz, int p, int q) {
  int p_root = Find(id, p);
  int q_root = Find(id, q);
  if (p_root == q_root) return;
  if (sz[p_root] < sz[q_root]) {
    sz[q_root] += sz[p_root];
    id[p_root] = id[q_root];
  } else {
    sz[p_root] += sz[q_root];
    id[q_root] = id[p_root];
	}
	return;
}
int movingCount(int threshold, int rows, int cols) {
	if(rows <= 0 || cols <= 0) return 0;
  vector<int> id(rows * cols), sz(rows * cols, 1);
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      //将不满足坐标和的id值设为-1，否则设为其下标
      id[i * cols + j] = IsVaildVal(i, j, threshold) ? i * cols + j : -1;
    }
  }
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
			int pos = i * cols + j;
      if (id[pos] != -1 && i + 1 < rows && id[pos + cols] != -1)
				Union(id, sz, pos, pos + cols);
      if (id[pos] != -1 && j + 1 < cols && id[pos + 1] != -1)
				Union(id, sz, pos, pos + 1);
		}
	}
	int count = 0;
	int base_id = Find(id, 0);
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      int pos = i * cols + j;
      if (id[pos] != -1 && Find(id, pos) == base_id)
				++count;
		}
	}
	return count;
}

//---------------------------栈和队列-----------------------

//滑动窗口的最大值
vector<int> maxInWindows(const vector<int> &num, unsigned int size) {
  if (num.empty() || num.size() < size || size <= 0) return {};
  vector<int> re;
  deque<int> store;
  for (int i = 0; i < num.size(); ++i) {
    while (!store.empty() && num[store.back()] <= num[i])
      store.pop_back();  //保证队列首元素是当前窗口最大值下标
    while (!store.empty() && i + 1 - store.front() > size)
      store.pop_front();  //保证首元素在当前窗口范围
    store.push_back(i);
    if (i + 1 >= size) re.push_back(num[store.front()]);
  }
  return re;
}

//包含min函数的栈
void push(int value) {
  g_dataStack.push(value);
  if (g_minStack.empty() || g_minStack.top() >= value) g_minStack.push(value);
}
void pop() {
  if (g_dataStack.empty()) throw "stack is empty!";
  if (g_dataStack.top() == g_minStack.top()) g_minStack.pop();
  g_dataStack.pop();
}
int top() {
  if (g_dataStack.empty()) throw "stack is empty!";
  return g_dataStack.top();
}
int min() {
  if (g_minStack.empty()) throw "stack is empty!";
  return g_minStack.top();
}

//栈的压入、弹出序列
bool IsPopOrder(vector<int> pushV, vector<int> popV) {
  //使用栈进行模拟
  if (pushV.empty() || popV.size() != pushV.size()) return false;
  stack<int> simulate_stack;
  for (int i = 0, j = 0; i < pushV.size(); ++i) {
    simulate_stack.push(pushV[i]);
    while (!simulate_stack.empty() && popV[j] == simulate_stack.top()) {
      ++j;
      simulate_stack.pop();
    }
  }
  return simulate_stack.empty();
}

//---------------------------链表---------------------------

//从尾到头打印链表
vector<int> printListFromTailToHead(ListNode *head) {
  //非递归版本，使用栈
  vector<int> store_vec;
  stack<int> store_stack;
  while (head) {
    store_stack.push(head->val);
    head = head->next;
  }
  store_vec.reserve(store_stack.size());
  while (!store_stack.empty()) {
    store_vec.push_back(store_stack.top());
    store_stack.pop();
  }
  return store_vec;
}

//链表中倒数第k个结点
ListNode *FindKthToTail(ListNode *pListHead, unsigned int k) {
  //使用两个相距k的节点，当第二个节点到达尾部时，第一个节点即为倒数第k个结点
  ListNode *first = pListHead, *second = pListHead;
  while (k--) {
    if (!second) break;
    second = second->next;
  }
  if (!k) return nullptr;
  while (second) {
    first = first->next;
    second = second->next;
  }
  return first;
}

//反转链表
ListNode *ReverseList(ListNode *pHead) {
  //递归版本
  if (!pHead || !pHead->next) return pHead;
  ListNode *reverseHead = ReverseList(pHead->next);
  //反转过程
  pHead->next->next = pHead;
  pHead->next = nullptr;
  return reverseHead;
}
ListNode *ReverseList2(ListNode *pHead) {
  //迭代版本
  if (!pHead || !pHead->next) return pHead;
  ListNode *currNode = pHead;    //当前指针
  ListNode *reverseHead = NULL;  //新链表的头指针
  ListNode *preNode = NULL;      //当前指针的前一个结点

  while (currNode) {  //当前结点不为空时才执行
    ListNode *nextNode =
        currNode->next;  //链断开之前一定要保存断开位置后边的结点
    if (nextNode == NULL)  //当Next为空时，说明当前结点为尾节点
      reverseHead = currNode;
    currNode->next = preNode;  //指针反转
    preNode = currNode;
    currNode = nextNode;
  }
  return reverseHead;
}

//合并两个排序的链表
ListNode *Merge(ListNode *pHead1, ListNode *pHead2) {
  ListNode *result = new ListNode(0);
  ListNode *temp = result;
  while (pHead1 && pHead2) {
    if (pHead1->val < pHead2->val) {
      temp->next = pHead1;
      pHead1 = pHead1->next;
    } else {
      temp->next = pHead2;
      pHead2 = pHead2->next;
    }
    temp = temp->next;
  }
  temp->next = pHead1 ? pHead1 : pHead2;
  return result->next;
}

//复杂链表的复制
RandomListNode *Clone(RandomListNode *pHead) {
  if (!pHead) return pHead;
  RandomListNode *cloneNode = pHead;
  //复制每个节点到被复制节点的next
  while (cloneNode) {
    RandomListNode *copyNode = new RandomListNode(cloneNode->label);
    //将copyNode插入cloneNode之后
    copyNode->next = cloneNode->next;
    cloneNode->next = copyNode;
    cloneNode = copyNode->next;
  }
  cloneNode = pHead;
  while (cloneNode) {
    if (cloneNode->random)
      // random节点的复制
      //注意：random节点作为RandomListNode，在上一个while中也复制了
      cloneNode->next->random = cloneNode->random->next;
    cloneNode = cloneNode->next->next;
  }
  //进行拆分
  cloneNode = pHead->next;
  while (pHead->next) {
    RandomListNode *temp = pHead->next;
    pHead->next = temp->next;
    pHead = temp;
  }
  return cloneNode;
}

//两个链表的第一个公共结点
ListNode *FindFirstCommonNode(ListNode *pHead1, ListNode *pHead2) {
  ListNode *p1 = pHead1, *p2 = pHead2;
  //两个链表跑一趟后互换首节点再跑一次，必然会到达相交节点
  while (p1 != p2) {
    p1 = p1 ? p1->next : pHead2;
    p2 = p2 ? p2->next : pHead1;
  }
  return p1;
}

//链表中环的入口结点
ListNode *EntryNodeOfLoop(ListNode *pHead) {
  //快慢双指针
  if (!pHead) return pHead;
  ListNode *fast_node = pHead, *slow_node = pHead;
  while (fast_node && slow_node) {
    slow_node = slow_node->next;
    if (fast_node->next) {
      fast_node = fast_node->next->next;
    } else {
      return nullptr;  //这说明无环，有环必然next不会为空
    }
    if (fast_node == slow_node) break;
  }
  fast_node = pHead;
  while (fast_node != slow_node) {
    fast_node = fast_node->next;
    slow_node = slow_node->next;
  }
  return fast_node;
}

//删除链表中重复的结点
ListNode *deleteDuplication(ListNode *pHead) {
  if (!pHead || !pHead->next) return pHead;
  ListNode *current;
  if (pHead->next && pHead->next->val == pHead->val) {
    current = pHead->next->next;
    while (current && current->val == pHead->val) {
      current = current->next;
    }
    return deleteDuplication(current);
  } else {
    current = pHead->next;
    pHead->next = deleteDuplication(current);
    return pHead;
  }
}

//---------------------------数学---------------------------

//斐波那契数列
int Fibonacci(int n) {
  if (n < 0) throw "n is negetive";
  if (n == 0) return 0;
  int front = 0, back = 1;  //记录斐波那契数列的f(n)和f(n+1)
  while (--n) {
    back += front;
    front = back - front;
  }
  return back;
}

//数值的整数次方
double Power(double base, int exponent) {
  //位运算
  if (base == 0 && exponent == 0) throw "undefine 0^0==1";
  if (base == 0) return 0;
  if (exponent == 0) return 1;
  if (exponent < 0) return Power(1 / base, -exponent);
  double re = 1;
  while (exponent) {
    if (exponent & 1) re *= base;
    base *= base;
    exponent >>= 1;
  }
  return re;
}

//孩子们的游戏(圆圈中最后剩下的数)
int LastRemaining_Solution(int n, int m) {
  //递归问题f(n,m)
  // f(n,m) = (f(n-1, m) + m) % n
  // f(1,m) = 0
  if (n <= 0 || m <= 0)
    throw "input is non-positive";  //按照牛客网的要求不应该抛异常，应该输出-1
  int re = 0;
  for (int i = 2; i <= n; ++i) {
    re = (re + m) % i;
  }
  return re;
}