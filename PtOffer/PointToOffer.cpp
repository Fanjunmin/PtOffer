#include "stdafx.h"
#include "PointToOffer.h"

//------------------------Array����-----------------------

//��ά�����еĲ���
bool Find(int target, vector<vector<int>> array) {
  //�����½ǿ�ʼ���ң�С��target���ƣ�����target����
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

//��ת�������С����
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

//��������˳��ʹ����λ��ż��ǰ��
void reOrderArray(vector<int> &array) {
	//���������˼·
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

//�����г��ִ�������һ�������
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

//���������������
int FindGreatestSumOfSubArray(vector<int> array) {
  //���ö�̬�滮����,F[i]��ʾarray[0, i]�����������������
  // F[i] = max(array[i], F[i] + array[i])
	if (array.empty()) return 0;
	int max_sum = array[0], temp = array[0];
	for (int i = 1; i < array.size(); ++i) {
		temp = temp >= 0 ? temp + array[i] : array[i];
		max_sum = max(max_sum, temp);
	}
	return max_sum;
}

//�������ų���С����
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

//�����е������
long mergeCount(vector<int> &data, int lo, int hi) {
  // data[lo, hi]
  if (hi == lo) return 0;
  int mid = (lo + hi) / 2;
  long leftCount = mergeCount(data, lo, mid); 
	long rightCount = mergeCount(data, mid + 1, hi);
	long countSum = leftCount + rightCount;
  int *temp = new int[hi - lo + 1];  //��ʱ����
  int i = mid, j = hi, k = hi - lo;
  while (i >= lo && j >= mid + 1) {
    if (data[i] > data[j]) {
			//��������ݶ��������
      //��ĩβ���бȽϣ����ǰһ������Ԫ�ش��ں�һ�����䣬�����(j-mid)�������
      countSum += (j - mid);
      temp[k--] = data[i--];
    } else {
      temp[k--] = data[j--];
    }
  }
  //��ʣ��Ԫ�ط�����ʱ������,[lo, mid]��[mid+1, hi]��������һ������Ϊ��
  for (; j >= mid + 1; --j) temp[k--] = data[j];
  for (; i >= lo; --i) temp[k--] = data[i];
  for (int l = lo; l <= hi; ++l) data[l] = temp[l - lo];
  delete[] temp;
  return countSum % 1000000007;
}
int InversePairs(vector<int> data) {
  //���ù鲢�����ԭ��O(nlog n)
  if (data.empty()) return 0;
  return mergeCount(data, 0, data.size() - 1);
}

//���������������г��ֵĴ���
int GetNumberOfK(vector<int> data, int k) {
  //����STL��ע�⺯��lower_bound()��upper_bound()
	auto first = lower_bound(data.begin(), data.end(), k);
	auto last = upper_bound(data.begin(), data.end(), k);
	return last - first;
}

//������ֻ����һ�ε�����
void FindNumsAppearOnce(vector<int> data, int *num1, int *num2) {
	//����λ��������:b^b = 0 a^0 = a
	//a = a^b^b = b^a^b = b^b^a
	if (data.empty()) return;
	int result = 0;
	for (auto d : data) {
		result ^= d;		//���յĽ��result=num1^num2
	}
	result -= result & (result - 1); //�������ݣ�result�����λ��1
	//num1��num2�������λ��Ȼһ��Ϊ1��һ��Ϊ0����Ƿ�������
	*num1 = 0;
	*num2 = 0;
	for (auto d : data) {
		if (d & result) *num1 ^= d;
		else *num2 ^= d;
	}
}

//�������ظ�������
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

//˳ʱ���ӡ����
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

//��ΪS��������������
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

//����
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

//��ΪS����������
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

//�˿���˳��
bool IsContinuous(vector<int> numbers) {
	if (numbers.empty()) return false;
	if (numbers.size() == 1) return true;
	sort(numbers.begin(), numbers.end());
	int queen_count = 0, distance = 0;	//��¼��С���ĸ����ͷ�������֮��ľ����
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

//�������е���λ��
void Insert(int num) {
  //�������㣺������ȶ��е�topԪ��С�ڵ�����С���ȶ���topԪ��
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

//��С��k����
vector<int> GetLeastNumbers_Solution(vector<int> input, int k) {
	//ʹ��STL�⺯��make_heap();
	int length = input.size();
	if (length < k) return {};
  auto iter = input.begin();
	for(; iter != input.begin() + k; ++iter) {
		make_heap(iter, input.end(), greater<int>());		//������С��
	}
	return vector<int>(input.begin(), iter);
}


//---------------------------��--------------------------------

//�ؽ�������
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

//�����ӽṹ
bool isSubtree(TreeNode *pRoot1, TreeNode *pRoot2) {
  //�ж���pRoot2Ϊ���ڵ�����Ƿ�����pRoot1Ϊ���ڵ����������
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

//�������ľ���
void Mirror(TreeNode *pRoot) {
	if (pRoot && (pRoot->left || pRoot->right)) {
    std::swap(pRoot->left, pRoot->right);
		Mirror(pRoot->left);
    Mirror(pRoot->right);
	}
	return;
}

//�������´�ӡ������
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

//�����������ĺ����������
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

//�������к�Ϊĳһֵ��·��
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
	store.pop_back();		//����
}
vector<vector<int>> FindPath(TreeNode *root, int expectNumber) {
  vector<vector<int>> vec_store;
	vector<int> store;
	if(root) FindPath(vec_store, store, root, expectNumber);
	return vec_store;
}

//������������˫������
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

//���������
int TreeDepth(TreeNode *pRoot) {
	//�ݹ�汾
  return pRoot ? 1 + max(TreeDepth(pRoot->left), 
												 TreeDepth(pRoot->right)) : 0;
}
int TreeDepth2(TreeNode *pRoot) {
	//�����汾
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
	return depth - 1; //��Ҷ�ڵ�Ŀպ��ӽڵ�Ҳ����һ���ˣ����Լ�1
}

//ƽ�������
bool IsBalanced(TreeNode *pRoot, int &pDepth) {
	if (!pRoot) {
		pDepth = 0;
		return true;
	}
	int left_depth, right_depth;	//��¼���������ĸ߶�
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

//����������һ�����
TreeLinkNode *GetNext(TreeLinkNode *pNode) {
	if (!pNode) return pNode;
	if (pNode->right) {		//���Һ��ӵ����
		pNode = pNode->right;
		while (pNode->left) {
			pNode =  pNode->left;
		}
		return pNode;
	} 
	else {		//û���Һ��ӵ����
		TreeLinkNode *parent = pNode->next;
		if (!parent) {
			return parent;
		} 
		else {
			if (pNode == parent->left) {	//�ýڵ����丸�ڵ������
				return parent;
			} 
			else {	//�ýڵ����丸�ڵ���Һ��ӣ������������
				while (parent->next && parent == parent->next->right) {
					parent = parent->next;
				}
				return parent->next;
			}
		}
	}
}

//�ԳƵĶ�����
bool isSymmetrical(TreeNode *leftChild, TreeNode *rightChild) {
	if (!leftChild && !rightChild) {
		//��������ͬʱΪ��
		return true;
	}
	else if (leftChild && rightChild) {
		//������������Ϊ��
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

//�Ѷ�������ӡ�ɶ���
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

//��֮����˳���ӡ������
vector<vector<int>> Print2(TreeNode *pRoot) {
	if (!pRoot) return {};
	vector<vector<int>> result;
	stack<TreeNode *> odd, even;
	even.push(pRoot);		//�ӵ����п�ʼ
	while (!even.empty() || !odd.empty()) {
		vector<int> line;
		if (odd.empty()) {
			while (!even.empty()) {
				TreeNode *curr_node = even.top();
				even.pop();
				line.push_back(curr_node->val);
				if (curr_node->left) odd.push(curr_node->left);
				if (curr_node->right) odd.push(curr_node->right);	//ע�⣬�������
			}
		}
		else {
			while (!odd.empty()) {
				TreeNode *curr_node = odd.top();
				odd.pop();
				line.push_back(curr_node->val);
				if (curr_node->right) even.push(curr_node->right);
				if (curr_node->left) even.push(curr_node->left);		//ע�⣬���Һ���
			}			
		}
		result.push_back(line);
	}
	return result;
}

//�����������ĵ�k�����
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

//���л�������
//���������������vector�洢
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


//---------------------------����-------------------------------

//�����е�·��
bool hasPath(char *matrix, int rows, int cols, char *str);
bool hasPath(char *matrix, int rows, int cols, char *str, bool *flag, int x,
             int y, int index);

//�����˵��˶���Χ
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
      //������������͵�idֵ��Ϊ-1��������Ϊ���±�
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

//---------------------------ջ�Ͷ���-----------------------

//�������ڵ����ֵ
vector<int> maxInWindows(const vector<int> &num, unsigned int size) {
  if (num.empty() || num.size() < size || size <= 0) return {};
  vector<int> re;
  deque<int> store;
  for (int i = 0; i < num.size(); ++i) {
    while (!store.empty() && num[store.back()] <= num[i])
      store.pop_back();  //��֤������Ԫ���ǵ�ǰ�������ֵ�±�
    while (!store.empty() && i + 1 - store.front() > size)
      store.pop_front();  //��֤��Ԫ���ڵ�ǰ���ڷ�Χ
    store.push_back(i);
    if (i + 1 >= size) re.push_back(num[store.front()]);
  }
  return re;
}

//����min������ջ
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

//ջ��ѹ�롢��������
bool IsPopOrder(vector<int> pushV, vector<int> popV) {
  //ʹ��ջ����ģ��
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

//---------------------------����---------------------------

//��β��ͷ��ӡ����
vector<int> printListFromTailToHead(ListNode *head) {
  //�ǵݹ�汾��ʹ��ջ
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

//�����е�����k�����
ListNode *FindKthToTail(ListNode *pListHead, unsigned int k) {
  //ʹ���������k�Ľڵ㣬���ڶ����ڵ㵽��β��ʱ����һ���ڵ㼴Ϊ������k�����
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

//��ת����
ListNode *ReverseList(ListNode *pHead) {
  //�ݹ�汾
  if (!pHead || !pHead->next) return pHead;
  ListNode *reverseHead = ReverseList(pHead->next);
  //��ת����
  pHead->next->next = pHead;
  pHead->next = nullptr;
  return reverseHead;
}
ListNode *ReverseList2(ListNode *pHead) {
  //�����汾
  if (!pHead || !pHead->next) return pHead;
  ListNode *currNode = pHead;    //��ǰָ��
  ListNode *reverseHead = NULL;  //�������ͷָ��
  ListNode *preNode = NULL;      //��ǰָ���ǰһ�����

  while (currNode) {  //��ǰ��㲻Ϊ��ʱ��ִ��
    ListNode *nextNode =
        currNode->next;  //���Ͽ�֮ǰһ��Ҫ����Ͽ�λ�ú�ߵĽ��
    if (nextNode == NULL)  //��NextΪ��ʱ��˵����ǰ���Ϊβ�ڵ�
      reverseHead = currNode;
    currNode->next = preNode;  //ָ�뷴ת
    preNode = currNode;
    currNode = nextNode;
  }
  return reverseHead;
}

//�ϲ��������������
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

//��������ĸ���
RandomListNode *Clone(RandomListNode *pHead) {
  if (!pHead) return pHead;
  RandomListNode *cloneNode = pHead;
  //����ÿ���ڵ㵽�����ƽڵ��next
  while (cloneNode) {
    RandomListNode *copyNode = new RandomListNode(cloneNode->label);
    //��copyNode����cloneNode֮��
    copyNode->next = cloneNode->next;
    cloneNode->next = copyNode;
    cloneNode = copyNode->next;
  }
  cloneNode = pHead;
  while (cloneNode) {
    if (cloneNode->random)
      // random�ڵ�ĸ���
      //ע�⣺random�ڵ���ΪRandomListNode������һ��while��Ҳ������
      cloneNode->next->random = cloneNode->random->next;
    cloneNode = cloneNode->next->next;
  }
  //���в��
  cloneNode = pHead->next;
  while (pHead->next) {
    RandomListNode *temp = pHead->next;
    pHead->next = temp->next;
    pHead = temp;
  }
  return cloneNode;
}

//��������ĵ�һ���������
ListNode *FindFirstCommonNode(ListNode *pHead1, ListNode *pHead2) {
  ListNode *p1 = pHead1, *p2 = pHead2;
  //����������һ�˺󻥻��׽ڵ�����һ�Σ���Ȼ�ᵽ���ཻ�ڵ�
  while (p1 != p2) {
    p1 = p1 ? p1->next : pHead2;
    p2 = p2 ? p2->next : pHead1;
  }
  return p1;
}

//�����л�����ڽ��
ListNode *EntryNodeOfLoop(ListNode *pHead) {
  //����˫ָ��
  if (!pHead) return pHead;
  ListNode *fast_node = pHead, *slow_node = pHead;
  while (fast_node && slow_node) {
    slow_node = slow_node->next;
    if (fast_node->next) {
      fast_node = fast_node->next->next;
    } else {
      return nullptr;  //��˵���޻����л���Ȼnext����Ϊ��
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

//ɾ���������ظ��Ľ��
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

//---------------------------��ѧ---------------------------

//쳲���������
int Fibonacci(int n) {
  if (n < 0) throw "n is negetive";
  if (n == 0) return 0;
  int front = 0, back = 1;  //��¼쳲��������е�f(n)��f(n+1)
  while (--n) {
    back += front;
    front = back - front;
  }
  return back;
}

//��ֵ�������η�
double Power(double base, int exponent) {
  //λ����
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

//�����ǵ���Ϸ(ԲȦ�����ʣ�µ���)
int LastRemaining_Solution(int n, int m) {
  //�ݹ�����f(n,m)
  // f(n,m) = (f(n-1, m) + m) % n
  // f(1,m) = 0
  if (n <= 0 || m <= 0)
    throw "input is non-positive";  //����ţ������Ҫ��Ӧ�����쳣��Ӧ�����-1
  int re = 0;
  for (int i = 2; i <= n; ++i) {
    re = (re + m) % i;
  }
  return re;
}