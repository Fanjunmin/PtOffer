#include "stdafx.h"
#include "PTHead.h"

/*********************************************************/
/*------------------------Array����-----------------------*/
/**********************************************************/

//��ά�����еĲ���
bool Find(int target, vector<vector<int> > array) {
	//�����½ǿ�ʼ���ң�С��target���ƣ�����target����
	int rowLen = array.size();
	int colLen = array.empty() ? 0 : array[0].size();
	int i = rowLen - 1, j = 0;
	while (i >= 0 && j < colLen) {
		if (array[i][j] == target)
			return true;
		else if (array[i][j] < target)
			++j;
		else
			--i;
	}
	return false;
}

//��ת�������С����
/*˳����Ұ汾O(n)
int minNumberInRotateArray(vector<int> rotateArray) {
if(rotateArray.empty())
return 0;
int i= 1;
for(; i < rotateArray.size(); ++i) {
if(rotateArray[i] < rotateArray[i - 1])
break;
}
return rotateArray[i];
}*/
int minNumberInRotateArray(vector<int> rotateArray) {
	//���ֲ��Ұ汾O(log n)~O(n)
	if (rotateArray.empty())
		return 0;
	int left = 0, mid = 0, right = rotateArray.size() - 1;
	while (rotateArray[left] >= rotateArray[right]) {
		//��һ��Ԫ���ϸ�С�����һ��Ԫ����˵��û�з�����ת
		if (left + 1 == right) {
			mid = right;
			break;
		}
		mid = (left + right) / 2;
		if (rotateArray[mid] == rotateArray[left] && rotateArray[mid] == rotateArray[right]) {
			//��������������������ȫ���,���ʱ����Ҫ˳����ҵ�һ��С�������������
			for (int i = left + 1; i < right; ++i) {
				if (rotateArray[i] < rotateArray[mid]) {
					return rotateArray[i];
				}
			}
		}
		if (rotateArray[mid] >= rotateArray[left]) {
			left = mid;
		}
		else if (rotateArray[mid] <= rotateArray[right]) {
			right = mid;
		}
	}
	return rotateArray[mid];
}

//��������˳��ʹ����λ��ż��ǰ��
/*
//ʹ��STL��׼�⺯��partition()
void reOrderArray(vector<int> &array) {
stable_partition(array.begin(), array.end(), [](int x){return x % 2 == 1;});
}
*/
void reOrderArray(vector<int> &array) {
	//������ð������(ð���������ȶ�������)��ǰż����ͽ�������
	int len = array.size();
	for (int i = 0; i < len - 1; ++i) {
		for (int j = 0; j < len - i - 1; ++j) {
			if ((array[j] % 2 == 0) && (array[j + 1] % 2 == 1))
				swap(array[j], array[j + 1]);
		}
	}
}

//�����г��ִ�������һ�������
int MoreThanHalfNum_Solution(vector<int> numbers) {
	//ʹ��map
	typedef map<int, int> intIntMap;
	typedef pair<int, int> intIntPair;
	intIntMap m;
	for (auto number : numbers)
		++m[number];
	auto iter = max_element(m.begin(), m.end(), [](intIntPair a, intIntPair b) {return a.second < b.second; });
	return iter->second > numbers.size() / 2 ? iter->first : 0;
}

//���������������
int FindGreatestSumOfSubArray(vector<int> array) {
	//���ö�̬�滮����,F[i]��ʾarray[0, i]�����������������
	//F[i] = max(array[i], F[i] + array[i])
	if (array.empty()) return 0;
	int MaxSum = array[0], temp = array[0];
	for (int i = 1; i < array.size(); ++i) {
		temp = temp >= 0 ? temp + array[i] : array[i];
		MaxSum = MaxSum >= temp ? MaxSum : temp;
	}
	return MaxSum;
}

//�������ų���С����
string PrintMinNumber(vector<int> numbers) {
	//����to_string()ת��Ϊstring����
	//������������string x,y ����x+y��y+x�ıȽϽ�������
	if (numbers.empty()) return "";
	vector<string> svec(numbers.size());
	transform(numbers.begin(), numbers.end(), svec.begin(), [](int x) { return to_string(x); });
	sort(svec.begin(), svec.end(), [](string x, string y) { return (x + y) < (y + x); });
	for (int i = 1; i < svec.size(); ++i) {
		svec[0] += svec[i];
	}
	return svec[0];
}

//�����е������
/*
//brute force ��ʱ
int InversePairs(vector<int> data) {
if(data.empty()) return 0;
const int p = 1000000007;
int sum = 0;
for(int i = 0; i < data.size(); ++i) {
for(int j = 0; j < data.size(); ++j) {
if(i < j && data[i] > data[j]) {
sum += 1;
sum %= p;
}
}
}
return sum;
}*/
long mergeCount(vector<int>& data, int lo, int hi) {
	//data[lo, hi]
	if (hi == lo) return 0;
	int mid = (lo + hi) / 2;
	long leftCount = mergeCount(data, lo, mid), rightCount = mergeCount(data, mid + 1, hi), countSum = 0;
	int* temp = new int[hi - lo + 1];    //��ʱ����
	int i = mid, j = hi, k = hi - lo;
	while (i >= lo && j >= mid + 1) {
		if (data[i] > data[j]) {
			//��ĩβ���бȽϣ����ǰһ������Ԫ�ش��ں�һ�����䣬�����(j - mid)�������
			countSum += (j - mid);
			temp[k--] = data[i--];
		}
		else {
			temp[k--] = data[j--];
		}
	}
	//��ʣ��Ԫ�ط�����ʱ������,[lo, mid]��[mid+1, hi]��������һ������Ϊ��
	for (; j >= mid + 1; --j)    temp[k--] = data[j];
	for (; i >= lo; --i)  temp[k--] = data[i];
	for (int l = lo; l <= hi; ++l)  data[l] = temp[l - lo];
	delete[]temp;
	return (countSum + leftCount + rightCount) % 1000000007;
}
int InversePairs(vector<int> data) {
	//���ù鲢�����ԭ��O(nlog n)
	if (data.empty()) return 0;
	return mergeCount(data, 0, data.size() - 1);
}

//���������������г��ֵĴ���
/*
//ʹ��STL�㷨count ѭ�����O(n)
int GetNumberOfK(vector<int> data, int k) {
return count(data.begin(), data.end(), k);
}
//����STL��multimap�����ײ��Ժ����Ϊ����,����ɱ�O(n),��ѯ�ɱ�O(log n)
int GetNumberOfK(vector<int> data, int k) {
multiset<int> msData(data.begin(), data.end());
return msData.count(k);
}
//����STL�⺯��lower_bound()��upperBound(),O(log n)
int GetNumberOfK(vector<int> data ,int k) {
auto iter1 = lower_bound(data.begin(), data.end(), k);
auto iter2 = upper_bound(data.begin(), data.end(), k);
return static_cast<int>(iter2 - iter1);
}
*/
int GetNumberOfK(vector<int> data, int k) {
	//���ֲ��ҷǵݹ�汾
	auto iter1 = lower_bound(data.begin(), data.end(), k);
	auto iter2 = upper_bound(data.begin(), data.end(), k);
	return static_cast<int>(iter2 - iter1);
}

//������ֻ����һ�ε�����
/*
//����ɢ�б�map
void FindNumsAppearOnce(vector<int> data,int* num1,int *num2) {
map<int, int> myMap;
for(auto d : data) ++myMap[d];
int s[2], i = 0;
for(auto mm : myMap) {
if(mm.second == 1)
s[i++] = mm.first;
}
*num1 = s[0];
*num2 = s[1];
}*/
void FindNumsAppearOnce(vector<int> data, int* num1, int *num2) {
	//����λ����^���� a^b^b = b^a^b = a
	if (!data.empty()) {
		int result = 0;
		for (auto d : data) result ^= d; //d = num1 * num2
		int n = result - (result & (result - 1));    //�����λ��1�ڵ�kλ��������Ϊ0
		*num1 = *num2 = 0;
		for (auto d : data) {
			//����data�еĳ������ε�������ͬ�����ĵ�kλ��Ȼ��ͬ
			//��������:�������n��λ���Ƿ�Ϊ0
			if (d & n) *num1 ^= d;
			else *num2 ^= d;
		}
	}
}

//�������ظ�������
bool duplicate(int numbers[], int length, int* duplication) {
	//����ɢ�б�map
	map<int, int> myMap;
	for (int i = 0; i < length; ++i) {
		++myMap[numbers[i]];
		if (myMap[numbers[i]] == 2) {
			*duplication = numbers[i];
			return true;
		}
	}
	return false;
}

//˳ʱ���ӡ����
vector<int> printMatrix(vector<vector<int> > matrix) {
	if (matrix.empty()) return {};
	int m = matrix.size(), n = matrix[0].size();
	vector<int> spiral(m * n);
	int  i = 0, j = 0, k = 0, index = 0;
	while (true) {
		for (int k = j; k < n; ++k) spiral[index++] = matrix[i][k];
		if (++i >= m) break;
		for (int k = i; k < m; ++k) spiral[index++] = matrix[k][n - 1];
		if (j >= --n) break;
		for (int k = n - 1; k >= j; --k) spiral[index++] = matrix[m - 1][k];
		if (i >= --m) break;
		for (int k = m - 1; k >= i; --k) spiral[index++] = matrix[k][j];
		if (++j >= n) break;
	}
	return spiral;
}

//��ΪS��������������
vector<vector<int> > FindContinuousSequence(int sum) {
	if (sum < 2) return {};
	vector<vector<int> > result;
	int lo = 1, hi = 2, mid = (sum + 1) >> 1, S = lo + hi;
	while (lo < mid && hi < sum) {
		while (sum < S) {
			S -= lo;
			++lo;
		}
		if (sum == S) {
			vector<int> vec(hi - lo + 1);
			iota(vec.begin(), vec.end(), lo);
			result.push_back(vec);
		}
		++hi;
		S += hi;
	}
	return result;
}

//����
int GetUglyNumber_Solution(int index) {
	if (index < 1) return 0;
	vector<int> vec(index, 1);
	int i = 0, j = 0, k = 0;
	for (int pos = 1; pos < index; ++pos) {
		vec[pos] = std::min(vec[i] * 2, std::min(vec[j] * 3, vec[k] * 5));
		if (vec[pos] == vec[i] * 2) ++i;
		if (vec[pos] == vec[j] * 3) ++j;
		if (vec[pos] == vec[k] * 5) ++k;
	}
	return vec[index - 1];
}

//��ΪS����������
vector<int> FindNumbersWithSum(vector<int> array, int sum) {
	vector<int> result;
	if (array.size() < 2) return result;
	map<int, int> myMap;
	int multi = INT_MAX;
	for (int i = 0; i < array.size(); ++i) {
		if (myMap[sum - array[i]] >= 1 && array[i] * (sum - array[i]) <= multi) {
			result.resize(2);
			result[1] = array[i];
			result[0] = sum - array[i];
		}
		myMap[array[i]] = 1;
	}
	return result;
}

//�˿���˳��
bool IsContinuous(vector<int> numbers) {
	//O(nlog n)����
	if (numbers.empty()) return false;
	if (numbers.size() == 1) return true;
	sort(numbers.begin(), numbers.end());
	int zeroCount = 0, distanceCount = 0;
	for (int i = 0; i < numbers.size(); ++i) {
		if (numbers[i] == 0) ++zeroCount;
		else {
			if (i + 1 < numbers.size()) {
				if (numbers[i + 1] == numbers[i]) return false;
				else distanceCount += (numbers[i + 1] - numbers[i] - 1);
			}
		}
	}
	return distanceCount <= zeroCount;
}


//�������е���λ��
priority_queue<int> priQue1;    //��С���ȶ���
priority_queue<int, vector<int>, greater<int>> priQue2;  //������ȶ���
int insertCount = 0;
//�������㣺������ȶ��е�topԪ��С�ڵ�����С���ȶ���topԪ��
void Insert(int num) {
	++insertCount;
	if (insertCount % 2 == 0) {
		priQue1.push(num);
		priQue2.push(priQue1.top());
		priQue1.pop();
	}
	else {
		priQue2.push(num);
		priQue1.push(priQue2.top());
		priQue2.pop();
	}
}
double GetMedian() {
	if (priQue1.size() == priQue2.size()) return (priQue1.top() + priQue2.top()) / 2.0;
	else return priQue1.size() > priQue2.size() ? priQue1.top() : priQue2.top();
}

//��С��k����
vector<int> GetLeastNumbers_Solution(vector<int> input, int k) {
	int len = input.size();
	if (k > len) return {};
	for (int i = 0; i < k; ++i) {
		make_heap(input.begin(), input.end() - i, greater<int>());  //��С��
		std::swap(input[0], input[len - i - 1]);
	}
	return vector<int>(input.end() - k, input.end());
}

/************************************************************/
/*----------------------------�ַ���-------------------------*/
/************************************************************/

//����ת�ַ���
/*
//O(n)�ռ临�Ӷ�
string LeftRotateString(string str, int n) {
if(str.empty()) return str;
int len = str.size();
n %= len;
string re1(str.begin(), str.begin() + n), re2(str.begin() + n, str.end());
return re2 + re1;
}
//STL rotate()����,���ݲ�ͬ�ĵ��������ò�ͬ�ײ�ʵ��,string�ĵ�����Ϊ������ʵ�����
//forward iterator:����һ��һ����������
//bidirectional iterator:����reverse
//rand access iterator:���ÿ�ǰ�κͺ���������
string LeftRotateString(string str, int n) {
if(str.empty()) return str;
n %= str.size();
rotate(str.begin(), str.begin() + n, str.end());
return str;
}
*/
string LeftRotateString(string str, int n) {
	if (str.empty()) return str;
	int len = str.size();
	n %= len;
	int re = len - n, i = 0, j = n;
	while (n <= re) {
		for (; i < n; ++i, ++j) swap(str[i], str[j]);
		re -= n;
	}
	//cout << str << i << j << re << endl;
	while (re--) {
		swap(str[i++], str[j++]);
	}
	return str;
}

//��ʾ��ֵ���ַ���
bool isNumeric(char* string) {
	//�������еķǷ����
	if (string == NULL) return false;
	bool hasE = false, hasDot = false, hasSign = false; //��ʶ�Ƿ���E��С���㣬����λ
	int len = strlen(string);
	for (int i = 0; i < len; ++i) {
		if (string[i] == 'e' || string[i] == 'E') {
			if (i + 1 == len || hasE) return false;  //e�������������Ҳ��ܳ�����������e
			hasE = true;
		}
		else if (string[i] == '+' || string[i] == '-') {
			if (!hasSign && i > 0 && string[i - 1] != 'e' && string[i - 1] != 'E') {
				//��һ�γ��ַ���λ����������λ������e����һλ
				return false;
			}
			if (hasSign && string[i - 1] != 'e' && string[i - 1] != 'E') {
				//i ��Ȼ���� 0
				//�Ѿ����ֹ�����λ������ֵķ���λ������e����һλ
				return false;
			}
			hasSign = true;
		}
		else if (string[i] == '.') {
			if (hasDot || hasE) return false;    //С����ֻ�ܳ���һ�Σ�����e���沢���ܳ���С����
			hasDot = true;
		}
		else if (string[i] > '9' || string[i] < '0') {
			return false;
		}
	}
	return true;
}

//���ַ���ת��������
long power(int e, int m) {
	//����e^m
	return m == 0 ? 1 : (m % 2 ? power(e, m / 2) * e : power(e, m / 2));
}

int StrToInt(string str) {
	if (str.empty()) return 0;
	int hasE = 0, hasSign = 0;
	long coe = 0, exp = 0;
	for (int i = 0; i < str.size(); ++i) {
		if (i == 0 && (str[i] == '+' || str[i] == '-')) {
			//��λ������������
			if (str[0] == '+') hasSign = 1;
			else if (str[0] == '-') hasSign = -1;
		}
		else if (i != 0 && (str[i] == '+' || str[i] == '-')) return 0; //����λ���ַ���λ������0
		else if (str[i] == 'e' || str[i] == 'E') {
			//������ֶ��e����0
			if (hasE == 1) return 0;
			else hasE = 1;
		}
		else if (hasE == 0 && str[i] <= '9' && str[i] >= '0') {
			//ϵ�����ֵ���
			coe = (10 * coe + str[i] - '0');
			if ((hasSign == 0 || hasSign == 1) && coe > INT_MAX) return 0;
			else if (hasSign == -1 && -coe < INT_MIN) return 0;
		}
		else if (hasE == 1 && str[i] <= '9' && str[i] >= '0') {
			//ָ�����ֵ���
			exp = (10 * exp + str[i] - '0');
			coe *= power(10, exp);
			if ((hasSign == 0 || hasSign == 1) && coe > INT_MAX) return 0;
			else if (hasSign == -1 || -coe < INT_MIN) return 0;
		}
		else return 0;
	}
	if (hasSign == -1) return -coe;
	else return coe;
}

//�ַ���������
void pBackTracking(set<string>& strSet, string& str, int beg) {
	if (beg == str.size()) {
		strSet.insert(str);
		return;
	}
	for (int i = beg; i < str.size(); ++i) {
		swap(str[i], str[beg]);
		pBackTracking(strSet, str, beg + 1);
		swap(str[i], str[beg]);
	}
}

vector<string> Permutation(string str) {
	if (str.empty()) return {};
	set<string> strSet;
	pBackTracking(strSet, str, 0);
	vector<string> strVec(strSet.begin(), strSet.end());
	return strVec;
}

//�滻�ո�
void replaceSpace(char *str, int length) {
	//length �ַ�������
	if (str == NULL) return;
	int spaceLen = 0;
	for (int i = 0; str[i] != '\0'; ++i) {
		if (*(str + i) == ' ') ++spaceLen;
	}
	for (int j = strlen(str) - 1; j >= 0; --j) {
		int i = j + 2 * spaceLen;
		if (str[j] == ' ') {
			//���ո�����Ϊ"%20"
			str[i - 2] = '%';
			str[i - 1] = '2';
			str[i] = '0';
			--spaceLen;
		}
		else {
			str[i] = str[j];
		}
	}
}

//��һ��ֻ����һ�ε��ַ�
int FirstNotRepeatingChar(string str) {
	map<char, int> myMap;
	for (int i = 0; i < str.size(); ++i) {
		++myMap[str[i]];
	}
	for (int i = 0; i < str.size(); ++i) {
		if (myMap[str[i]] == 1)
			return i;
	}
	return -1;
}

//������ʽƥ��
bool match(char* str, char* pattern) {
	//ֱ��ʹ��cpp11��������ʽ
	//��̬�滮���ߵݹ������е���
	if (!str && !pattern) return false;
	regex re(pattern);
	return regex_match(str, re);
}

//��ת����˳����
string ReverseSentence(string str) {
	reverse(str.begin(), str.end());
	string::iterator firstIter = str.begin(), secondIter = str.begin();
	while (secondIter < str.end()) {
		if (*secondIter == ' ') {
			reverse(firstIter, secondIter);
			firstIter = secondIter + 1;
		}
		++secondIter;
	}
	reverse(firstIter, str.end());
	return str;
}


//�ַ����е�һ�����ظ����ַ�
list<char> dataList;
map<char, int> countMap;
void Insert(char ch) {
	//Insert one char from stringstream
	if (dataList.empty() || countMap[ch] == 0) {
		dataList.insert(dataList.end(), ch);
		countMap[ch] = 1;
	}
	else if (countMap[ch] == 1)
		dataList.remove(ch);
}
char FirstAppearingOnce() {
	//return the first appearence once char in current stringstream
	return dataList.empty() ? '#' : dataList.front();
}

/**********************************************************/
/*----------------------Tree----------------------------*/
/**********************************************************/

//�ؽ�������
TreeNode* reConstructBinaryTree(vector<int> pre, vector<int> vin) {
	if (pre.empty() || vin.empty() || pre.size() != vin.size())
		return NULL;
	if (pre.size() == 1) {
		TreeNode* node = new TreeNode(pre[0]);
		return node;
	}
	TreeNode* node = new TreeNode(pre[0]);
	int i = 0;
	for (; i < vin.size(); ++i) {
		if (vin[i] == pre[0])    //�ҵ�ͷ�ڵ�
			break;
	}
	//����ݹ�
	vector<int> leftPre(pre.begin() + 1, pre.begin() + i + 1);
	vector<int> rightPre(pre.begin() + i + 1, pre.end());
	vector<int> leftVin(vin.begin(), vin.begin() + i);
	vector<int> rightVin(vin.begin() + i + 1, vin.end());
	node->left = reConstructBinaryTree(leftPre, leftVin);
	node->right = reConstructBinaryTree(rightPre, rightVin);
	return node;
}

//�����ӽṹ
bool isSubtree(TreeNode* pRoot1, TreeNode* pRoot2) {
	//�ж�ͷ�ڵ�pRoot2���ڵ����Ƿ���ͷ�ڵ�pRoot1���ڵ���������
	if (!pRoot2) return true;
	if (!pRoot1) return false;
	return pRoot1->val != pRoot2->val ? false :
		isSubtree(pRoot1->left, pRoot2->left)
		&& isSubtree(pRoot1->right, pRoot2->right);
}

bool HasSubtree(TreeNode* pRoot1, TreeNode* pRoot2) {
	//Լ��������������һ�������ӽṹ
	if (!pRoot2 || !pRoot1) return false;
	return isSubtree(pRoot1, pRoot2)
		|| HasSubtree(pRoot1->left, pRoot2)
		|| HasSubtree(pRoot1->right, pRoot2);
}

//�������ľ���
void Mirror(TreeNode *pRoot) {
	if (pRoot && (pRoot->left || pRoot->right)) {
		swap(pRoot->left, pRoot->right);
		if (pRoot->left) Mirror(pRoot->left);
		if (pRoot->right) Mirror(pRoot->right);
	}
}

//�������´�ӡ������
vector<int> PrintFromTopToBottom(TreeNode* root) {
	//ʹ�ö���
	if (!root) return {};
	queue<TreeNode*> que;
	que.push(root);
	vector<int> vec;
	while (!que.empty()) {
		TreeNode* temp = que.front();
		vec.push_back(temp->val);
		que.pop();
		if (temp->left) que.push(temp->left);
		if (temp->right) que.push(temp->right);
	}
	return vec;
}

//�����������ĺ����������
bool judgeBST(vector<int> sequence) {
	if (sequence.empty()) return true;
	int head = sequence.back(), i = 0;  //ͷ�ڵ�ض�Ϊ���һ��Ԫ��
	for (; i < sequence.size() - 1; ++i) {
		//�ҵ�����������������ĵ�һ�����
		if (sequence[i] > head)  break;
	}
	for (int j = i; j < sequence.size() - 1; ++j) {
		//����������������С�ڵ���head��ֵ���򷵻�false
		if (sequence[j] <= head) return false;
	}
	vector<int> leftSeq(sequence.begin(), sequence.begin() + i);    //����������
	vector<int> rightSeq(sequence.begin() + i, sequence.end() - 1); //����������
																	//�ݹ�
	return judgeBST(leftSeq) && judgeBST(rightSeq);
}

bool VerifySquenceOfBST(vector<int> sequence) {
	if (sequence.empty()) return false;
	return judgeBST(sequence);
}

//�������к�Ϊĳһֵ��·��
void FindPath(vector<vector<int> >&vvec, TreeNode* root, int expNum, vector<int> temp) {
	temp.push_back(root->val);
	if (!root->left && !root->right) {
		if (root->val == expNum) {
			vvec.push_back(temp);
			return;
		}
		return;
	}
	if (root->left) {
		FindPath(vvec, root->left, expNum - root->val, temp);
	}
	if (root->right) {
		FindPath(vvec, root->right, expNum - root->val, temp);
	}
	temp.pop_back();
}

vector<vector<int> > FindPath(TreeNode* root, int expectNumber) {
	vector<vector<int> > vvec;
	vector<int> vec;
	if (root) FindPath(vvec, root, expectNumber, vec);
	return vvec;
}

//������������˫������
/*
//������ ��֪���������
void goLeft(TreeNode* root, stack<TreeNode*>& sta) {
while(root) {
sta.push(root);
root = root->left;
}
}
TreeNode* Convert(TreeNode* pRootOfTree) {
if(!pRootOfTree) return pRootOfTree;
stack<TreeNode*> sta;
TreeNode* root = new TreeNode(-1), *temp = root; //�ڱ��ڵ�,->rightָ�������������ڵ�
while(true) {
goLeft(pRootOfTree, sta);
if(sta.empty()) break;
TreeNode* nextRoot = sta.top();
sta.pop();
root->right = nextRoot;
nextRoot->left = root;
pRootOfTree = pRootOfTree->right;
root = nextRoot;
}
return temp->right;
}*/
//�ݹ��
void recurConvert(TreeNode* root, TreeNode*& pre) {
	if (!root) return;
	recurConvert(root->left, pre);
	root->left = pre;
	if (pre) pre->right = root;
	pre = root;
	recurConvert(root->right, pre);
}
TreeNode* Convert(TreeNode* pRootOfTree) {
	if (!pRootOfTree) return pRootOfTree;
	TreeNode* pre = 0;
	recurConvert(pRootOfTree, pre);
	TreeNode* res = pRootOfTree;
	while (res->left) res = res->left;
	return res;
}

//���������
int TreeDepth(TreeNode* pRoot) {
	return pRoot == NULL ? 0 : 1 + max(TreeDepth(pRoot->left), TreeDepth(pRoot->right));
}

//ƽ�������
bool IsBalanced(TreeNode* pRoot, int& pDepth) {
	if (pRoot == NULL) {
		pDepth = 0;
		return true;
	}
	int left, right;    //��¼���������ĸ߶�
	if (IsBalanced(pRoot->left, left) && IsBalanced(pRoot->right, right)) {
		int diff = left - right;
		if (-1 <= diff && diff <= 1) {
			pDepth = 1 + (left > right ? left : right);
			return true;
		}
	}
	return false;
}

bool IsBalanced_Solution(TreeNode* pRoot) {
	int pDepth = 0;
	return IsBalanced(pRoot, pDepth);
}

//����������һ�����
TreeLinkNode* GetNext(TreeLinkNode* pNode) {
	if (!pNode) return pNode;
	if (pNode->right) {
		//�ڵ����Һ��ӣ��Һ��ӵ�����ڵ㼴Ϊ��һ���ڵ�
		pNode = pNode->right;
		while (pNode->left) {
			pNode = pNode->left;
		}
		return pNode;
	}
	else {
		//û���Һ���
		TreeLinkNode* parent = pNode->next;
		if (parent) {
			//�ýڵ��Ǹ��ڵ�����ӣ����ڵ㼴Ϊ��һ���ڵ�
			if (parent->left == pNode) {
				return parent;
			}
			//�ýڵ��Ǹ��ڵ���Һ��ӣ��ظ��ڵ�������
			else {
				while (parent->next && parent == parent->next->right)
					parent = parent->next;
				return parent->next;
			}
		}
		else return NULL;
	}
}

//�ԳƵĶ�����
bool isSymmetrical(TreeNode* lChild, TreeNode* rChild) {
	if (lChild == NULL && rChild == NULL) return true;
	else if (lChild && rChild) {
		if (lChild->val != rChild->val) return false;
		else {
			return isSymmetrical(lChild->left, rChild->right) && isSymmetrical(lChild->right, rChild->left);
		}
	}
	else return false;
}
bool isSymmetrical(TreeNode* pRoot) {
	if (pRoot == NULL) return true;
	return isSymmetrical(pRoot->left, pRoot->right);
}

//�Ѷ�������ӡ�ɶ���
void recursion(TreeNode* pRoot, vector<vector<int>> &vec, int index) {
	if (pRoot == NULL) return;
	else vec[index].push_back(pRoot->val);
	recursion(pRoot->left, vec, index + 1);
	recursion(pRoot->right, vec, index + 1);
}
vector<vector<int> > Print2(TreeNode* pRoot) {
	int h = TreeDepth(pRoot);
	if (pRoot == NULL) return {};
	vector<int> v{};
	vector<vector<int>> vec(h, v);
	int index = 0;
	recursion(pRoot, vec, index);
	return vec;
}

//��֮����˳���ӡ������
vector<vector<int> > Print(TreeNode* pRoot) {
	vector<vector<int> > vvec = {};
	if (!pRoot) return vvec;
	stack<TreeNode*> oddSta;//��������ջ��Ų�ͬ������ڵ�
	stack<TreeNode*> evenSta;
	oddSta.push(pRoot);
	while (!oddSta.empty() || !evenSta.empty()) {
		vector<int> vec = {};
		if (oddSta.empty()) {
			//ջ1�գ�ջ2��Ԫ�س�ջ������vec,������������ջ1
			while (!evenSta.empty()) {
				TreeNode* node = evenSta.top();
				evenSta.pop();
				vec.push_back(node->val);
				if (node->right) oddSta.push(node->right);
				if (node->left) oddSta.push(node->left);
			}
		}
		else {
			//ջ2�գ�ջ2��Ԫ�س�ջ������vec�����Һ���������ջ2
			while (!oddSta.empty()) {
				TreeNode* node = oddSta.top();
				oddSta.pop();
				vec.push_back(node->val);
				if (node->left) evenSta.push(node->left);
				if (node->right) evenSta.push(node->right);
			}
		}
		vvec.push_back(vec);
	}
	return vvec;
}

//�����������ĵ�k�����
void KthNode(TreeNode* pRoot, vector<TreeNode*>& TN) {
	if (!pRoot) return;
	else {
		KthNode(pRoot->left, TN);
		TN.push_back(pRoot);
		KthNode(pRoot->right, TN);
	}
}
TreeNode* KthNode(TreeNode* pRoot, int k) {
	//�������������
	vector<TreeNode*> TN;
	KthNode(pRoot, TN);
	return (TN.size() < k || k <= 0) ? NULL : TN[k - 1];
}

//���л�������
//���������������vector�洢
void Serialize(vector<int>& sto, TreeNode *root) {
	if (!root) sto.push_back(0xffff);
	else {
		sto.push_back(root->val);
		Serialize(sto, root->left);
		Serialize(sto, root->right);
	}
}
char* Serialize(TreeNode *root) {
	vector<int> sto;
	Serialize(sto, root);
	int* res = new int[sto.size()];
	for (int i = 0; i < sto.size(); ++i) res[i] = sto[i];
	return (char*)res;
}
TreeNode* Deserialize(int*& str) {
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
TreeNode* Deserialize(char *str) {
	int *p = (int *)str;
	return Deserialize(p);
}

/****************************************************************/
/*---------------------------����-------------------------------*/
/****************************************************************/

//�����е�·��
bool hasPath(char* matrix, int rows, int cols, char* str, bool* flag, int x, int y, int index) {
	int m_index = x * cols + y;
	if (x < 0 || x >= rows || y < 0 || y >= cols || flag[m_index] || matrix[m_index] != str[index])
		//λ��Խ����߸�λ���Ѿ����ʻ����ַ���ƥ��
		return false;
	if (str[index + 1] == '\0') return true; //�ַ�����ĩβ
	flag[m_index] = true;
	if (hasPath(matrix, rows, cols, str, flag, x - 1, y, index + 1) ||
		hasPath(matrix, rows, cols, str, flag, x, y - 1, index + 1) ||
		hasPath(matrix, rows, cols, str, flag, x + 1, y, index + 1) ||
		hasPath(matrix, rows, cols, str, flag, x, y + 1, index + 1)) return true;
	flag[m_index] = false;
	return false;
}

bool hasPath(char* matrix, int rows, int cols, char* str) {
	if (!str || rows <= 0 || cols <= 0) return false;
	bool* flag = new bool[rows * cols];
	memset(flag, 0, rows * cols);
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			if (hasPath(matrix, rows, cols, str, flag, i, j, 0)) return true;
		}
	}
	delete[] flag;
	return false;
}


//�����˵��˶���Χ
bool compVal(int x, int y, int threshold) {
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

int findById(vector<int> id, int p) {
	if (p == -1) return -1;
	else if (p == id[p]) return p;
	else return findById(id, id[p]);
}

void unionById(vector<int>& id, int p, int q) {
	int i = findById(id, p), j = findById(id, q);
	if (i == j) return;
	i <= j ? id[q] = i : id[p] = j;
}

int movingCount(int threshold, int rows, int cols) {
	vector<int> id(rows * cols);    //idΪ��ά���鰴����ֱ
	int count = 0;
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			//������������͵�idֵ��Ϊ-1��������Ϊ���±�
			id[i * cols + j] = (compVal(i, j, threshold) ? i * cols + j : -1);
		}
	}
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			//union����
			if (i + 1 < rows && compVal(i, j, threshold) && compVal(i + 1, j, threshold))
				unionById(id, i * cols + j, (i + 1) * cols + j);
			if (j + 1 < cols && compVal(i, j, threshold) && compVal(i, j + 1, threshold))
				unionById(id, i * cols + j, i * cols + j + 1);
		}
	}
	for (int i = 0; i < rows * cols; ++i) {
		//idΪ0�����꼴�����������ܹ������˵���
		if (findById(id, i) == 0)
			++count;
	}
	return count;
}


/***************************************************************/
/*----------------------------ջ�Ͷ���--------------------------*/
/****************************************************************/

//�������ڵ����ֵ
vector<int> maxInWindows(const vector<int>& num, unsigned int size) {
	//ʹ�ö���
	if (num.empty() || num.size() < size || !size)
		//�����������������Ϊ�ջ��ߴ��ڳ��ȴ������鳤�Ȼ��ߴ��ڳ���Ϊ��
		return {};
	deque<int> deq;
	vector<int> re;
	for (int i = 0; i < num.size(); ++i) {
		while (!deq.empty() && num[deq.back()] <= num[i]) deq.pop_back();
		while (!deq.empty() && i - deq.front() + 1 > size) deq.pop_front();
		deq.push_back(i);
		if (i + 1 >= size) re.push_back(num[deq.front()]);
	}
	return re;
}

//����min������ջ
stack<int> dataStack, minStack;
void push(int value) {
	dataStack.push(value);
	if (minStack.empty() || minStack.top() >= value) {
		minStack.push(value);
	}
}
void pop() {
	if (dataStack.top() == minStack.top()) {
		minStack.pop();
	}
	dataStack.pop();
}
int top() {
	return dataStack.top();
}
int min() {
	return minStack.top();
}

//ջ��ѹ�롢��������
bool IsPopOrder(vector<int> pushV, vector<int> popV) {
	if (pushV.empty() || pushV.size() != popV.size()) return false;
	stack<int> sta;
	for (int i = 0, j = 0; i < pushV.size(); ++i) {
		sta.push(pushV[i]);
		while (j < popV.size() && sta.top() == popV[j]) {
			sta.pop();
			++j;
		}
	}
	return sta.empty();
}

/*****************************************************************/
/*-----------------------------����-------------------------------*/
/******************************************************************/

//��β��ͷ��ӡ����
/*�ݹ�汾
void print(ListNode* head, vector<int> &vec){
if(head == NULL)
return;
print(head->next, vec);
vec.push_back(head->val);
}
vector<int> printListFromTailToHead(ListNode* head) {
vector<int> vec;
print(head, vec);
return vec;
}*/
vector<int> printListFromTailToHead(ListNode* head) {
	//�ǵݹ�汾
	vector<int> vec;
	stack<int> sta;
	ListNode* temp = head;
	while (temp != NULL) {
		sta.push(temp->val);
		temp = temp->next;
	}
	while (!sta.empty()) {
		vec.push_back(sta.top());
		sta.pop();
	}
	return vec;
}

//�����е�����k�����
ListNode* FindKthToTail(ListNode* pListHead, unsigned int k) {
	//һ�������{first, last}�ľ���Ϊk,��last����ĩλʱ��first��Ϊ������k�����
	ListNode* first = pListHead, *last = pListHead;
	while (k) {
		if (!last) break;
		last = last->next;
		--k;
	}
	if (k != 0) return NULL; //������С��k
	while (last) {
		last = last->next;
		first = first->next;
	}
	return first;
}

//��ת����
ListNode* ReverseList(ListNode* pHead) {
	//�ݹ�棻
	if (pHead == NULL || pHead->next == NULL)
		return pHead;
	ListNode *pReverseNode = ReverseList(pHead->next);
	//��ת�������
	pHead->next->next = pHead;
	pHead->next = NULL;
	return pReverseNode;
}
/*
ListNode* ReverseList(ListNode* pHead) {
//�ǵݹ��
if(pHead == NULL || pHead->next == NULL)
return pHead;
ListNode* pNode=pHead;//��ǰָ��
ListNode* pReverseHead=NULL;//�������ͷָ��
ListNode* pPrev=NULL;//��ǰָ���ǰһ�����

while(pNode != NULL){//��ǰ��㲻Ϊ��ʱ��ִ��
ListNode* pNext=pNode->next;//���Ͽ�֮ǰһ��Ҫ����Ͽ�λ�ú�ߵĽ��
if(pNext==NULL)//��pNextΪ��ʱ��˵����ǰ���Ϊβ�ڵ�
pReverseHead=pNode;
pNode->next=pPrev;//ָ�뷴ת
pPrev=pNode;
pNode=pNext;
}
return pReverseHead;
}*/

//�ϲ��������������
ListNode* Merge(ListNode* pHead1, ListNode* pHead2) {
	ListNode* result = new ListNode(0); //�趨һ��ͷ�ڵ�,�ýڵ�֮��ʼ�ϲ�
	ListNode* temp = result;    //ֻ�践��temp�ڵ��next
	while (pHead1 && pHead2) {
		if (pHead1->val <= pHead2->val) {
			result->next = pHead1;
			pHead1 = pHead1->next;
		}
		else {
			result->next = pHead2;
			pHead2 = pHead2->next;
		}
		result = result->next;
	}
	result->next = (pHead1 ? pHead1 : pHead2);
	return temp->next;
}

//��������ĸ���
RandomListNode* Clone(RandomListNode* pHead) {
	if (!pHead) return pHead;
	RandomListNode *cloneNode = pHead;
	//����ÿ���ڵ������������
	while (cloneNode) {
		RandomListNode* Node = new RandomListNode(cloneNode->label);
		//���������ǽ�Node���뵽cloneNode��cloneNode->next֮��
		Node->next = cloneNode->next;
		cloneNode->next = Node;
		cloneNode = Node->next;
	}
	cloneNode = pHead;
	while (cloneNode) {
		//��ָ��random���и���
		if (cloneNode->random)
			cloneNode->next->random = cloneNode->random->next;
		cloneNode = cloneNode->next->next;
	}
	//�ٽ��в��
	cloneNode = pHead->next;
	while (pHead->next) {
		RandomListNode* temp = pHead->next;
		pHead->next = temp->next;
		//temp->next = pHead->next->next;
		//pHead = pHead->next;
		pHead = temp;
	}
	return cloneNode;
}

//��������ĵ�һ���������
ListNode* FindFirstCommonNode(ListNode* pHead1, ListNode* pHead2) {
	ListNode *p1 = pHead1, *p2 = pHead2;
	while (p1 != p2) {
		//ʵ����������ˣ�������ͬʱ��һ�˾Ϳ���ֱ�ӵõ�������
		//���Ȳ�ͬʱ���ڶ����������ڻ����ڵ㣬���Ȳ�û����
		p1 = p1 ? p1->next : pHead2;
		p2 = p2 ? p2->next : pHead1;
	}
	return p1;
}

//�����л�����ڽ��
ListNode* EntryNodeOfLoop(ListNode* pHead) {
	//ʹ��˫ָ��p1,p2; p1ÿ��ǰ��1����λ,p2ÿ��ǰ��2����λ
	//�л���Ȼ����������֮��p2��Ϊԭ��,��ͬʱ������1ǰ�����ٴ�������Ȼ�ǻ������
	ListNode* p1 = pHead, *p2 = pHead;
	while (p1 && p2) {
		p1 = p1->next;
		if (p2->next) p2 = p2->next->next;
		else return nullptr;
		if (p1 == p2) break;
	}
	if (!p1 || !p2) return nullptr;
	p2 = pHead;
	while (p1 != p2) {
		p1 = p1->next;
		p2 = p2->next;
	}
	return p1;
}

//ɾ���������ظ��Ľ��
ListNode* deleteDuplication(ListNode* pHead) {
	if (!pHead || !pHead->next) return pHead;
	ListNode* current;
	if (pHead->next->val == pHead->val) {
		current = pHead->next->next;
		while (current && current->val == pHead->val)
			current = current->next;
		return deleteDuplication(current);
	}
	else {
		current = pHead->next;
		pHead->next = deleteDuplication(current);
		return pHead;
	}
}

/******************************************************/
/*--------------------math��ѧ-------------------------*/
/******************************************************/

//쳲���������
int Fibonacci(int n) {
	int pre = 0, last = 1;  //������¼f(n-2)��f(n-1)
	if (n == 0) return pre;
	while (--n) {
		last += pre;
		pre = last - pre;
	}
	return last;
}

//��ֵ�������η�
/*�ݹ�汾
double Power(double base, int exponent) {
if(exponent == 0) return 1;
else if(exponent < 0)return Power(1 / base, -exponent);
else {
double temp = Power(base, exponent/2) * Power(base, exponent/2);
return temp * temp * (exponent % 2 ? base : 1);
}
}*/
//λ����汾
double Power(double base, int exponent) {
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
	//Լɪ������ f(n, m)
	//���ƹ�ϵΪ��f(1, m) = 0, f(n, m) = (f(n - 1, m) + m) % n;
	if (n < 1 || m < 1) return -1;
	int last = 0;
	for (int i = 2; i < n; ++i) {
		last = (last + m) % i;
	}
	return last;
}