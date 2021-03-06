// PtOffer.cpp: 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include "gtest/gtest.h"
#include "PointToOffer.h"

//--------------------------TestCase事件-------------------- 

//包含min函数的栈
class MinStackTest : public testing::Test {
public:
		virtual void SetUp() { 
		}
		virtual void TearDown(){
			while (!g_dataStack.empty())
				g_dataStack.pop();
			while (!g_minStack.empty())
				g_minStack.pop();
		}
};

//---------------------------TESTCASE-----------------------

//---------------------------Array数组-----------------------
//二维数组中的查找
TEST(FindTest, HandleAnyInput) {
  vector<vector<int>> vec1 = {{}, {}, {}};
	EXPECT_EQ(Find(1, vec1), false);
	EXPECT_NE(Find(2, {}), true);
	vector<vector<int>> vec2 = {{1,2,3}, {4,5,6}, {7,8,9}};
	EXPECT_EQ(Find(4, vec2), true);
	EXPECT_EQ(Find(0, vec2), false);
}

//旋转数组的最小数字
TEST(minNumberInRotateArrayTest, HandleAnyInput) {
	EXPECT_EQ(minNumberInRotateArray({}), 0);
  EXPECT_NE(minNumberInRotateArray({3,4,5,1,2}), 2);
	EXPECT_EQ(minNumberInRotateArray({2,1,1,1,2,2}), 1);
}

//调整数组顺序使奇数位于偶数前面
TEST(reOrderArrayTest, handleAnyInput) {
	vector<int> shouldbe = {1,3,5,7,2,4,6,8};
	vector<int> vec1 = {2,4,6,8,1,3,5,7};
	vector<int> vec2 = {1,2,3,4,5,6,7,8};
	reOrderArray(vec1);
	EXPECT_EQ(vec1, shouldbe);
  reOrderArray(vec2);
	EXPECT_EQ(vec2, shouldbe);
}

//数组中出现次数超过一半的数字
TEST(MoreThanHalfNum_SolutionTest, handAnyInput) {
  EXPECT_EQ(MoreThanHalfNum_Solution({}), 0);
	EXPECT_EQ(MoreThanHalfNum_Solution({1, 2, 3}), 0);
  EXPECT_EQ(MoreThanHalfNum_Solution({1, 2, 2, 2, 3}), 2);
}

//连续子数组的最大和
TEST(FindGreatestSumOfSubArrayTest, handleAnyInput) {
	EXPECT_EQ(FindGreatestSumOfSubArray({1, 2, -2, 3, 4, 5, -1}), 13);
  EXPECT_EQ(FindGreatestSumOfSubArray({1, -2, 3, 4, 5, -1}), 12);
}

//把数组排成最小的数
TEST(PrintMinNumberTest, handleAnyInput) {
	EXPECT_EQ(PrintMinNumber({3, 32, 321}), "321323");
  EXPECT_EQ(PrintMinNumber({1, 11, 1001, 10}), "100110111");
}

//数组中的逆序对
TEST(InversePairsTest, handleAnyInput) {
	EXPECT_EQ(InversePairs({1,2,3,4,5,6,7,0}), 7);
  EXPECT_EQ(InversePairs({2,4,6,8,1,3,5,7}), 10);
}

//数字在排序数组中出现的次数
TEST(GetNumberOfKTest, handleAnyInput) {
	EXPECT_EQ(GetNumberOfK({0,1,1,2,2,3,3,3,4}, 3), 3);
  EXPECT_EQ(GetNumberOfK({}, 1), 0);
}

//数组中只出现一次的数字
TEST(FindNumsAppearOnceTest, handleAnyInput) {
	int *num1 = new int(0), *num2 = new int(0);
  FindNumsAppearOnce({1, 1, 2, 2, 3, 4}, num1, num2);
	EXPECT_TRUE(*num1 == 3 && *num2 == 4);
  FindNumsAppearOnce({3, 5, 1, 1, 2, 2, 3, 4}, num1, num2);
	EXPECT_TRUE(*num1 == 5 && *num2 == 4);
	delete num1, num2;
}

//数组中重复的数字
TEST(duplicateTest, handleAnyInput) {
	int num[5] = {2, 3, 4, 5, 2};
	int a = 0, *dup = &a;
	EXPECT_TRUE(duplicate(num, 5, dup));
	EXPECT_EQ(*dup, 2);
}

//顺时针打印矩阵
TEST(printMatrixTest, hanleAnyInput) {
	vector<vector<int>> matrix1 = {{1, 2}, {3, 4}};
  vector<vector<int>> matrix2 = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
	vector<int> result1 = {1, 2, 4, 3};
	vector<int> result2 = {1,2,3,6,9,8,7,4,5};
	EXPECT_TRUE(printMatrix(matrix1) == result1);
	EXPECT_TRUE(printMatrix(matrix2) == result2);
}

//和为S的连续正数序列
TEST(FindContinuousSequenceTest, handleAnyInput) {
	vector<vector<int>> result1 = {};
  vector<vector<int>> result2 = {{2,3,4}, {4,5}};
	vector<vector<int>> result3 = {{9,10,11,12,13,14,15,16}, {18,19,20,21,22}};
	EXPECT_TRUE(FindContinuousSequence(2) == result1);
	EXPECT_TRUE(FindContinuousSequence(9) == result2);
	EXPECT_TRUE(FindContinuousSequence(100) == result3);
}

//丑数
TEST(GetUglyNumber_SolutionTest, handleAnyInput) {
	EXPECT_EQ(GetUglyNumber_Solution(1), 1);
  EXPECT_EQ(GetUglyNumber_Solution(3), 3);
	EXPECT_EQ(GetUglyNumber_Solution(5), 5);
  EXPECT_EQ(GetUglyNumber_Solution(9), 10);
	EXPECT_EQ(GetUglyNumber_Solution(11), 15);
}

//和为S的两个数字
TEST(FindNumbersWithSumTest, handleAnyInput) {
	vector<int> result1 = {1, 5};
	EXPECT_TRUE(FindNumbersWithSum({1,2,3,4,5}, 6) == result1);
	vector<int> result2 = {1, 6};
	EXPECT_TRUE(FindNumbersWithSum({1,2,3,4,5,6}, 7) == result2);
	vector<int> result3;
  EXPECT_TRUE(FindNumbersWithSum({1, 2, 3, 4, 5, 6}, 18) == result3);

}

//扑克牌顺子
TEST(IsContinuousTest, handleAnyInput) {
  EXPECT_FALSE(IsContinuous({}));
  EXPECT_TRUE(IsContinuous({0}));
  EXPECT_FALSE(IsContinuous({1,5,7,0,0,0}));
	EXPECT_TRUE(IsContinuous({1,3,5,0,0}));
}

//数据流中的中位数
TEST(GetMedianTest, handleAnyInput) {
	Insert(0);
  EXPECT_DOUBLE_EQ(GetMedian(), 0);
	Insert(4);
  EXPECT_DOUBLE_EQ(GetMedian(), 2.0);
	Insert(7);
  EXPECT_DOUBLE_EQ(GetMedian(), 4.0);
	Insert(5);
  EXPECT_DOUBLE_EQ(GetMedian(), 4.5);
}

//最小的k个数
TEST(GetLeastNumbers_SolutionTest, handleAnyInput) {
  vector<int> vec = {4, 5, 1, 6, 2, 7, 3, 8};
	vector<int> result1 = {1, 2, 3};
  vector<int> result2 = {1, 2, 3, 4, 5};
	EXPECT_TRUE(GetLeastNumbers_Solution(vec, 3) == result1);
  EXPECT_TRUE(GetLeastNumbers_Solution(vec, 5) == result2);
}


//---------------------------栈和队列-----------------------

TEST(movingCountTest, HandleAnyInput) {
	EXPECT_EQ(movingCount(8, 10, 2), 17);
  EXPECT_EQ(movingCount(3, 5, 3), 9);
}

TEST(maxInWindowsTest, HandleAnyInput) {
	vector<int> vec1{}, vec2 = {2,3,4,2,6,2,5,1};
  vector<int> re = {4,4,6,6,6,5};
  EXPECT_TRUE(maxInWindows(vec1, -1).empty());
  EXPECT_TRUE(maxInWindows(vec2, 9).empty());
  EXPECT_TRUE(maxInWindows(vec2, 3) == re);
}

//包含min函数的栈
TEST_F(MinStackTest, Input1) {
  //EXPECT_ANY_THROW(min());
	push(4);
  EXPECT_NE(min(), 3);
  push(2);
  EXPECT_EQ(min(), 2);
  pop();
  push(5);
  EXPECT_EQ(min(), 4);
}		
TEST_F(MinStackTest, Input2) {
	//TODO(junminfan@126.com):测试套之间有影响，未知原因
  EXPECT_TRUE(g_minStack.empty());
	//EXPECT_ANY_THROW(min());
  push(3);	//若最小值大于4，min()还是返回4,和上一个测试关联了？
  EXPECT_NE(min(), 2);
  push(10);
  EXPECT_EQ(min(), 3);
  push(7);
  EXPECT_EQ(min(), 3);
}	

//栈的压入、弹出序列
TEST(IsPopOrderTest, HanleAnyInput) {
	vector<int> pushV = {1, 2, 3, 4, 5}; 
	vector<int> popV1 = {5, 4, 3, 2, 1};
  vector<int> popV2 = {4, 3, 2, 1};
  vector<int> popV3 = {5, 3, 2, 1, 4};
  EXPECT_TRUE(IsPopOrder(pushV, popV1));
  EXPECT_FALSE(IsPopOrder(pushV, popV2));
  EXPECT_FALSE(IsPopOrder(pushV, popV3));
}

//---------------------------链表---------------------------

TEST(printListFromTailToHeadTest, HandleAnyInput) {
  ASSERT_TRUE(printListFromTailToHead(nullptr) == vector<int>());
}
//---------------------------数学---------------------------

TEST(FibonacciTest, HandleAnyInput) { 
  EXPECT_THROW(Fibonacci(-1), char*);
  EXPECT_EQ(Fibonacci(0), 0);
  EXPECT_EQ(Fibonacci(1), 1);
  EXPECT_EQ(Fibonacci(10), 55);
}

TEST(PowerTest, HandleAnyInput) {
	EXPECT_ANY_THROW(Power(0, 0));
  EXPECT_DOUBLE_EQ(Power(1, 100), 1);
  EXPECT_DOUBLE_EQ(Power(1000.0001, 0), 1);
  EXPECT_DOUBLE_EQ(Power(1.1, 2), 1.21);
	EXPECT_NEAR(Power(1.11, 2), 1.232100001, 10e-10);
}

TEST(LastRemaining_SolutionTest, HandleAnyInput) {
  EXPECT_ANY_THROW(LastRemaining_Solution(0, -1));
  EXPECT_EQ(LastRemaining_Solution(1000, 2), 976);
  EXPECT_EQ(LastRemaining_Solution(1000, 1), 999);
  EXPECT_EQ(LastRemaining_Solution(10, 3), 3);
}

int main(int argc, char** argv)
{
  std::cout << "Start Running Google Test: \n";
	testing::GTEST_FLAG(output) = "xml:Ouput.xml";
	testing::InitGoogleTest(&argc, argv);
	RUN_ALL_TESTS();
  return 0;
}

