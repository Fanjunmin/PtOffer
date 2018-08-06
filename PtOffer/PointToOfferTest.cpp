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

//---------------------------TESTCASE---------------------------

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

