// PtOffer.cpp: 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include "gtest/gtest.h"
#include "PointToOffer.h"

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
	testing::GTEST_FLAG(output) = "xml:Ouput.xml";
	testing::InitGoogleTest(&argc, argv);
	RUN_ALL_TESTS();
  return 0;
}

