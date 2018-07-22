#include "PointToOffer.h"
#include "stdafx.h"

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
	if (n <= 0 || m <= 0) throw "input is non-positive";
	int re = 0;
	for (int i = 2; i <= n; ++i) {
		re = (re + m) % i;
	}
	return re;
}