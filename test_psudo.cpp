#include<stdio.h>
#include "psudo.cpp" //comment out to run in vivado
int hand_num_nn(float X[8], int y);

int main()
{
	// normalized input
	float X[8]= {0.084395,0.776438,0.523252,0.000000,0.000000,0.218584,0.001409,0.261626};

	float i = hand_num_nn(X,2);
	if (i>=0.5){printf("Yes, you have diabetes\n");}
	else{printf("No, you don't have diabetes\n");}
	return 0;
}
