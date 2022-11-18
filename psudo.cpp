// front prop
// take intial 8 inputs
// initial array of size 8
// apply for loop on FC1
//
//      add bias
//      normalize
//      apply activation
// get result 128 long array
// take to next layer

#include "math.h"
#include "stdio.h"
#include "weight.h" 


// This is the top function defined during project initialization
int hand_num_nn(float X[8], int y)
{

#pragma HLS INTERFACE s_axilite port = return bundle = CRTL_BUS
#pragma HLS INTERFACE s_axilite port = y bundle = CRTL_BUS
#pragma HLS INTERFACE bram port = X



float h1[128];
float h2[128];
float h3[128];
float h1_mid[1][128];
int i = 0, j = 0;
float ans = 0;
// float ans[128];
	// Feed Forward Network Implementation
	// first layer 1x8 8x128 -> 1x128
	for (i = 0; i < 128; i++)
	{
		h1[i]=0;
                // multiplying weights
		for (j = 0; j < 8; j++)
		{
			h1[i] = h1[i] + (X[j] * fc1_big[j][i]);
		}
		// adding bias
		h1[i] += b1_big[i];
		//reLu activation
		if(h1[i]<0)
		{h1[i] = 0;}
	}

    // second layer 1x128 128x128 -> 1x128
	for (i = 0; i < 128; i++)
	{
		h2[i]=0;
        // multiplying weights
		for (j = 0; j < 128; j++)
		{
			h2[i] += (h1[j] * fc2_big[j][i]);
		}
		h2[i] += b2_big[i];
                // reLU activation
		if(h2[i]<0)
		{h2[i] = 0;}    
	}

        // third layer 1x128 128x128 -> 1x128
	for (i = 0; i < 128; i++)
	{
		h3[i]=0;
                // multiplying weights
		for (j = 0; j < 128; j++)
		{
			h3[i] += (h2[j] * fc3_big[j][i]);
		}
		h3[i] += b3_big[i];
        if(h3[i]<0)
		{h3[i] = 0;}     
	}

        
        //outlayer 1x128 128x1 -> 1x1
        // multiplying weights
		for (j = 0; j < 128; j++)
		{
            ans = ans + (h3[j]*outlayer_big[j][0]);
			
		}
    // adding biases
	ans = ans + b4_big[0];

	//sigmoid activation
    ans = 1.0 / (1.0 + exp(-1 * ans));
	
		
	printf("%f\n",ans);

	return ans;
    
}
        
