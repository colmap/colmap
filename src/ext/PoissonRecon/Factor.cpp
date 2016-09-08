/*
Copyright (c) 2006, Michael Kazhdan and Matthew Bolitho
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of
conditions and the following disclaimer. Redistributions in binary form must reproduce
the above copyright notice, this list of conditions and the following disclaimer
in the documentation and/or other materials provided with the distribution. 

Neither the name of the Johns Hopkins University nor the names of its contributors
may be used to endorse or promote products derived from this software without specific
prior written permission. 

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO THE IMPLIED WARRANTIES 
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, PROCUREMENT OF SUBSTITUTE  GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.
*/

//////////////////////
// Polynomial Roots //
//////////////////////
#include <math.h>
#include "Factor.h"
int Factor(double a1,double a0,double roots[1][2],double EPS){
	if(fabs(a1)<=EPS){return 0;}
	roots[0][0]=-a0/a1;
	roots[0][1]=0;
	return 1;
}
int Factor(double a2,double a1,double a0,double roots[2][2],double EPS){
	double d;
	if(fabs(a2)<=EPS){return Factor(a1,a0,roots,EPS);}

	d=a1*a1-4*a0*a2;
	a1/=(2*a2);
	if(d<0){
		d=sqrt(-d)/(2*a2);
		roots[0][0]=roots[1][0]=-a1;
		roots[0][1]=-d;
		roots[1][1]= d;
	}
	else{
		d=sqrt(d)/(2*a2);
		roots[0][1]=roots[1][1]=0;
		roots[0][0]=-a1-d;
		roots[1][0]=-a1+d;
	}
	return 2;
}
// Solution taken from: http://mathworld.wolfram.com/CubicFormula.html
// and http://www.csit.fsu.edu/~burkardt/f_src/subpak/subpak.f90
int Factor(double a3,double a2,double a1,double a0,double roots[3][2],double EPS){
	double q,r,r2,q3;

	if(fabs(a3)<=EPS){return Factor(a2,a1,a0,roots,EPS);}
	a2/=a3;
	a1/=a3;
	a0/=a3;

	q=-(3*a1-a2*a2)/9;
	r=-(9*a2*a1-27*a0-2*a2*a2*a2)/54;
	r2=r*r;
	q3=q*q*q;

	if(r2<q3){
		double sqrQ=sqrt(q);
		double theta = acos ( r / (sqrQ*q) );
		double cTheta=cos(theta/3)*sqrQ;
		double sTheta=sin(theta/3)*sqrQ*SQRT_3/2;
		roots[0][1]=roots[1][1]=roots[2][1]=0;
		roots[0][0]=-2*cTheta;
		roots[1][0]=-2*(-cTheta*0.5-sTheta);
		roots[2][0]=-2*(-cTheta*0.5+sTheta);
	}
	else{
		double s1,s2,sqr=sqrt(r2-q3);
		double t;
		t=-r+sqr;
		if(t<0){s1=-pow(-t,1.0/3);}
		else{s1=pow(t,1.0/3);}
		t=-r-sqr;
		if(t<0){s2=-pow(-t,1.0/3);}
		else{s2=pow(t,1.0/3);}
		roots[0][1]=0;
		roots[0][0]=s1+s2;
		s1/=2;
		s2/=2;
		roots[1][0]= roots[2][0]=-s1-s2;
		roots[1][1]= SQRT_3*(s1-s2);
		roots[2][1]=-roots[1][1];
	}
	roots[0][0]-=a2/3;
	roots[1][0]-=a2/3;
	roots[2][0]-=a2/3;
	return 3;
}
double ArcTan2(double y,double x){
	/* This first case should never happen */
	if(y==0 && x==0){return 0;}
	if(x==0){
		if(y>0){return PI/2.0;}
		else{return -PI/2.0;}
	}
	if(x>=0){return atan(y/x);}
	else{
		if(y>=0){return atan(y/x)+PI;}
		else{return atan(y/x)-PI;}
	}
}
double Angle(const double in[2]){
	if((in[0]*in[0]+in[1]*in[1])==0.0){return 0;}
	else{return ArcTan2(in[1],in[0]);}
}
void Sqrt(const double in[2],double out[2]){
	double r=sqrt(sqrt(in[0]*in[0]+in[1]*in[1]));
	double a=Angle(in)*0.5;
	out[0]=r*cos(a);
	out[1]=r*sin(a);
}
void Add(const double in1[2],const double in2[2],double out[2]){
	out[0]=in1[0]+in2[0];
	out[1]=in1[1]+in2[1];
}
void Subtract(const double in1[2],const double in2[2],double out[2]){
	out[0]=in1[0]-in2[0];
	out[1]=in1[1]-in2[1];
}
void Multiply(const double in1[2],const double in2[2],double out[2]){
	out[0]=in1[0]*in2[0]-in1[1]*in2[1];
	out[1]=in1[0]*in2[1]+in1[1]*in2[0];
}
void Divide(const double in1[2],const double in2[2],double out[2]){
	double temp[2];
	double l=in2[0]*in2[0]+in2[1]*in2[1];
	temp[0]= in2[0]/l;
	temp[1]=-in2[1]/l;
	Multiply(in1,temp,out);
}
// Solution taken from: http://mathworld.wolfram.com/QuarticEquation.html
// and http://www.csit.fsu.edu/~burkardt/f_src/subpak/subpak.f90
int Factor(double a4,double a3,double a2,double a1,double a0,double roots[4][2],double EPS){
	double R[2],D[2],E[2],R2[2];

	if(fabs(a4)<EPS){return Factor(a3,a2,a1,a0,roots,EPS);}
	a3/=a4;
	a2/=a4;
	a1/=a4;
	a0/=a4;

	Factor(1.0,-a2,a3*a1-4.0*a0,-a3*a3*a0+4.0*a2*a0-a1*a1,roots,EPS);

	R2[0]=a3*a3/4.0-a2+roots[0][0];
	R2[1]=0;
	Sqrt(R2,R);
	if(fabs(R[0])>10e-8){
		double temp1[2],temp2[2];
		double p1[2],p2[2];

		p1[0]=a3*a3*0.75-2.0*a2-R2[0];
		p1[1]=0;

		temp2[0]=((4.0*a3*a2-8.0*a1-a3*a3*a3)/4.0);
		temp2[1]=0;
		Divide(temp2,R,p2);

		Add     (p1,p2,temp1);
		Subtract(p1,p2,temp2);

		Sqrt(temp1,D);
		Sqrt(temp2,E);
	}
	else{
		R[0]=R[1]=0;
		double temp1[2],temp2[2];
		temp1[0]=roots[0][0]*roots[0][0]-4.0*a0;
		temp1[1]=0;
		Sqrt(temp1,temp2);
		temp1[0]=a3*a3*0.75-2.0*a2+2.0*temp2[0];
		temp1[1]=                  2.0*temp2[1];
		Sqrt(temp1,D);
		temp1[0]=a3*a3*0.75-2.0*a2-2.0*temp2[0];
		temp1[1]=                 -2.0*temp2[1];
		Sqrt(temp1,E);
	}

	roots[0][0]=-a3/4.0+R[0]/2.0+D[0]/2.0;
	roots[0][1]=        R[1]/2.0+D[1]/2.0;

	roots[1][0]=-a3/4.0+R[0]/2.0-D[0]/2.0;
	roots[1][1]=        R[1]/2.0-D[1]/2.0;

	roots[2][0]=-a3/4.0-R[0]/2.0+E[0]/2.0;
	roots[2][1]=       -R[1]/2.0+E[1]/2.0;

	roots[3][0]=-a3/4.0-R[0]/2.0-E[0]/2.0;
	roots[3][1]=       -R[1]/2.0-E[1]/2.0;
	return 4;
}

int Solve(const double* eqns,const double* values,double* solutions,int dim){
	int i,j,eIndex;
	double v,m;
	int *index=new int[dim];
	int *set=new int[dim];
	double* myEqns=new double[dim*dim];
	double* myValues=new double[dim];

	for(i=0;i<dim*dim;i++){myEqns[i]=eqns[i];}
	for(i=0;i<dim;i++){
		myValues[i]=values[i];
		set[i]=0;
	}
	for(i=0;i<dim;i++){
		// Find the largest equation that has a non-zero entry in the i-th index
		m=-1;
		eIndex=-1;
		for(j=0;j<dim;j++){
			if(set[j]){continue;}
			if(myEqns[j*dim+i]!=0 && fabs(myEqns[j*dim+i])>m){
				m=fabs(myEqns[j*dim+i]);
				eIndex=j;
			}
		}
		if(eIndex==-1){
			delete[] index;
			delete[] myValues;
			delete[] myEqns;
			delete[] set;
			return 0;
		}
		// The position in which the solution for the i-th variable can be found
		index[i]=eIndex;
		set[eIndex]=1;

		// Normalize the equation
		v=myEqns[eIndex*dim+i];
		for(j=0;j<dim;j++){myEqns[eIndex*dim+j]/=v;}
		myValues[eIndex]/=v;

		// Subtract it off from everything else
		for(j=0;j<dim;j++){
			if(j==eIndex){continue;}
			double vv=myEqns[j*dim+i];
			for(int k=0;k<dim;k++){myEqns[j*dim+k]-=myEqns[eIndex*dim+k]*vv;}
			myValues[j]-=myValues[eIndex]*vv;
		}
	}
	for(i=0;i<dim;i++){solutions[i]=myValues[index[i]];}
	delete[] index;
	delete[] myValues;
	delete[] myEqns;
	delete[] set;
	return 1;
}
