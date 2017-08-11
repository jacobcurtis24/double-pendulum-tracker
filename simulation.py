from __future__ import division
from math import *
from cmath import *
import numpy as np
from numpy.linalg import det
import matplotlib.pyplot as plt

############################################################################################################
# Usage: use the function runSim(a,b) to run the simulation.
# Initial state is defined by global variable globalYINIT=(angle, p, angle, p)
# runSim returns plotX, plotT1, plotT2, which are: times, angle1, and angle2
# The stuff after that plots these, and finds the dispersion
############################################################################################################

def zeros(n,m):
	ret=[]
	for i in range(m):
		ret.append([0]*n)
	return ret

def lyapunov(Y):
	R=A12+B12*cos(Y[2]-Y[0])
	S=A11*A22-R**2
	T=A22*Y[1]-R*Y[3]
	U=-R*Y[1]+A11*Y[3]
	S_TERM=B12*sin(Y[2]-Y[0])
	C_TERM=B12*cos(Y[2]-Y[0])
	DQ=-(T*U*C_TERM+(T*Y[1]+U*Y[3])*S_TERM**2)/S**2+4.00*R*T*U*S_TERM**2/S**3
	J11=(S*Y[3]-2.00*A22*U)*S_TERM/S**2
	J33=(S*Y[1]-2.00*R*U)*S_TERM/S**2
	J12=A22/S
	J13=(S*Y[3]-2.00*R*T)*S_TERM/S**2
	J14=-R/S
	J21=-C1*cos(Y[0])+DQ
	J22=-J11
	J23=-DQ
	J24=J33
	J31=-J33
	J32=J14
	J34=A11/S
	J41=J23
	J42=-J13
	J43=-C2*cos(Y[2])+DQ
	J44=-J33
	J=[[J11,J12,J13,J14],[J21,J22,J23,J24],[J31,J32,J33,J34],[J41,J42,J43,J44]]
	C = det(J)
	B=-(0.50*(J[0][0]**2 + J[1][0]*J[0][1] + J[2][2]**2 + J[3][2]*J[2][3]) + J[1][3]*J[3][1] + J[1][2]*J[2][1])
	D=B**2-C
	if D >= 0.0:
		Z=[-B+sqrt(D), -B-sqrt(D)]
		ZZ=[complex(sqrt(np.max([Z[0],0.0])),sqrt(-np.min([Z[0],0.0]))),  complex(sqrt(np.max([Z[1],0.0])),sqrt(-np.min([Z[1],0.0])))]
		ROOTS=[ZZ[0].real,-ZZ[0].real,ZZ[1].real,-ZZ[1].real]
	else:
		RE=-B
		IM=sqrt(-D)
		SIM=sqrt(0.5*(-RE+sqrt(RE**2+IM**2)))
		SRE=IM/(2.0*SIM)
		ROOTS=[complex(SRE,SIM).real,complex(-SRE,-SIM).real,complex(SRE,-SIM).real,complex(-SRE,SIM).real]
	return ROOTS

def d_pend(Y,T):
	DY=np.array([0]*4, dtype=complex)
	R=A12+B12*cos(Y[2]-Y[0])
	S=A11*A22-R**2
	DY[0]=(A22*Y[1]-R*Y[3])/S
	DY[2]=(-R*Y[1]+A11*Y[3])/S
	Q=DY[0]*DY[2]*B12*sin(Y[2]-Y[0])
	DY[1]=-C1*sin(Y[0])+Q
	DY[3]=-C2*sin(Y[2])-Q
	return DY

def rk4(x,t,tau,derivsRK):
	half_tau = 0.5*tau
	F1 = derivsRK(x,t)  
	t_half = t + half_tau
	xtemp = x + half_tau*F1
	F2 = derivsRK(xtemp,t_half)  
	xtemp = x + half_tau*F2
	F3 = derivsRK(xtemp,t_half)
	t_full = t + tau
	xtemp = x + tau*F3
	F4 = derivsRK(xtemp,t_full)
	xout = x + tau/6.*(F1 + F4 + 2.*(F2+F3))

	return xout	

def runSim(disp1, disp2):
	global globalYINIT
	T_INIT = 0.0
	N_SELECT=2

	Y_INIT = globalYINIT+np.array([disp1, 0.0, disp2, 0.0])

	F_NAME='d_chaos_am'

	N_STEPS = 10000
	DT = 0.001
	T = T_INIT
	YA = Y_INIT
	TIME = np.arange(0,N_STEPS,DT)
	YM = zeros(4, N_STEPS+1)
	#YL = zeros(4, N_STEPS+1)
	#YH = zeros(4, N_STEPS+1)
	EIGEN = zeros(4, N_STEPS+1)
	YM[0]=YA
	EIGEN[0]=lyapunov(YA)

	plotT1=[]
	plotT2=[]
	plotX=[]

	for I in range(1,N_STEPS+1):
		#if I%int(N_STEPS/10)==0: print I, "/", N_STEPS
		YB = rk4(YA,T,DT,d_pend)
		T = T+DT
		YA = YB
		YM[I] = YA
		EIGEN[I] = lyapunov(YA)
		
		plotT1.append(YA[0].real)
		plotT2.append(YA[2].real)
		plotX.append(T)

	return plotX, plotT1, plotT2, EIGEN

M_SCALE=1.0
L_SCALE=1000.0

G     =    9802.894276/L_SCALE
L1    =     173.000000/L_SCALE
M1U   =   10123.312735/(M_SCALE*L_SCALE)
M1R   = 1262802.391807/(M_SCALE*L_SCALE**2)
M2    =     110.362483/M_SCALE
M2U   =    8200.761340/(M_SCALE*L_SCALE)
M2R   =  919788.854796/(M_SCALE*L_SCALE**2)
MB    =       7.369667/M_SCALE
MBRAA =      48.200077/(M_SCALE*L_SCALE**2)
MBRAB =      55.763811/(M_SCALE*L_SCALE**2)
MBRBB =     205.992010/(M_SCALE*L_SCALE**2)
MS    =       2.014000/M_SCALE
MSR   =      30.371120/(M_SCALE*L_SCALE**2)
MC    =       3.704000/M_SCALE
MCR   =      27.089327/(M_SCALE*L_SCALE**2)
MN    =       0.291000/M_SCALE
MNR   =       5.366072/(M_SCALE*L_SCALE**2)

A11= 2.0*M1R+4.0*MBRBB+(M2+2.0*MB+MC+2.0*MS+MN)*L1**2
A12= 2.0*MBRAB
A22= M2R+2.0*MBRAA+MCR+2.0*MSR+MNR
B12= M2U*L1
C1=  (2.0*M1U+(M2+2.0*MB+MC+2.0*MS+MN)*L1)*G
C2=  M2U*G

OMEGA_A = C2/A22
OMEGA_B = (C1-C2)/(A11+2.0*(A12-B12)+A22)
OMEGA_C = (C1+C2)/(A11+2.0*(A12+B12)+A22)




############################################################################################################
# Plot variables
############################################################################################################

#Starting points! angle1, p1, angle2, p2
globalYINIT_1=np.array([-pi*45/180,0,-pi*45/180,0])
globalYINIT_2=np.array([5*pi/12,0,5*pi/12,0])
globalYINIT_3=np.array([(pi/2)+42.8*pi/180,0,(pi/2)+9.1*pi/180,0])

#this is the initial starting point
globalYINIT=globalYINIT_1

# number of MC runs to do
N=10 #should be 100 

#variance on starting angles 
variance=0.0005

############################################################################################################
# Plot
############################################################################################################

f1 = plt.figure()
f2 = plt.figure()
ax1 = f1.add_subplot(111)
ax2 = f2.add_subplot(111)

ax1.set_title("Angles over Time\n$\Theta_1={0}\pi$, $\Theta_2={1}\pi$".format(globalYINIT[0]/pi,globalYINIT[2]/pi))
ax1.set_ylabel("Angle [radians]")
ax1.set_xlabel("Time [s]")

col=['r','g','b','y','c','m','k']
style=["-","--",":"]
metaT1=[]
metaT2=[]
lyapunov_exponents = []
first_exponent = []
count = 0
#run MC simulations
for i in range(N):
	print "run ", i
	disp1=np.random.normal(0,variance)
	disp2=np.random.normal(0,variance)
	#print "="*50,"\n",i, "/", N, disp1,disp2
	time, T1, T2, EIGEN=runSim(disp1,disp2)
	metaT1.append(T1)
	metaT2.append(T2)
	lyapunov_exponents.append(EIGEN)
	for t in lyapunov_exponents[i]:
		first_exponent.append(t[0])
		count += 1

	if i%10==0:
		ax1.plot(time,T1, "r-", label="$\Theta_1$")
		ax1.plot(time,T2, "b-", label="$\Theta_2$")
print "number of lyapunov exponents calculated: ", count
print "average lyapunov exponent is ", np.average(first_exponent)
print "with variance ", np.var(first_exponent)

#calculate dispersion
dispT1=[]
dispT2=[]
for t in range(len(time)):
	angleSliceT1=[]
	angleSliceT2=[]
	for i in range(len(metaT1)): angleSliceT1.append(metaT1[i][t])
	for i in range(len(metaT2)): angleSliceT2.append(metaT2[i][t])

	dispT1.append(np.std(angleSliceT1))
	dispT2.append(np.std(angleSliceT2))

ax2.set_title("Dispersion over Time\n$\Theta_1={0}\pi$, $\Theta_2={1}\pi$. $\sigma={2}, N={3}$".format(globalYINIT[0]/pi,globalYINIT[2]/pi, variance, N))
ax2.set_ylabel("Dispersion [radians]")
ax2.set_xlabel("Time [s]")

ax1.set_title("Angles over Time\n$\Theta_1={0}\pi$, $\Theta_2={1}\pi$. $\sigma={2}, N={3}$".format(globalYINIT[0]/pi,globalYINIT[2]/pi, variance, N))
ax1.set_ylabel("Angle [radians]")
ax1.set_xlabel("Time [s]")

ax2.plot(time,dispT1, "r-", label="Dispersion in $\Theta_1$")
ax2.plot(time,dispT2, "b-", label="Dispersion in $\Theta_2$")
ax2.legend(loc=2)
plt.show()















