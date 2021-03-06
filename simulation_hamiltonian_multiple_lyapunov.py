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

def runSim(Y_INIT):
	global globalYINIT
	T_INIT = 0.0


	F_NAME='d_chaos_am'

	N_STEPS = 10000
	DT = 0.001
	T = T_INIT
	YA = Y_INIT
	TIME = np.arange(0,N_STEPS,DT)
	YM = zeros(4, N_STEPS+1)
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
		
		plotT1.append(YA)
		plotX.append(T)

	return plotX, plotT1, EIGEN

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
# Plot
############################################################################################################

f1 = plt.figure()
ax1 = f1.add_subplot(111)

# ax1.set_title("Angles over Time\n$\Theta_1={0}\pi$, $\Theta_2={1}\pi$".format(globalYINIT[0]/pi,globalYINIT[2]/pi))
# ax1.set_ylabel("Angle [radians]")
# ax1.set_xlabel("Time [s]")

col=['r','g','b','y','c','m','k']
style=["-","--",":"]
selected_positions1 = []
selected_positions2 = []
#run one simulation
init_pos = [(-60 / 180 * np.pi, 0, -60 / 180 * np.pi, 0), (-55 / 180 * np.pi, 0, -55 / 180 * np.pi, 0), (-65 / 180 * np.pi, 0, -65 / 180 * np.pi, 0)]
time = []
runs = []
for i in init_pos:
	time, T1, eigen = runSim(i)
	eigen = np.array(eigen)
	runs.append(eigen)

lyapunov_exps = np.zeros((3, len(time)))
for k, i in enumerate(runs):
	for j in range(len(time)):
		if j == 0:
			lyapunov_exps[k][j] = np.real(i[0][0])
		else:
			lyapunov_exps[k][j] = np.real(np.mean(i[0:j, 0]))
			
ax1.set_title("Lyapunov with slightly varying initial conditions.")
ax1.set_ylabel("Lyapunov exponent")
ax1.set_xlabel("Time [s]")
ax1.plot(time, lyapunov_exps[0, :], "r-", label="60 degrees")
ax1.plot(time, lyapunov_exps[1, :], "b-", label="55 degrees")
ax1.plot(time, lyapunov_exps[2, :], "g-", label="65 degrees")
ax1.legend(loc='best')

plt.show()

# make a plot of several hamilonians at 45 deg +/- a few degrees