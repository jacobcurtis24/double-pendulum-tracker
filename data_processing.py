from __future__ import division
from math import *
from cmath import *
import numpy as np
from numpy.linalg import det
import matplotlib.pyplot as plt
import sys
import csv
import time

def lyapunov(Y):
	R=A12+B12*cos(Y[2]-Y[0])
	S=A11*A22-R**2
	T=A22*Y[1]-R*Y[3]
	U=-R*Y[1]+A11*Y[3]
	S_TERM=B12*sin(Y[2]-Y[0])
	C_TERM=B12*cos(Y[2]-Y[0])
	DQ=-(T*U*C_TERM+(T*Y[1]+U*Y[3])*S_TERM**2)/S**2+4.00*R*T*U*S_TERM**2/S**3
	J11 = (S*Y[3]-2.00*A22*U)*S_TERM/S**2
	J33 = (S*Y[1]-2.00*R*U)*S_TERM/S**2
	J12 = A22/S
	J13 = (S*Y[3]-2.00*R*T)*S_TERM/S**2
	J14 = -R/S
	J21 = -C1*cos(Y[0])+DQ
	J22 = -J11
	J23 = -DQ
	J24 = J33
	J31 = -J33
	J32 = J14
	J34 = A11/S
	J41 = J23
	J42 = -J13
	J43 = -C2*cos(Y[2])+DQ
	J44 = -J33
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

M_SCALE=1.0
L_SCALE=1000.0
#L_SCALE=1

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

A11= 2.0*M1R+4.0*MBRBB+(M2+2.0*MB+MC+2.0*MS+MN)*(L1**2)
A12= 2.0*MBRAB
A22= M2R+2.0*MBRAA+MCR+2.0*MSR+MNR
B12= M2U*L1
C1=  (2.0*M1U+(M2+2.0*MB+MC+2.0*MS+MN)*L1)*G
C2=  M2U*G

OMEGA_A = C2/A22
OMEGA_B = (C1-C2)/(A11+2.0*(A12-B12)+A22)
OMEGA_C = (C1+C2)/(A11+2.0*(A12+B12)+A22)

# set this to the time between frames
DT = 1 / 240

def get_csv_physlet(name):
	data_list = []
	with open(name, 'rb') as csvfile:
		inputdata = csv.reader(csvfile, delimiter = ',', quotechar = '|')
		# use count to ignore first two lines
		count = 0
		for row in inputdata:
			if count > 1:
				# convert tracker number format (string type) to number
				data_row = []
				for value in row:
					number = value.split('E', 1)
					if len(number) == 2:
						data_row.append(float(number[0]) * np.power(10, float(number[1])))
					else:
						data_row.append(0)
				data_list.append(data_row)
			count += 1
	return data_list

def get_csv(name):
	data_list = []
	with open(name, 'rb') as csvfile:
		inputdata = csv.reader(csvfile, delimiter = ',', quotechar = '|')
		for row in inputdata:
			data_row = []
			for value in row:
				data_row.append(float(value))
			data_list.append(data_row)
	return data_list

def make_csv(name, data):
	file_name = name
	f = open(file_name, 'wb')
	f.close()
	with open(file_name, 'wb') as csvfile:
		writedata = csv.writer(csvfile, delimiter = ',', quotechar = '|')
		for k in data:
			writedata.writerow(k)


def analyze(primary_positions, secondary_positions, video_name):
	# subtract one since we index forward one
	max_index = min(len(primary_positions), len(secondary_positions)) - 1

	# format of data list: time, theta_1, theta_1 dot, theta_2, theta_2 dot
	data = np.zeros((max_index, 5))

	data_size = 0
	i = 0

	# convert positions to angle measurements (this is just subtracting positions)
	secondary_rotations = 0
	while i < max_index:
		# add time
		data[data_size][0] = primary_positions[i][0]
		# add theta_1
		theta_1 = np.arctan2(primary_positions[i][2], primary_positions[i][1]) + np.pi / 2
		if theta_1 > np.pi:
			theta_1 -= 2 * np.pi
		data[data_size][1] = theta_1
		# add theta_2
		theta_2 = np.arctan2(secondary_positions[i][2] - primary_positions[i][2], secondary_positions[i][1] - primary_positions[i][1]) + np.pi / 2
		if theta_2 > np.pi:
			theta_2 -= 2 * np.pi
		if data[data_size - 1][3] - 2 * np.pi * secondary_rotations > 0 and theta_2 < 0 and theta_2 < -np.pi / 2:
			secondary_rotations += 1
		data[data_size][3] = theta_2 + secondary_rotations * 2 * np.pi
		data_size += 1
		i += 1

	# derivatives calculated average the left and right derivatives
	# could improve this by using the 5 point derivative
	i = 1
	while i < data_size - 1:
		data[i][2] = (data[i + 1][1] - data[i - 1][1]) / (2 * DT)
		data[i][4] = (data[i + 1][3] - data[i - 1][3]) / (2 * DT)
		i += 1

	# trim off first row that has uninitialized zero velocity
	data = data[range(1, len(data)), :]

	# will now calculate the Hamiltonian
	hamiltonian = np.zeros((data.shape[0], 2))
	for i, k in enumerate(data):
		hamiltonian[i][0] = k[0]
		H = A11 * np.power(k[2], 2) / 2 + (A12 + B12 * np.cos(k[3] - k[1])) * k[2] * k[4] + A22 * np.power(k[4], 2) / 2 - C1 * np.cos(k[1]) - C2 * np.cos(k[3])
		hamiltonian[i][1] = H

	# will now calculate the lyapunov exponents
	lyapunov_exp = np.zeros((data.shape[0], 5))
	
	for i, k in enumerate(data):
		lyapunov_exp[i][0] = k[0]
		p1 = A11 * k[2] + (A12 + B12 * np.cos(k[3] - k[1])) * k[4]
		p2 = A22 * k[4] + (A12 + B12 * np.cos(k[3] - k[1])) * k[2]
		L = lyapunov([k[1], p1, k[3], p2])
		for p in range(4):
			lyapunov_exp[i][p + 1] = L[p]

	lyapunov_avg = np.zeros((data.shape[0], 5))
	for k in range(1, data.shape[0]):
		lyapunov_avg[k][0] = lyapunov_exp[k][0]
		lyapunov_avg[k][1:5] = np.average(lyapunov_exp[0:k, 1:5], axis=0)

	print "First Lyapunov Exponent: ", np.average(lyapunov_exp[:, 1])
	print "Second Lyapunov Exponent: ", np.average(lyapunov_exp[:, 2])
	print "Third Lyapunov Exponent: ", np.average(lyapunov_exp[:, 3])
	print "Fourth Lyapunov Exponent: ", np.average(lyapunov_exp[:, 4])
	print "Initial Hamiltonian Value: ", hamiltonian[0][1]
	print "Final Hamiltonian Value: ", hamiltonian[hamiltonian.shape[0] - 1][1]
	print "Expected Final Hamiltonian: ", -C1 - C2

	# write data to csv files
	make_csv(video_name + "_raw_lyapunov_data.csv", lyapunov_exp)
	make_csv(video_name + "_averaged_lyapunov_data.csv", lyapunov_avg)
	make_csv(video_name + "_hamiltonian_data.csv", hamiltonian)
	make_csv(video_name + "_position_data.csv", data)

	# make plots
	f1 = plt.figure()
	ax1 = f1.add_subplot(111)
	ax1.set_ylabel("Hamiltonian")
	ax1.set_xlabel("Time [s]")
	ax1.plot(hamiltonian[:, 0], hamiltonian[:, 1], "r.", label="$\Theta_1$")


	f2 = plt.figure()
	ax2 = f2.add_subplot(111)
	ax2.set_ylabel("Secondary Position [radians]")
	ax2.set_xlabel("Time [s]")
	ax2.plot(data[:,0], data[:,3], "r-", label="$\dot{\Theta_2}$")

	f3 = plt.figure()
	ax3 = f3.add_subplot(111)
	ax3.set_ylabel("Cumulative average Lyapunov Exponents")
	ax3.set_xlabel("Time [s]")
	ax3.plot(lyapunov_exp[:,0], lyapunov_avg[:,1], "r-", label="$\lambda_1$")

	plt.show()