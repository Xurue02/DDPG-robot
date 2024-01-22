import numpy as np
import matplotlib.pyplot as plt
import math


def trans_matrix(k, l, phi):
    
    si=np.linspace(0,l, num = 6);
    T= np.zeros((len(si),16));
    
    for i in range(len(si)):
        s=si[i];
        c_ks=np.cos(k*s);
        s_ks=np.sin(k*s);
        c_phi=np.cos(phi);
        s_phi=np.sin(phi);
        c_p=(1-c_ks)/k if k != 0 else 0;
        s_p=s_ks/k if k != 0 else s;

        Ry=np.array([c_ks,0,s_ks,c_p,0,1,0,0,-s_ks,0,c_ks,s_p,0,0,0,1]).reshape(4,4)
        #print('Ry\n',Ry);
        Rz=np.array([c_phi,-s_phi,0,0,s_phi,c_phi,0,0,0,0,1,0,0,0,0,1]).reshape(4,4)
        #print('Rz\n',Rz)
        Rz2=np.array([c_phi,s_phi,0,0,-s_phi,c_phi,0,0,0,0,1,0,0,0,0,1]).reshape(4,4)
        #print('Rz2\n',Rz2)

        if k == 0:
            Ry=np.array([c_ks,0,s_ks,c_p,0,1,0,0,-s_ks,0,0,s,0,0,0,1]).reshape(4,4)

        T_01 = np.matmul(np.matmul(Rz, Ry), Rz2) #Transformation matrix
        T[i, :] = np.reshape(T_01, (1, T_01.size), order='F') #reshape the matrix to 1 row, size of T column
        
    return T

def multiple_trans_matrix(T2, T_tip):
    
    Tc=np.zeros((len(T2[:,0]),len(T2[0,:])));
    for k in range(len(T2[:,0])):
        #Tc[k,:].reshape(-1,1)
        p = np.matmul(T_tip,(np.reshape(T2[k,:],(4,4),order='F')))
        Tc[k,:] = np.reshape(p,(16,),order='F');
    return Tc

def two_section_robot(k1, k2, l, phi1, phi2):
    '''
    * Homogeneous transformation matrix :k to x,y
    * Mapping from configuration parameters to task space for the tip of the continuum robot
    
    Parameters
    ----------
    k1 : float
        curvature value for section 1.
    k2 : float
        curvature value for section 2.
    l : list
        cable length contains all sections

    Returns
    -------
    T: numpy array
        4*4 Transformation matrices containing orientation and position
    '''
    c_ks1, s_ks1, c_phi1, s_phi1 = np.cos(k1 * l[0]), np.sin(k1 * l[0]), np.cos(phi1), np.sin(phi1)
    c_ks2, s_ks2, c_phi2, s_phi2 = np.cos(k2 * l[1]), np.sin(k2 * l[1]), np.cos(phi2), np.sin(phi2)

    c_p1, s_p1 = ((1 - c_ks1) / k1, s_ks1 / k1) if k1 != 0 else (0, l[0])
    c_p2, s_p2 = ((1 - c_ks2) / k2, s_ks2 / k2) if k2 != 0 else (0, l[1])
    
    Ry1 = np.array([c_ks1, 0, s_ks1, c_p1, 0, 1, 0, 0, -s_ks1, 0, c_ks1, s_p1, 0, 0, 0, 1]).reshape(4, 4)
    Rz1 = np.array([c_phi1, -s_phi1, 0, 0, s_phi1, c_phi1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]).reshape(4, 4)
    Rz2_1 = np.array([c_phi1, s_phi1, 0, 0, -s_phi1, c_phi1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]).reshape(4, 4)

    Ry2 = np.array([c_ks2, 0, s_ks2, c_p2, 0, 1, 0, 0, -s_ks2, 0, c_ks2, s_p2, 0, 0, 0, 1]).reshape(4, 4)
    Rz2 = np.array([c_phi2, -s_phi2, 0, 0, s_phi2, c_phi2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]).reshape(4, 4)
    Rz2_2 = np.array([c_phi2, s_phi2, 0, 0, -s_phi2, c_phi2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]).reshape(4, 4)

    # Directly calculate the combined transformation matrix
    T_1 = np.matmul(np.matmul(Rz1, Ry1), Rz2_1)
    #print('T_1',T_1)
    T_2 = np.matmul(np.matmul(Rz2, Ry2), Rz2_2)
    #print('T_2',T_2)
    T_combined = np.matmul(T_1, T_2)
    
    return T_combined
print('succuss')
    


