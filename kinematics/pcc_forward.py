import numpy as np
import matplotlib.pyplot as plt



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

def arc1_point(T1,s1_hole,d):

    arc1_points=[]
    for i in range(len(T1)):# num=5 for s1 set above
        #print('T is',np.reshape(T[i, :], (4, 4), order='F'))
        for j in range(len(s1_hole)): # three holes per disk
            x = d * np.cos(s1_hole[j])
            y = d * np.sin(s1_hole[j])
            z = 0
            p = np.matmul(np.reshape(T1[i, :], (4, 4), order='F'), np.array([x, y, z, 1]))
            arc1_points.append(p)

    return arc1_points

def arc2_point(T2_cc,T2,s2_hole,d):
    
    arc2_points = arc1_point(T2_cc,s2_hole,d)
    for i in range(len(T2)):
        for j in range(len(s2_hole)):
            x = d * np.cos(s2_hole[j])
            y = d * np.sin(s2_hole[j])
            z = 0
            p = np.matmul(np.reshape(T2[i, :], (4, 4), order='F'), np.array([x, y, z, 1]))
            arc2_points.append(p)
    return arc2_points


def cable_len(T1_hole,T2_hole):
    l1_len, l2_len, l3_len, l4_len, l5_len, l6_len = 0, 0, 0, 0, 0, 0
    T1_reshaped = np.array(T1_hole).reshape(5, 3, 4)

    for i in range(4):
        l1_len += np.linalg.norm(T1_reshaped[i+1, 0, :3] - T1_reshaped[i, 0, :3])
        l2_len += np.linalg.norm(T1_reshaped[i+1, 1, :3] - T1_reshaped[i, 1, :3])
        l3_len += np.linalg.norm(T1_reshaped[i+1, 2, :3] - T1_reshaped[i, 2, :3])

    T2_reshaped = np.array(T2_hole).reshape(10, 3, 4)
   
    for i in range(9):        
        l4_len += np.linalg.norm(T2_reshaped[i+1, 0, :3] - T2_reshaped[i, 0, :3])
        l5_len += np.linalg.norm(T2_reshaped[i+1, 1, :3] - T2_reshaped[i, 1, :3])
        l6_len += np.linalg.norm(T2_reshaped[i+1, 2, :3] - T2_reshaped[i, 2, :3])
    
    return l1_len,l2_len,l3_len, l4_len, l5_len, l6_len


def specific2(l1, l2, l3, d):
    # Specific mapping for section2
    l = 1/3 * (l1 + l2 + l3)
    if l2 != l1:
        A = (np.sqrt(6)*(l-l3))/(np.sqrt(2)*(l2-l1))
    else:
        A = (np.sqrt(6)*(l-l3))/(np.sqrt(2)*(0.001))
        
    x = A + 1
    y = 1 - A
    phi = np.arctan2(y, x)
    c_phi = x/(np.sqrt(x**2 + y**2))
    s_phi = y/(np.sqrt(x**2 + y**2))
    c_phi_1 = ((np.sqrt(6) - np.sqrt(2))/4 * c_phi + (np.sqrt(2) + np.sqrt(6))/4 * s_phi)
    k = (l - l1)/(l * d * c_phi_1)
    
    return k, phi

def specific1(l1, l2, l3, d):
    # Specific mapping for section1 
    l = 1/3 * (l1 + l2 + l3)
    #A = (l - l3)/(l2 - l1)
    if l2 != l1:
        A = (l - l3) / (l2 - l1)
    else:
        A = (l - l3) / 0.001
    
    x = (3*np.sqrt(2) + np.sqrt(6))*A - np.sqrt(2) + np.sqrt(6)
    y = np.sqrt(6) + np.sqrt(2) - (3*np.sqrt(2) - np.sqrt(6))*A
    phi = np.arctan2(y, x)
    c_phi = x/(np.sqrt(x**2 + y**2))
    s_phi = y/(np.sqrt(x**2 + y**2))
    c_phi_1 = ((np.sqrt(2) - np.sqrt(6))/4 * c_phi + (np.sqrt(2) + np.sqrt(6))/4 * s_phi)
    k = (l - l1)/(l * d * c_phi_1)
    
    return k, phi

#def get_points(l1,l2,l3,l4,l5,l6):
def get_points(cab_lens):       
        l1 = cab_lens[0]
        l2 = cab_lens[1]
        l3 = cab_lens[2]
        l4 = cab_lens[3]
        l5 = cab_lens[4]
        l6 = cab_lens[5]

        d = 0.35285
        beta = np.deg2rad([75, 195, 315])
        L2_1,L2_2,L2_3 = 0,0,0

        # Center axis length section1
        l_1 = 1/3 * (l1 + l2 + l3)
        # Calculate k1 and phi1 using specific1 function
        k1, phi1 = specific1(l1, l2, l3, d)

        # Create arc points for section2 part1 using obtained k1 and phi1
        T1 = trans_matrix(k1,l_1,phi1) #get transformation matrix reshaped in [1*16] in n array within length l and with size
        B1 = arc1_point(T1,beta,d)
        # Find the arc length for section2 part1
        B1_reshaped = np.array(B1).reshape(5, 3, 4)
    
        for i in range(4):
             L2_1 += np.linalg.norm(B1_reshaped[i+1, 0, :3] - B1_reshaped[i, 0, :3])
             L2_2 += np.linalg.norm(B1_reshaped[i+1, 1, :3] - B1_reshaped[i, 1, :3])
             L2_3 += np.linalg.norm(B1_reshaped[i+1, 2, :3] - B1_reshaped[i, 2, :3])


    # Find the arc length for section2 part2 by subtraction
        l4_2 = l4 - L2_1
        l5_2 = l5 - L2_2
        l6_2 = l6 - L2_3
        l_2 = 1/3 * (l4_2 + l5_2 + l6_2)
    # Calculate k2 and phi2 using specific2 function
        k2, phi2 = specific2(l4_2, l5_2, l6_2, d)
                      
        l =[l_1,l_2]
        points = two_section_robot(k1, k2, l, phi1, phi2)
        tip_x,tip_y,tip_z = np.array([points[0,3],points[1,3],points[2,3]])


        return tip_x,tip_y,tip_z   


