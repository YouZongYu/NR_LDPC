import numpy as np

def nrCRCencode(encIn, CRCtype):
    if CRCtype=='24A':
        numCRC = 24
        poly = np.zeros(numCRC+1)
        poly[numCRC-np.array([24, 23, 18, 17, 14, 11, 10, 7, 6, 5, 4, 3, 1, 0])] = 1
    elif CRCtype=='24B':
        numCRC = 24
        poly = np.zeros(numCRC+1)
        poly[numCRC-np.array([24, 23, 6, 5, 1, 0])] = 1
    elif CRCtype=='24C':
        numCRC = 24
        poly = np.zeros(numCRC+1)
        poly[numCRC-np.array([24, 23, 21, 20, 17, 15, 13, 12, 8, 4, 2, 1, 0])] = 1
    elif CRCtype=='16':
        numCRC = 16
        poly = np.zeros(numCRC+1)
        poly[numCRC-np.array([16, 12, 5, 0])] = 1
    elif CRCtype=='11':
        numCRC = 16
        poly = np.zeros(numCRC+1)
        poly[numCRC-np.array([11, 10, 9, 5, 0])] = 1
    else:
        numCRC = 6
        poly = np.zeros(numCRC+1)
        poly[numCRC-np.array([6, 5, 0])] = 1

    enc = encIn.astype('bool')
    poly = poly.reshape(-1,1)
    encZP = np.concatenate((enc,np.zeros((numCRC,1),dtype='bool')))
    for i in range(0,enc.shape[0]):
        if i==0:
            tmp = encZP[0:numCRC+1]
        else:
            tmp = np.append(dif[1:],encZP[numCRC+i]).reshape(-1,1)
        
        if(tmp[0]==True):
            dif = np.logical_xor(tmp,poly)
        else:
            dif = tmp

    CRCbits = dif[1:]
    encOut = np.concatenate((enc,CRCbits)).reshape(-1,1)
    return encOut.astype('uint16')

