import numpy as np
from scipy.sparse import csr_matrix
from numba import jit

def nrLDPCencode(encIn,mcs):
    if mcs.BGN==1 and mcs.SetIndex!=6:
        EncType = 1
    elif mcs.BGN==1 and mcs.SetIndex==6:
        EncType = 2
    elif mcs.BGN==2 and mcs.SetIndex!=3 and mcs.SetIndex!=7:
        EncType = 3
    else:
        EncType = 4

    u = encIn.astype('uint16')
    tmp = ((mcs.Ha @ u).T % 2).reshape(4,-1)
    Lambda = tmp.T

    if EncType==1 or EncType==4 :
        P11 = np.sum(Lambda,axis=1) %2
    elif EncType==2:
        P11 = np.roll(np.sum(Lambda,axis=1),105%mcs.Z) %2
    else:
        P11 = np.roll(np.sum(Lambda,axis=1),1) %2

    if EncType==1 or EncType==4 :
        P12 = np.logical_xor(Lambda[:,0], np.roll(P11,-1))
    else:
        P12 = np.logical_xor(Lambda[:,0], P11)

    if EncType==1 or EncType==4:
        P14 = np.logical_xor(Lambda[:,3], np.roll(P11,-1))
    else:
        P14 = np.logical_xor(Lambda[:,3], P11)

    if EncType==3 or EncType==4 :
        P13 = np.logical_xor(Lambda[:,1], P12)
    else:
        P13 = np.logical_xor(Lambda[:,2], P14)

    P1 = np.concatenate((P11,P12,P13,P14)).reshape(-1,1)
    P2 = (mcs.Hc@u + mcs.Hd@P1)%2
    enc = np.concatenate((u[2*mcs.Z:],P1,P2))
    return enc

@jit
def nrLDPCdecode(LLR,Z,Num1,maxIter,Hsize,RowNum1,RowWiseIdxPerLayer):
    rxLLR = np.concatenate((np.zeros((2*Z,1)),LLR))
    R = np.zeros((Num1,1))
    SoftOut = rxLLR
    for k in range(0,maxIter):
        count = 0
        for i in range(0,Z*Hsize[0]):
            NumOneLayer = np.int64(RowNum1[i,0])
            tidx = np.arange(count,count+NumOneLayer)
            Qseg = SoftOut[RowWiseIdxPerLayer[tidx]] - R[tidx]
            Sseg = Qseg<0
            Vseg = -np.log(np.tanh(np.abs(Qseg)/2))
            infidx = np.where(np.isinf(Vseg))[0]
            if infidx.size==1:
                oidx = np.concatenate((np.arange(0,infidx[0]),np.arange(infidx[0]+1,NumOneLayer)))
                tS = (-1)**(np.sum(Sseg[oidx])%2)
                tV = np.sum(Vseg[oidx])
                R[count+infidx] = -tS*np.log(np.tanh(tV/2))
            elif infidx.size==0:
                tV = np.sum(Vseg)-Vseg
                tS = (-1)**((np.sum(Sseg)%2)+Sseg)
                R[tidx] = -tS*np.log(np.tanh(tV/2))
            count += NumOneLayer
            if not np.any(np.isnan(R[tidx])):
                SoftOut[RowWiseIdxPerLayer[tidx]] = Qseg + R[tidx]
        FinalOut = SoftOut.copy()
        dec = FinalOut<0
        count = 0
        check = 0
        for i in range(0,Z*Hsize[0]):
            NumOneLayer = np.int64(RowNum1[i,0])
            tidx = np.arange(count,count+NumOneLayer)
            check = np.sum(dec[RowWiseIdxPerLayer[tidx]])%2
            if check :
                break
            count += NumOneLayer
        if not check :
            break

    return dec[:Z*(Hsize[1]-Hsize[0])]

# def nrLDPCdecode(LLR,mcs):
#     rxLLR = np.concatenate((np.zeros((2*mcs.Z,1)),LLR))
#     R = np.zeros((mcs.Num1,1))
#     SoftOut = rxLLR
#     for k in range(0,mcs.maxIter):
#         count = 0
#         for i in range(0,mcs.Hsize[0]*mcs.Z):
#             tidx = np.arange(count,count+mcs.RowNum1[i])
#             Qseg = SoftOut[mcs.RowWiseIdxPerLayer[tidx]] - R[tidx]
#             Sseg = Qseg<0
#             Vseg = -np.log(np.tanh(np.abs(Qseg)/2))
#             infidx = np.where(np.isinf(Vseg))[0]
#             if infidx.size==1:
#                 oidx = np.setdiff1d(np.arange(0,mcs.RowNum1[i]),infidx)
#                 tS = (-1)**(np.sum(Sseg[oidx])%2)
#                 tV = np.sum(Vseg[oidx])
#                 R[count+infidx] = -tS*np.log(np.tanh(tV/2))
#             elif infidx.size==0:
#                 tV = np.sum(Vseg)-Vseg
#                 tS = (-1)**((np.sum(Sseg)%2)+Sseg)
#                 R[tidx] = -tS*np.log(np.tanh(tV/2))
#             count += mcs.RowNum1[i]
#             SoftOut[mcs.RowWiseIdxPerLayer[tidx]] = Qseg + R[tidx]
#         if any(np.isnan(SoftOut)):
#             break
#         FinalOut = SoftOut.copy()
#     dec = FinalOut<0
#     return dec[:mcs.Z*(mcs.Hsize[1]-mcs.Hsize[0])]