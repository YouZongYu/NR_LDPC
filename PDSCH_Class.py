import numpy as np
from scipy.io import loadmat
from scipy.sparse import csr_matrix
from scipy.sparse import find

class mcs:
    def __init__(self):
        self.Z = None
        self.SetIndex = None
        self.BGN = None
        self.A = None
        self.Hsize = None
        self.outlen = None
        self.LTB = None
        self.maxIter = None
        self.Ha = None
        self.Hc = None
        self.Hd = None
        self.RowNum1 = None
        self.Num1 = None
        self.RowWiseIdx = None
        self.RowWiseIdxPerLayer = None
    def Setup(self, BGN, Z, LTB, maxiter):
        if BGN==1:
            self.Hsize = np.array([46, 68])
            MatData = loadmat('BG1.mat')
        else:
            self.Hsize = np.array([42, 52])
            MatData = loadmat('BG2.mat')

        self.A = Z*(self.Hsize[1]-self.Hsize[0]) - LTB
        self.outlen = Z*(self.Hsize[1]-2)
        RowIdx = MatData['RowIdx']
        ColIdx = MatData['ColIdx']
        PerMat = MatData['PerMat']
        RowIdx = RowIdx.astype('uint16')
        ColIdx = ColIdx.astype('uint16')
        PerMat = PerMat.astype('uint16')
        LiftTable = MatData['LiftTable']

        self.LTB = LTB
        self.BGN = BGN
        self.Z = Z
        self.maxIter = maxiter
        self.LDPCcodec(RowIdx, ColIdx, PerMat)
    def LDPCcodec(self, RowIdx, ColIdx, PerMat):
        temp = self.Z
        while temp%2==0:
            temp = temp>>1
            if temp==2:
                break
        SetIndex = int((temp-1)/2)
        rowidx = np.array([])
        colidx = np.array([])
        for i in range(0,RowIdx.shape[0]):
            rowidx = np.concatenate((rowidx,np.arange(RowIdx[i]*self.Z,(RowIdx[i]+1)*self.Z)))
            liftidx = np.arange(ColIdx[i]*self.Z,(ColIdx[i]+1)*self.Z)
            colidx = np.concatenate((colidx,np.roll(liftidx,self.Z-PerMat[i,SetIndex]%self.Z)))
        H = csr_matrix((np.ones(rowidx.shape), (rowidx,colidx)),shape=self.Hsize*self.Z,dtype=np.uint16)
        
        if self.BGN==1:
            self.Ha = H[0:self.Z*4,0:self.Z*22]
            self.Hc = H[self.Z*4:,0:self.Z*22]
            self.Hd = H[self.Z*4:,self.Z*22:self.Z*26]
        else:
            self.Ha = H[0:self.Z*4,0:self.Z*10]
            self.Hc = H[self.Z*4:,0:self.Z*10]
            self.Hd = H[self.Z*4:,self.Z*10:self.Z*14]

        self.RowNum1 = np.sum(H,axis=1)
        self.Num1 = np.sum(self.RowNum1)
        idx0,idx1= find(H.T)[:2]
        self.RowWiseIdx = idx1 * H.shape[1] + idx0
        self.RowWiseIdxPerLayer = idx0
        self.SetIndex = SetIndex