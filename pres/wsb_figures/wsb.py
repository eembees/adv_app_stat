import numpy as np


def CUSUM(arr, s, e, b, ):#thres):
    n = e - s + 1
    x = np.sqrt(
        (e-b)/(n*(b-s+1))
    )* np.sum(arr[s:b]) - \
        np.sqrt(
            (b-s +1)/(n*(e-b))
        )*np.sum(arr[b:e])
    return np.abs(x)

def getBSCUSUM(arr, s, e):
    bs = range(e)
    b_CUSUM = np.array([CUSUM(arr, s,e,b) for b in bs])
    return b_CUSUM


# def BinSeg(arr,thres, s=0, e=None, bs:list=[]):
#     if e == None:
#         e = len(arr) - 1
#
#     if e - s < 1:
#         return bs
#
#     trace = arr[s:e+1]
#     b_CUSUM = getBSCUSUM(trace, s, e)
#     b_0 = np.argmax(b_CUSUM)
#     if b_CUSUM[b_0] > thres:
#         bs.append(b_0+s) # Add s to make sure we have the right index
#         bs = BinSeg(arrs[s:s+b_0],thres=thres, s=s, e=b_0, bs=bs)
#         bs = BinSeg(arrs[s+b_0:e],thres=thres, s=s+b_0, e=e,bs=bs)
#     else:
#         return bs

def makeFMT(M, T):
    s_M = np.random.randint(low=1, high=T, size=M)
    e_M = np.ones_like(s_M)
    for m, s_m in enumerate(s_M):
        e_M[m] = np.random.randint(low=s_M[m], high=T)
    # FMT = np.concatenate([s_M,e_M], axis=1)
    FMT = np.transpose([s_M,e_M])
    # FMT = list(zip(s_M, e_M))
    return FMT



class BinSeg:
    def __init__(self, trace:np.ndarray, C:float=1.0, M:int = 5000):
        self.trace = trace
        self.thres = C * np.sqrt(np.log(len(trace)))
        self.bs    = []
        self.wbs    = []
        self.N_t   = len(trace)
        self.M     = M
        self.FMT   = makeFMT(M=self.M, T=self.N_t)

    def printParams(self):
        print('Threshold')
        print(self.thres)

    def updateBS(self, s=0, e=None):
        if e == None:
            e = len(self.trace) - 1

        if e - s >= 1:
            print('Now trying s={} and e={}'.format(s, e))
            arr = self.trace[s:e]
            b_CUSUM = getBSCUSUM(arr, s, e)
            b_0 = np.argmax(b_CUSUM)
            if b_CUSUM[b_0] > self.thres:
                print('Found event at {}'.format(b_0))
                self.bs.append(b_0 + s)  # Add s to make sure we have the right index
                self.updateBS(s=s, e=b_0)
                self.updateBS(s=b_0+1, e=e)

    def updateWBS(self, s=0, e=None):
        if e == None:
            e = len(self.trace) - 1

        if e - s >= 1:

            # Find indices of working set
            M_se = np.argwhere((self.FMT[:,1]<=e) & (self.FMT[:,0]>=s) )


            # M_se = np.argwhere((self.FMT[:,0]>=s and self.FMT[:,1]<=e))
            # print(M_se)

            # M_se = [ se_pair for se_pair in self.FMT if (se_pair[0]>=s and se_pair[1]<=e) ]
            # optional augmenting of set
            #
            # arr = self.trace[s:e]
            # b_CUSUM = getBSCUSUM(arr, s, e)
            # b_0 = np.argmax(b_CUSUM)
            # if b_CUSUM[b_0] > self.thres:
            #     self.wbs.append(b_0 + s)  # Add s to make sure we have the right index
            #     self.updateWBS(s=s, e=b_0)
            #     self.updateWBS(s=b_0+1, e=e)





if __name__ == '__main__':
    # Generate data
    arr_pt_1 = np.random.normal(loc = 100, scale = 50, size=23)
    arr_pt_2 = np.random.normal(loc = 170, scale = 50, size=50)
    arr_pt_3 = np.random.normal(loc = 70, scale = 50, size=50)
    arr_pt_4 = np.random.normal(loc = 50, scale = 50, size=100-23)


    arrs = [
        arr_pt_1,
        arr_pt_2,
        arr_pt_3,
        arr_pt_4,
    ]

    trace = np.concatenate(arrs)

    C = 1.3

    myBS = BinSeg(trace=trace, C=C,M=100)

    # myBS.printParams()
    # myBS.updateBS()
    myBS.updateWBS(s=170, e=199)

    # print(myBS.bs)

    # print(set(makeFMT(10, 10)))
