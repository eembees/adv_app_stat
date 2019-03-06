import numpy as np


def CUSUM(arr, s, e, b, ):#thres):
    n = e - s + 1

    x = np.sqrt(
        (e-b)/(n*(b-s+1))
            )* np.sum(arr[s:b]) - \
        np.sqrt(
        (b-s+1)/(n*(e-b))
            )* np.sum(arr[b:e])
    return np.abs(x)

def getBSCUSUM(arr, s, e):
    # s_temp = 0
    # e_temp = e - s
    # bs = range(e_temp - 1)
    bs = range(len(arr))
    b_CUSUM = np.array([CUSUM(arr, s, e, b) for b in bs])
    return b_CUSUM

def getWBSCUSUM(arr, s, e):
    # s_temp = 0
    # e_temp = e - s
    # bs = range(e_temp - 1)
    # bs = range(len(arr))
    bs = range(e-s) + s
    b_CUSUM = np.array([CUSUM(arr, s, e, b) for b in bs])
    return b_CUSUM


def makeFMT(M, T, spacing :int= 3):
    s_M = np.random.randint(low=1, high=T-(spacing+1), size=M)
    e_M = np.ones_like(s_M)
    for m, s_m in enumerate(s_M):
        e_M[m] = np.random.randint(low=s_m+spacing, high=T)
    # FMT = np.concatenate([s_M,e_M], axis=1)
    FMT = np.transpose([s_M,e_M])
    # FMT = list(zip(s_M, e_M))
    return FMT



class BinSeg:
    def __init__(self, trace:np.ndarray, C:float=1.0, M:int = 5000):
        self.trace = trace
        self.thres = C * np.sqrt(2*np.log(len(trace)))
        self.bs    = []
        self.wbs   = []
        self.N_t   = len(trace)
        self.M     = M
        self.FMT   = makeFMT(M=self.M, T=self.N_t)

    def printParams(self):
        print('Threshold')
        print(self.thres)

    def updateThreshold(self, C):
        self.thres = C * np.sqrt(2*np.log(len(self.trace)))

    def resetAllBS(self):
        self.bs = []
        self.wbs = []

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
            # print('New run with s: {}  e: {}  '.format(s, e))
            # Find indices of working set
            M_se = np.argwhere((self.FMT[:,1]<=e) & (self.FMT[:,0]>=s) & (self.FMT[:,0]<self.FMT[:,1]) )
            # print('Made M_SE')

            # For each m get the max b and cumsum
            # all_CUSUMs = []
            bs_cands = []
            bs_sums  = []


            for m in M_se:
                se_pair = self.FMT[m]
                s_m = se_pair[0,0]
                e_m = se_pair[0,1]
                # print('Here is s_m  {}  and e_m  {}  '.format(s_m,e_m))

                # arr_temp = self.trace[s_m:e_m]

                m_CUSUM = getWBSCUSUM(self.trace, s = s_m, e = e_m)
                # all_CUSUMs.append(m_CUSUM)

                b_0_rel = np.nanargmax( m_CUSUM)
                b_0_true = b_0_rel + s_m

                b_CUSUM = m_CUSUM[b_0_rel]

                bs_cands.append(b_0_true)
                bs_sums.append(b_CUSUM)

            try:
                ind_max_cus = np.argmax(bs_sums)
                max_CUSUM = bs_sums[ind_max_cus]
                b_0 = bs_cands[ind_max_cus]

                if max_CUSUM > self.thres:
                    # print('Found event at {}'.format(b_0))
                    self.wbs.append(b_0)  # Don't add S because we get a masked arr now

                    self.updateWBS(s=s, e=b_0)
                    self.updateWBS(s=b_0 + 1, e=e)

            except ValueError:
                # print('something is wrong')
                pass
        '''
        if e - s >= 1:
            print('Now trying s={} and e={}'.format(s, e))
            arr = self.trace[s:e]
            b_CUSUM = getBSCUSUM(arr, s, e)
            b_0 = np.argmax(b_CUSUM)
            if b_CUSUM[b_0] > self.thres:
                print('Found event at {}'.format(b_0))
                self.bs.append(b_0 + s)  # Add s to make sure we have the right index
                self.updateBS(s=s, e=b_0)
                self.updateBS(s=b_0 + 1, e=e)
        '''

    #
    # def updateWBS(self, s=0, e=None):
    #     if e == None:
    #         e = len(self.trace) - 1
    #
    #     if e - s >= 1:
    #         print('New run with s: {}  e: {}  '.format(s, e))
    #         # Find indices of working set
    #         M_se = np.argwhere((self.FMT[:,1]<=e) & (self.FMT[:,0]>=s) )
    #         print('Made M_SE')
    #
    #
    #         # optional augmenting of set
    #
    #         # For each m get the cusum
    #         # all_CUSUMs = []
    #         bs_cands = []
    #         bs_CS    = []
    #
    #         for m in M_se:
    #             se_pair = self.FMT[m]
    #
    #             m_CUSUM = getBSCUSUM(self.trace, s = se_pair[0,0], e = se_pair[0,1])
    #             # all_CUSUMs.append(m_CUSUM)
    #             b_max = np.nanargmax( m_CUSUM)
    #
    #
    #             b_CUSUM = m_CUSUM[b_max]
    #
    #             bs_cands.append(b_max)
    #             bs_CS.append(b_CUSUM)
    #
    #
    #
    #
    #         print('Now trying to stack cumsums')
    #
    #         try:
    #             # arr_CUSUM = np.dstack(all_CUSUMs)[0]
    #             #
    #             # # Now the first index is b, second index is m
    #             # maxind = np.unravel_index(np.nanargmax(arr_CUSUM, axis=None), arr_CUSUM.shape)
    #             #
    #             # (b_0, m_0) = maxind
    #             #
    #             # print(self.FMT[m_0])
    #             # print(b_0)
    #             # exit()
    #             # max_CUSUM = arr_CUSUM[maxind]
    #
    #             ind_max_cus = np.argmax(bs_CS)
    #             max_CUSUM = bs_CS[ind_max_cus]
    #             b_0 = bs_cands[ind_max_cus]
    #
    #
    #
    #
    #             if max_CUSUM > self.thres:
    #                 print('Found event at {}'.format(b_0))
    #                 self.wbs.append(b_0)  # Don't add S because we get a masked arr now
    #
    #                 self.updateWBS(s=s, e=b_0)
    #                 self.updateWBS(s=b_0 + 1, e=e)
    #
    #         except ValueError:
    #             print('something is wrong')
    #             pass
    #







if __name__ == '__main__':
    import matplotlib.pyplot as plt

    sigma = 10
    # Generate data
    arr_pt_1 = np.random.normal(loc = 100, scale = sigma, size=23)
    arr_pt_2 = np.random.normal(loc = 170, scale = sigma, size=50)
    arr_pt_3 = np.random.normal(loc = 70, scale = sigma, size=50)
    arr_pt_4 = np.random.normal(loc = 20, scale = sigma, size=100-23)


    arrs = [
        arr_pt_1,
        arr_pt_2,
        arr_pt_3,
        arr_pt_4,
    ]

    trace = np.concatenate(arrs)

    C = 1

    fig, ax = plt.subplots()

    ax.plot(trace)


    #
    myBS = BinSeg(trace=trace, C=C, M=5000)
    #

    # myBS.updateBS()
    #
    # # myBS.printParams()
    #
    # print(myBS.bs)
    #
    # for xline in myBS.bs:
    #     ax.axvline(x = xline, ls='--', color='xkcd:dark red')

    #
    for C_i in [50, 70]:#,100,500,1000,5000]:
        print('This is with C = {}'.format(C_i))
        myBS.updateThreshold(C=C_i)
        myBS.printParams()
        myBS.resetAllBS()
        myBS.updateWBS()
        print(len(myBS.wbs))




    for xline in myBS.wbs:
        ax.axvline(x = xline, ls='--', color='xkcd:dark red')



    plt.show()

