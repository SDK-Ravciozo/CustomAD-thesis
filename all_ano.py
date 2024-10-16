import numpy as np

class AnoComplete:
    def __init__(self, nr:int, nb:int, decay=0.9, tw=50):
        self.nameAlg = 'AnoComplete'
        self.time = 1
        self.nr = nr
        self.nb = nb
        self.decay = decay
        self.param = np.random.randint(1,1<<16,2*nr).astype(int)
        self.matrix = np.zeros((nb,nb),int)
        self.tw = tw # time window
        
        #Inital randomly picked 1x1 submatirces:
        self.Scur = [np.random.randint(nb)]
        self.Tcur = [np.random.randint(nb)]
    
    
    def density(self, M, S, T) -> float:
        ''''''
        
        if not S or not T: return 0
        numerator = 0
        for s in S:
            for t in T:
                numerator += M[s][t]
                
        denominator = np.sqrt(len(S)*len(T))
        return numerator / denominator
    
    
    def rowsum(self, M, u:int, T) -> int:
        rowsum = 0
        for i in T:
            rowsum += M[u][i]
        return rowsum
        
        
    def colsum(self, M, S, v:int) -> int:
        colsum = 0
        for i in S:
            colsum += M[i][v]
        return colsum
    
    
    def call(self, u:int, v:int, t:int) -> float:
        ''''''
        
        if self.time < t:
            self.matrix = self.matrix*self.decay
            self.time = t
        u = ((347 * u) * self.param[0] + self.param[self.nr]) % self.nb
        v = ((347 * v) * self.param[0] + self.param[self.nr]) % self.nb
        self.matrix[u][v]+=1
        
        if self.name_alg == 'AnoEdgeLocal':
            self.Scur,self.Tcur = self.expand(self.matrix,self.Scur,self.Tcur,u,v)
            self.Scur,self.Tcur = self.condense(self.matrix,self.Scur,self.Tcur)
            return self.likelihood(self.matrix,self.Scur,self.Tcur,u,v)
        
        else:
            return self.EdgeSubDensity(self.matrix,u,v)

        
    def EdgeSubDensity(self, M, u:int, v:int) -> float:
        ''''''

        assert self.name_alg in ['AnoEdgeGlobal', 'AnoGraph'], \
        "The versions 'AnoEdgeLocal' and 'AnoGraph' do not use that method"

        S, Srem, T, Trem = list(range(self.nb)), list(range(self.nb)), list(range(self.nb)), list(range(self.nb))
        Scur, Tcur = [u], [v]
        Srem = Srem.remove(u)
        Trem = Trem.remove(v)
        dmax = self.density(M,Scur,Tcur)

        Stmp, Ttmp = Scur, Tcur # to be printed

        if self.name_alg == 'AnoGraph':
            while Srem or Trem:
                if Srem: up = S[np.argmax([rowsum(M,sp,Tcur) for sp in Srem])]
                if Trem: vp = T[np.argmax([colsum(M,Scur,tp) for tp in Trem])]
                if rowsum(M,up,Tcur)>colsum(M,Scur,vp):
                    Scur.append(up); Srem.remove(up)
                else:
                    Tcur.append(vp); Trem.remove(vp)
                dmax = max(dmax,self.density(M,Scur,Tcur))

                if dmax == self.density(M,Scur,Tcur):
                    Stmp,Ttmp = Scur,Tcur

        return dmax,Stmp,Ttmp
    
    
    
class AnoEdgeGlobal(AnoComplete):
    def __init__(self, nr:int, nb:int, decay=0.9, tw=50):
        super().__init__()
        self.name_alg = 'AnoEdgeGlobal'
             
    
class AnoEdgeLocal(AnoComplete):
    def __init__(self, nr:int, nb:int, decay=0.9, tw=50):
        super().__init__()
        self.nameAlg = 'AnoEdgeLocal'

    
    def likelihood(self,M,S,T,u:int,v:int)->float:
        numerator = 0
        denominator = 0
        for s in S:
            numerator+=M[s][v]
            denominator+=1
        for t in T:
            numerator+=M[u][t]
            denominator+=1
        return numerator/denominator
    
    def expand(self,M,S,T,u:int,v:int):
        S1 = S.copy(); T1 = T.copy()
        S1 = S1.append(u); T1 = T1.append(v)
        d0 = self.density(M,S,T)
        d1 = self.density(M,S1,T)
        d2 = self.density(M,S,T1)
        if d1>d0 and u not in S: S.append(u)
        if d2>d0 and v not in T: T.append(v)
        return S,T
    
    def condense(self,M,S,T):
        # increase or decrease?
        # delete untill decrease density
        while True:
            if len(S)==1 or len(T)==1: break
            if S: up = S[np.argmin([self.rowsum(M,sp,T) for sp in S])]
            if T: vp = T[np.argmin([self.colsum(M,S,tp) for tp in T])]
            S1 = S.copy(); T1 = T.copy()
            S1 = S1.remove(up); T1 = T1.remove(vp)
            d0 = self.density(M,S,T)
            d1 = self.density(M,S1,T)
            d2 = self.density(M,S,T1)
            if d0<d1 and d2<d1: S.remove(up)
            elif d0<d2 and d1<=d2: T.remove(vp)
            else: break
#             if d1<d0 and d2<d0: break
#             if d1>d2: S.remove(up)
#             elif d1<d2: T.remove(vp)
        return S,T
    
    
    
class AnoGraph(AnoComplete):
    def __init__(self, nr:int, nb:int, decay=0.5, tw=50):
        super().__init__()
        self.nameAlg = 'AnoGraph'
        
        
    def AnoGraphDensity(self,M,u:int,v:int)->float:
        # inintal
        Scur = list(range(self.nb)); Tcur = list(range(self.nb))
        dmax = self.density(M,Scur,Tcur)
        
        while Scur or Tcur:
            if Scur: up = Scur[np.argmin([self.rowsum(M,sp,Tcur) for sp in Scur])]
            if Tcur: vp = Tcur[np.argmin([self.colsum(M,Scur,tp) for tp in Tcur])]
            if self.rowsum(M,up,Tcur)<self.colsum(M,Scur,vp):
                Scur.remove(up)
            else:
                Tcur.remove(vp)
            dmax = max(dmax,self.density(M,Scur,Tcur))
        return dmax
    
    
class AnoGraph(AnoComplete):
    def __init__(self, nr:int, nb:int, decay=0.5, tw=50):
        super().__init__()
        self.nameAlg = 'AnoGraph'
    
    def AnographKDensity(self,M,k)->float:
        S = list(range(self.nb)); T = list(range(self.nb))
        dmax = 0
        for i in range(k):
            up,vp = divmod(np.argmax([M[s][t] for s in S for t in T]), len(S))
            dmax = max(dmax,self.EdgeSubDensity(M,up,vp))
        return dmax