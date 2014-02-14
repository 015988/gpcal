#    This file is part of EAP.
#
#    EAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    EAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with EAP. If not, see <http://www.gnu.org/licenses/>.

import operator
import math
import random
import inspect
import numpy
import time
import os
import string
import deap
import pickle
import copy
from numpy.linalg import norm, svd
from numpy import sin, cos, dot, cross, eye
from scipy.optimize import fmin_powell,fmin
from datetime import datetime

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
from scoop import futures


#from pylab import plot,show,savefig,grid


class treeRootType:
    """Root type for the tree, so we can have 2 trees as one"""
    #seemed easier than to try to change the individual type
    def __init__(self,u,v):
        self.u=u
        self.v=v
        
class calibData:
    """Data type for world and image points """
    #I didnt find a better way to represent a 4d structure with non uniform size
    def __init__(self,dat,ref,H,A,O,T,f_init,c_init,imW,imH):
       self.dat=dat
       self.ref=ref
       self.H=H
       self.A=A
       self.O=O
       self.T=T
       self.f_init=f_init
       self.c_init=c_init
       self.imW=imW
       self.imH=imH

#@profile
def treeRoot(u,v):
    """The function that is placed at the root of the tree """
    return treeRootType(u,v)

#@profile
def QtoR(Q):
    """
    From a quaternion Q, returns the rotation matrix R SO3
    """
    w,x,y,z=Q
    Nq = w**2.0 + x**2.0 + y**2.0 + z**2.0
    if (Nq > 0.0):
        s = 2/Nq 
    else:
        s = 0.0

    X, Y ,Z = x*s, y*s, z*s
    wX, wY, wZ = w*X, w*Y, w*Z
    xX, xY, xZ = x*X, x*Y, x*Z
    yY, yZ, zZ = y*Y, y*Z, z*Z
    return numpy.array([[ 1.0-(yY+zZ),       xY-wZ,        xZ+wY  ],
                        [      xY+wZ,   1.0-(xX+zZ),       yZ-wX  ],
                        [      xZ-wY,        yZ+wX,   1.0-(xX+yY) ]]).T
#@profile
def RtoQ(m):
    """
    Returns a quaternion in the right versor sense. - Robert Love in a random internet page
    """

    q = numpy.zeros(4,float)
    
    q[0] = 0.5*numpy.sqrt(1.0 + m[0,0] + m[1,1] + m[2,2])

    q04_inv = 1.0/(4.0*q[0])
    q[1] = (m[1,2] - m[2,1])*q04_inv
    q[2] = (m[2,0] - m[0,2])*q04_inv
    q[3] = (m[0,1] - m[1,0])*q04_inv
    return q

#@profile
def loadValues(datFile,refFile, imsFile):
    """
    Read a pair of files
    """

    s1=numpy.loadtxt(imsFile)
    imW=s1[0]
    imH=s1[1]

    
    d1=numpy.loadtxt(datFile)
    dat=numpy.ndarray(shape=(d1.shape[0],d1.shape[1]/2.0,2))
    dat[:,:,0]=d1[:,::2]
    dat[:,:,1]=d1[:,1::2]
    dat=numpy.compress(dat[:,0,0]!=-1,dat,axis=0);
    dat[:,:,0]=dat[:,:,0]/imW
    dat[:,:,1]=dat[:,:,1]/imH

    r1=numpy.loadtxt(refFile)
    ref=numpy.ndarray(shape=(r1.shape[0],r1.shape[1]/3.0,3))
    ref[:,:,0]=r1[:,::3]
    ref[:,:,1]=r1[:,1::3]
    ref[:,:,2]=r1[:,2::3]
    ref=numpy.compress(ref[:,0,0]!=-1,ref,axis=0);
    
    return dat,ref, 1, 1

#@profile
def extractVanishingPoints(Hkk,c_init):

    # matrix that subtract the principal point:
    Sub_cc = numpy.matrix([[1.0, 0.0, -c_init[0]],[0.0, 1.0, -c_init[1]],[0.0, 0.0, 1.0]])
    
    Hkk = Sub_cc * Hkk

    # Extract vanishing points (direct and diagonals):

    V_hori_pix = Hkk[:,0]
    V_vert_pix = Hkk[:,1]
    V_diag1_pix = (Hkk[:,0]+Hkk[:,1])/2.0
    V_diag2_pix = (Hkk[:,0]-Hkk[:,1])/2.0

    V_hori_pix = V_hori_pix/norm(V_hori_pix)
    V_vert_pix = V_vert_pix/norm(V_vert_pix)
    V_diag1_pix = V_diag1_pix/norm(V_diag1_pix)
    V_diag2_pix = V_diag2_pix/norm(V_diag2_pix)
    
    a1 = V_hori_pix[0]
    b1 = V_hori_pix[1]
    c1 = V_hori_pix[2]
    
    a2 = V_vert_pix[0]
    b2 = V_vert_pix[1]
    c2 = V_vert_pix[2]
    
    a3 = V_diag1_pix[0]
    b3 = V_diag1_pix[1]
    c3 = V_diag1_pix[2]
    
    a4 = V_diag2_pix[0]
    b4 = V_diag2_pix[1]
    c4 = V_diag2_pix[2]
    
    A_r = numpy.array([[(a1*a2)[0,0], (b1*b2)[0,0]],[(a3*a4)[0,0],  (b3*b4)[0,0]]]) #there is probably a better way to 'translate' matrix -> number than [0,0]
    
    b_r = -numpy.array([[(c1*c2)[0,0]],[(c3*c4)[0,0]]])
    
    return A_r, b_r

#@profile
def fixShape(vIn):
    if (vIn.shape[0]>vIn.shape[1]):
       return vIn.T
    else:
        return vIn


#@profile
def ComputeHomography(W,I):
    W=fixShape(W)
    I=fixShape(I)
    wt=W[2];
    wt[wt<1e-3]=1.0;
    W[2]=wt;

    nPoints=W.shape[1]
    A=numpy.zeros((3*nPoints,9))
    for p in range(0,nPoints-1):
        A[3*p+0]=[  0.0,            0.0,            0.0,          -W[0,p],       -W[1,p],       -W[2,p],        I[1,p]*W[0,p],  I[1,p]*W[1,p],  I[1,p]*W[2,p]]
        A[3*p+1]=[  W[0,p],         W[1,p],         W[2,p],        0.0,           0.0,           0.0,          -I[0,p]*W[0,p], -I[0,p]*W[1,p], -I[0,p]*W[2,p]]
        A[3*p+2]=[ -I[1,p]*W[0,p], -I[1,p]*W[1,p], -I[1,p]*W[2,p], I[0,p]*W[0,p], I[0,p]*W[1,p], I[0,p]*W[2,p], 0.0,            0.0,            0.0 ]
       

    u, s, vh = svd(A)
    H=numpy.reshape(vh[8,:]/vh[8,8],(3,3))
    return H

#@profile
def normalize_pixel(x_kk,fc,cc):
   return numpy.array([(x_kk[:,0] - cc[0])/fc[0], (x_kk[:,1] - cc[1])/fc[1]]).T


#@profile
def loadAllData(nFrames):
    """
    Loads and computes everything (files, H, A, RTs) before computation.
    Returns only nFrames per camera/(dat,ref)
    """
    curData=[]
    for f in os.listdir("."):
        if f.endswith(".dat"):
            print(str(datetime.now())+" - Loading "+ f)
            dTemp,rTemp,imWidth,imHeight=loadValues(f,f.replace('dat','ref'),f.replace('dat','ims'))
            c_init = numpy.array([imWidth, imHeight])/2.0 - 0.5;
            curH=[]
            usedLines=0
            V=numpy.zeros((2*dTemp.shape[0],2))
            b=numpy.zeros((2*dTemp.shape[0],1))
            print( str(datetime.now())+" - Computing Hs")
            for p in range(0,dTemp.shape[0]):
                curH.append(ComputeHomography(rTemp[p,:,:],dTemp[p,:,:]))
                A_kk, b_kk=extractVanishingPoints(curH[-1],c_init)
                V[(2*(usedLines)):2*(usedLines+1)]=A_kk;
                b[(2*(usedLines)):2*(usedLines+1)]=b_kk;
                usedLines+=1

            print( str(datetime.now())+" - Estimating fc")
            Vt,r1,r2,s=numpy.linalg.lstsq(numpy.dot(V.T,V),V.T)
            f_init = numpy.sqrt(numpy.abs(1/numpy.dot(Vt,b)))
            curA = numpy.array([[f_init[0][0], 0, c_init[0]],[0, f_init[1][0], c_init[1]],[ 0, 0, 1]])
            
            print(str(datetime.now())+" - Computing all frame errors")
            dtype = [('index', 'int'), ('error', float)]
            values=[]
            for p in range(0,dTemp.shape[0]):
                o,t=computeExtrinsic(dTemp[p],rTemp[p],f_init,c_init,optimize=False)
                tErr=frameError(dTemp[p],rTemp[p],[curA[0,0], curA[1,1], curA[0,2], curA[1,2], 0],o,t)
                if (not math.isnan(tErr)):
                    values.append((p,tErr))

            allErrors=numpy.array(values,dtype)               
            allErrors=numpy.sort(allErrors,order='error')
            picks=allErrors['index'][-(nFrames+1):-1]

            print( str(datetime.now())+" - Reducing frames")
            newD=numpy.zeros((nFrames,dTemp.shape[1],dTemp.shape[2]))
            newR=numpy.zeros((nFrames,rTemp.shape[1],rTemp.shape[2]))
            newH=[]
            usedLines=0;
            for p in picks:
                newD[usedLines]=dTemp[p]
                newR[usedLines]=rTemp[p]
                usedLines+=1
                newH.append(curH[p])

            dTemp=newD
            rTemp=newR
            curH=newH

            print (str(datetime.now())+" - Computing O Ts")
            cO,cT=[],[]
            for p in range(0,dTemp.shape[0]):
                oTemp,tTemp=computeExtrinsic(dTemp[p],rTemp[p],f_init,c_init,optimize=True)
                cO.append(oTemp)
                cT.append(tTemp)
            
            
            curData.append(calibData(dTemp, rTemp, curH, curA, cO, cT, f_init, c_init, imWidth, imHeight))
    return curData


#@profile
def frameError(cI, cW, curIntrinsic, curO, curT, funcModel=None,dispPlot=False):
    cI=fixShape(cI)
    cW=fixShape(cW)

    W=numpy.ones((4,cW.shape[1]));
    W[0:3]=cW

    RT=numpy.vstack([QtoR(curO).T,curT]).T
    curA=numpy.array([[curIntrinsic[0], curIntrinsic[4], curIntrinsic[2]],
                    [0,               curIntrinsic[1], curIntrinsic[3]],
                    [0,               0,               1]])
    xl=numpy.dot(RT,W)
    xl=xl/xl[2]
    if (funcModel!=None):
        TR=funcModel(xl[0],xl[1])
        if (dispPlot):
            plot(xl[0],xl[1],'ob')
            grid(True)
            plot(xl[0],xl[1],'+r')
            savefig("test.png")
            show()
        xl[0]-=TR.u
        xl[1]-=TR.v
    xl=numpy.dot(curA,xl)
    xl=xl[0:2]/xl[2]
    
    return numpy.sum((numpy.sum(numpy.abs(xl-cI))));

#@profile
def fcn_opt_OT(xIn,A,curI,curW):
    return frameError(curI, curW, [A[0,0], A[1,1], A[0,2], A[1,2], 0], xIn[0:4], xIn[4:7])
#@profile
def composeVector(A,dist,O,T):
    
    v=[A[0,0]/1000, A[1,1]/1000, A[0,2]/1000, A[1,2]/1000, A[0,1]]
    for d in range(0,len(dist)):
        v.extend(dist[d].tolist())
    
    for i in range(0,len(O)):
        v.extend((O[i]).tolist())
        v.extend((T[i]/1000).tolist())
    
    
    return v
#@profile
def decomposeVector(v,nDist):
    A=numpy.array([[v[0]*1000, v[4], v[2]*1000],[0, v[1]*1000, v[3]*1000],[0,0,1]])
    dist=v[5:(5+nDist)]
    nFrames=int((len(v)-5-nDist)/7) #5 from A, 7= 4 from O + 3 from T
    O=[]
    T=[]
    st=5+nDist;
    for f in range(0,nFrames):
        O.append(numpy.array(v[(st+7*f)  :(st+7*f)+4]))
        T.append(numpy.array(v[(st+7*f)+4:(st+7*f)+7])*1000)
        
        
    
    return A,dist,O,T

def simplifyInd(sInd):
    sInd=sInd.replace(r"'T'","T")
    sInd=sInd.replace(r"0.0","T")
    while True:
        n=len(sInd)
        sInd=sInd.replace("add(T, T)","T")
        sInd=sInd.replace("multiply(T, T)","T")
        sInd=sInd.replace("true_divide(T, T)","T")
        sInd=sInd.replace("subtract(T, T)","T")
        sInd=sInd.replace("r2(T, T)","T")
        sInd=sInd.replace("sqrt(T)","T")
        sInd=sInd.replace("x2(T)","T")
        sInd=sInd.replace("negative(T)","T")


        sInd=sInd.replace("r2(u, T)","multiply(T, x2(u))")
        sInd=sInd.replace("r2(T, u)","multiply(T, x2(u))")
        sInd=sInd.replace("r2(v, T)","multiply(T, x2(v))")
        sInd=sInd.replace("r2(T, v)","multiply(T, x2(v))")

        sInd=sInd.replace("add(u, u)","multiply(u, T)")
        sInd=sInd.replace("add(v, v)","multiply(v, T)")
        sInd=sInd.replace("subtract(u, u)","T")
        sInd=sInd.replace("subtract(v, v)","T")
        sInd=sInd.replace("true_divide(u, u)","T")
        sInd=sInd.replace("true_divide(v, v)","T")
        sInd=sInd.replace("r2(u, u)","multiply(T, x2(u))")
        sInd=sInd.replace("r2(v, v)","multiply(T, x2(v))")

        if (len(sInd)==n):
            break
    return sInd


#@profile
def fcn_opt_all(xIn, dat, ref,nConst,individual,pset,dispPlot=False):
    A,dist,O,T=decomposeVector(xIn,nConst)
    retErr=0

    newInd=[]
    nConst=0
    
    for x in individual:
        if (isinstance(x,deap.gp.Terminal) and (x.value=='T')):
            x.value=dist[nConst]
            newInd.append(x)
            nConst+=1
        else:
            newInd.append(x)

    func = toolbox.lambdify(expr=newInd, pset=pset)
    for f in range(0,len(O)):
        retErr+=frameError(dat[f],ref[f],[A[0,0], A[1,1], A[0,2], A[1,2], A[0,1]],O[f],T[f],func,dispPlot)
    return retErr/len(O)

#@profile
def evalSymbReg(individual, curData, pset):
    nConst=0
    newInd=copy.deepcopy(individual)
    tIndStr=simplifyInd(deap.gp.stringify(newInd))
    newInd.from_string(string=tIndStr,pset=pset)
    for x in newInd:
        if (isinstance(x,deap.gp.Terminal) and (x.value=='T')):
            nConst+=1

    dist=numpy.zeros((nConst,1))
    retErr=0;
    start_time=time.time()

    for d in range(0,len(curData)):   
        x0=composeVector(curData[d].A, dist,  curData[d].O, curData[d].T)
        nX,fErr,direc,nIt,nFun,warnflag=fmin_powell(func=fcn_opt_all,x0=x0,args=(curData[d].dat,curData[d].ref,nConst,newInd,pset),full_output=True,disp=False)#,ftol=1)#,xtol=0.1)#,maxfun=750)
        #        fcn_opt_all(nX,curData[d].dat,curData[d].ref,nConst,newInd,pset,dispPlot=True)
        if (math.isnan(fErr) or math.isinf(fErr)):
            return (numpy.inf,)
        retErr+=fErr

#        print( fErr, nFun, ellapsed_time, (ellapsed_time)/nFun, len(nX))
#    ellapsed_time=time.time()-start_time
#    print( str(datetime.now())+" dt  "+str(ellapsed_time) +" err: "+ str(retErr/len(curData)) +" "+ str(deap.gp.stringify(individual)))
    return (retErr/len(curData),)

#@profile
def computeExtrinsic(curI,curW,f_init,c_init,optimize=False):
    curI=fixShape(curI)
    curW=fixShape(curW)
    wt=curW[2]
    wt[wt==0]=1.0
    curW[2]=wt

    
    X_kk=curW;

    xn = normalize_pixel(curI.T,f_init,c_init).T
    Np = xn.shape[1]
    
    # Check for planarity of the structure:
    X_mean = numpy.mean(X_kk,axis=1)
    R_transform = numpy.eye(3) 
    T_transform = -numpy.dot(R_transform,X_mean)
    X_new = numpy.dot(R_transform,X_kk) + numpy.dot(T_transform,numpy.ones((3,Np)))
    # Compute the planar homography:
    H = ComputeHomography(X_new,xn);
    # De-embed the motion parameters from the homography:
    sc = numpy.mean([norm(H[:,0]), norm(H[:,1])])
    H = H/sc
        
    u1 = H[:,0]
    u1 = u1 / norm(u1)
    u2 = H[:,1] - numpy.dot(u1,H[:,1]) * u1
    u2 = u2 / norm(u2)
    u3 = numpy.cross(u1,u2)
    RRR = numpy.array([u1.tolist(), u2.tolist(), u3.tolist()]).T #dont ask why the T
    Rckk = QtoR(RtoQ(RRR));
    Tckk = H[:,2]
        
    #If Xc = Rckk * X_new + Tckk, then Xc = Rckk * R_transform * X_kk + Tckk + T_transform
    cT = Tckk + numpy.dot(Rckk,T_transform)
    Rckk = Rckk * R_transform
    cO = RtoQ(Rckk)
    A = numpy.array([[f_init[0][0], 0, c_init[0]],[0, f_init[1][0], c_init[1]],[ 0, 0, 1]])
#    frameError(curI,W,[A[0,0], A[1,1], A[0,2], A[1,2], 0],cO,cT)
    if (optimize):
        nX=fmin_powell(func=fcn_opt_OT,x0=numpy.concatenate((cO, cT)),args=(A,curI,curW),disp=False)
        cO=nX[0:4]
        cT=nX[4:7]
    return cO,cT

       
def r2(a,b):
    return numpy.add(a**2,b**2)
def x2(a):
    return a**2
#def invX(a):
#    a=[1/x if (x>0) else 0 for x in a]
#    return a



def myEaSimple(population, toolbox, cxpb, mutpb, ngen, stats=None, printInd=False):
    """Slightly modified version of deap.algorithms.eaSimple. This one can output the individuals to stdout"""
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit


    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen+1):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))
        
        # Vary the pool of individuals
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)
        
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
       
            
        # Replace the current population by the offspring
        population[:] = offspring
        
        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)

        print(logbook.stream)

        
        if printInd:
            f = open('curLog.txt', 'a')
            f.write("--------- generation "+str(gen)+" "+str(datetime.now())+"\n")
            for ind in population:
                f.write(str(ind.fitness.values[0]) +" "+ str(simplifyInd(deap.gp.stringify(ind)))+'\n')
            f.close()

    return population, logbook




                 

nFrames=5
ngen=10
npop=100
rSeed=13


pset = gp.PrimitiveSetTyped("MAIN", [float, float], ret_type=treeRootType)
pset.addPrimitive(treeRoot,         [float, float], treeRootType)
pset.addPrimitive(numpy.multiply,   [float, float], float)
pset.addPrimitive(numpy.add,        [float, float], float)
pset.addPrimitive(numpy.subtract,   [float, float], float)
pset.addPrimitive(numpy.divide,     [float, float], float)
pset.addPrimitive(r2,               [float, float], float)
pset.addPrimitive(numpy.sqrt,       [float],        float)
pset.addPrimitive(x2,               [float],        float)
pset.addPrimitive(numpy.negative,   [float],        float)
#pset.addPrimitive(invX,             [float],        float)


pset.addTerminal('T',float)

pset.renameArguments(ARG0="u",ARG1="v")


numpy.seterr(all='ignore')

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin, pset=pset)

toolbox = base.Toolbox()
toolbox.register("map", futures.map) # par. processing!
toolbox.register("expr", gp.genRamped, pset=pset, min_=2, max_=5, type_=treeRootType)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("lambdify", gp.lambdify, pset=pset)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=2, max_=5, type_=treeRootType)
toolbox.register('mutate', gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

#@profile
def main():
    print (str(datetime.now())+" - Starting main")
    random.seed(rSeed)

    print (str(datetime.now())+" - Loading files")
    try:
        print (str(datetime.now())+" - Loading saved values")
        pkl_file = open('data.pkl', 'rb')
        curData = pickle.load(pkl_file)
    except:
        print (str(datetime.now())+" - Loading failed, computing values")
        curData=loadAllData(nFrames)
        output = open('data.pkl', 'wb')
   
        # Pickle the list using the highest protocol available.
        pickle.dump(curData, output, -1)

    print (str(datetime.now())+" - loaded...")
    toolbox.register("evaluate", evalSymbReg, curData=curData, pset=pset) # <-fitness here    



    print (str(datetime.now())+" - Starting population")
    start_time = time.time()
    pop = toolbox.population(n=npop)
    t1=pop.pop()
    t2=pop.pop()
    t3=pop.pop()
    tN=pop.pop()
    
    pop.insert(0,t1.from_string(string='treeRoot(T, T)',pset=pset))
    pop.insert(0,t2.from_string(string='treeRoot(multiply(T,r2(u,v)),multiply(T,r2(u,v)))',pset=pset))
    pop.insert(0,t3.from_string(string='treeRoot(add(multiply(T,u),multiply(T,v)),add(multiply(T,u),multiply(T,v)))',pset=pset))
               
    pop.insert(0,tN.from_string(string='treeRoot(add(add(multiply(multiply(u,T),r2(u,v)),multiply(multiply(u,T),x2(r2(u,v)))), add(add(multiply(T,x2(u)),multiply(T,x2(v))), add(multiply(multiply(T,u),v),multiply(T,r2(u,v))))),add(add(multiply(multiply(v,T),r2(u,v)),multiply(multiply(v,T),x2(r2(u,v)))), add(add(multiply(T,x2(u)),multiply(T,x2(v))), add(multiply(multiply(T,u),v),multiply(T,r2(u,v))))))',pset=pset))

    #pop.insert(0,t1.from_string(string='treeRoot(add(T, T),multiply(T, T)',pset=pset))

    
#    tempInd=creator.Individual([pset.mapping['treeRoot'], pset.mapping['add'], pset.mapping['multiply'],pset.mapping['multiply'], pset.mapping['T'], pset.mapping['u'], pset.mapping['r2'], pset.mapping['u'], pset.mapping['v'], pset.mapping['add'], pset.mapping['add'], pset.mapping['multiply'], pset.mapping['T'], pset.mapping['r2'], pset.mapping['u'], pset.mapping['v'],pset.mapping['add'], pset.mapping['multiply'], pset.mapping['T'], pset.mapping['x2'], pset.mapping['u'], pset.mapping['multiply'], pset.mapping['T'], pset.mapping['x2'], pset.mapping['v'],pset.mapping['add'], pset.mapping['multiply'], pset.mapping['multiply'], pset.mapping['T'], pset.mapping['u'], pset.mapping['x2'], pset.mapping['r2'], pset.mapping['u'], pset.mapping['v'], pset.mapping['multiply'], pset.mapping['T'], pset.mapping['multiply'], pset.mapping['u'], pset.mapping['v'], pset.mapping['add'], pset.mapping['multiply'], pset.mapping['multiply'], pset.mapping['T'], pset.mapping['v'], pset.mapping['r2'],pset.mapping['u'], pset.mapping['v'],pset.mapping['add'],pset.mapping['add'], pset.mapping['multiply'], pset.mapping['T'], pset.mapping['r2'], pset.mapping['u'], pset.mapping['v'],pset.mapping['add'], pset.mapping['multiply'],  pset.mapping['T'], pset.mapping['x2'], pset.mapping['u'], pset.mapping['multiply'], pset.mapping['T'], pset.mapping['x2'], pset.mapping['v'], pset.mapping['add'], pset.mapping['multiply'], pset.mapping['multiply'], pset.mapping['T'], pset.mapping['v'], pset.mapping['x2'], pset.mapping['r2'], pset.mapping['u'], pset.mapping['v'], pset.mapping['multiply'], pset.mapping['T'], pset.mapping['multiply'], pset.mapping['u'], pset.mapping['v']])
#    pop.insert(0,tempInd)



    print( time.time() - start_time)
    
    print (str(datetime.now())+" - Registering more stuff")

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", min)
    stats.register("max", max)
    
    print (str(datetime.now())+" - Starting eaSimple")
    start_time=time.time()
    pop=myEaSimple(population=pop,toolbox=toolbox,cxpb=0.4,mutpb = 0.4, ngen=ngen, stats=stats, printInd=True)
    print ((time.time())-start_time)

    return pop, stats

if __name__ == "__main__":
    main()
