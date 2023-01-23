import os
import pickle
import sys
if __name__=="__main__":
    sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))
    sys.path.pop(0)

from lsvae.model import build_lsvae
import matplotlib.pyplot as plt
from attrdict import AttrDict
checkpoint_path = sys.argv[1]

params = pickle.load(open(f"{checkpoint_path}/params.pk", "rb"))
state = pickle.load(open(f"{checkpoint_path}/state.pk", "rb"))
config = pickle.load(open(f"{checkpoint_path}/config.pk", "rb"))
config = AttrDict(config)
print(params.keys())

import haiku as hk
import numpy as np

def grid_samples():
    if 'pendulum' in config.dataset:
        from datasets.pendulum.pendulum import render_pendulum, ImageSurface, Context, Format
        surface = ImageSurface(Format.ARGB32, 64, 64)
        ctx = Context(surface)
        for theta in np.arange(-3.1415, 3.14159, 0.05):
            state = np.array([theta, 0])
            img = render_pendulum(surface, ctx, 64, 64, state)
            yield {
                'images': img,
                'states': state
            }
    elif 'airsim' in config.dataset:
        import tensorflow_datasets as tfds
        data = tfds.load(config.dataset, split='grid')
        for s in tfds.as_numpy(data):
            yield {
                'images': s['images'][0],
                'states': s['states'][0]
            }

def generate_initial(theta):
    from datasets.pendulum.pendulum import render_pendulum, ImageSurface, Context, Format
    surface = ImageSurface(Format.ARGB32, 64, 64)
    ctx = Context(surface)
    state = np.array([theta, 0])
    img = render_pendulum(surface, ctx, 64, 64, state)
    return {
                'images': img,
                'states': state
            }

def f():
    lsvae = build_lsvae(config, 0)

    def encode(meas):
        return lsvae.obs_models[0].encode(meas, False, [])

    def init(x):
        pass

    def decode(meas):
        return lsvae.obs_models[0].decode(meas,False,[])



    return init, (encode, decode)

f = hk.multi_transform_with_state(f)

encode, decode = f.apply

def bivariateColor(Z1,Z2,cmap1 = plt.cm.YlOrRd, cmap2 = plt.cm.PuBuGn):
    Z1 = np.array(Z1)
    Z2 = np.array(Z2)
    # Rescale values to fit into colormap range (0->255)
    Z1_plot = np.array(255*(Z1-Z1.min())/(Z1.max()-Z1.min() + 1e-6), int)
    Z2_plot = np.array(255*(Z2-Z2.min())/(Z2.max()-Z2.min() + 1e-6), int)

    Z1_color = cmap1(Z1_plot)
    Z2_color = cmap2(Z2_plot)

    # Color for each point
    Z_color = np.sum([Z1_color, Z2_color], axis=0)/2.0

    return Z_color

xs = []
ys = []
color_x = []
color_y = []
b_x = []
b_y = []
b_cx = []
b_cy = []

rng = hk.PRNGSequence(42)
x = generate_initial(1.68)
dist, _ = encode(params, state, None, x)
initial_x = dist.multi_sample(next(rng), 1)[0]



###############################################
###############################################
###############################################
#############################################
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
#import control
import polytope as pc
from pytope import Polytope
n = 2 # dimension of parameters to be estimated
x0= initial_x
A=np.array([[0.99, 0.1], [0, 0.998]], dtype=np.float32)
B=np.array([0, 0.2], dtype=np.float32)#
BR=np.array([[0], [0.2]], dtype=np.float32)
Q=np.array([[10,0],[0,1]])#np.eye(n)#
Qf=1*Q
R=0.05#1#
AI=np.linalg.inv(A)
#Now, compute the X_f!
F=np.array([[1,0],[-1,0],[0,1],[0,-1]])
pi=np.pi
tdl=8#20#4#tdl for theta dot limit
f=np.array([pi,pi,tdl,tdl])
poly = Polytope(F, f)
#print(poly)
# plt.figure(100)
# plt.title('region of F')
# aa=poly.plot()
N=200#100#50#10#
# plt.xlim([-pi,N])
# plt.ylim([-15,N])
NOD=20#50#100#10#NOD means No., or Number#I think 100 is close enough
Fap=F#FAP means F*A^POWER

oht=1#0.01#
G=np.array([[oht],[-oht]])

tk=tdl#1#00#tk for ten kilo
g=np.array([[tk],[tk]])

Fi=F#Fi for F intersection
fi=f#fi for f intersection
for i in range(NOD):
    Fap=np.matmul(Fap,A)
    FB=np.matmul(Fap,B)
    FBR=FB.reshape((FB.shape[0],1))
    line1=np.concatenate((Fap,FBR),axis=1)
    zeros=np.zeros((G.shape[0],Fap.shape[1]))
    line2=np.concatenate((zeros,G),axis=1)
    Bigf=np.concatenate((line1,line2),axis=0)
    fr=f.reshape((f.shape[0],1))
    Bigfg=np.concatenate((fr,g),axis=0)
    xu=pc.Polytope(Bigf,Bigfg)
    xp=xu.project([1,2])
    Fnew=xp.A
    fnew=xp.b
    Fap=Fnew
    f=fnew
    Fi=np.concatenate((Fi,Fap),axis=0)#Fi for F intersection
    fi=np.concatenate((fi,f),axis=0)#
    polyi = Polytope(Fi, fi)
    # plt.figure(i)
    # aa=polyi.plot()
    # plt.title('region of Cinfty '+str(i))
    # plt.xlim([-pi,pi])
    # plt.ylim([-tdl,tdl])
#xlim=10
W=np.array([[1,0],[-1,0],[0,1],[0,-1]])
tb=1e-3
tdb=1.4#1.9#tdb for theta dot bound
fw=np.array([tb,tb,tdb,tdb])
polyw = Polytope(W, fw)
#print(poly)
# plt.figure(101)
# aa=polyw.plot()
# plt.title('region of W')
# plt.xlim([-5,5])
# plt.ylim([-5,5])
 
polyawA=np.matmul(polyw.A,AI)
polyaw=Polytope(polyawA,polyw.b)
# plt.figure(102)
# aw=polyaw.plot()
# plt.title('region of AW')
# plt.xlim([-5,5])
# plt.ylim([-5,5])

polya2wA=np.matmul(polyaw.A,AI)
polya2w=Polytope(polya2wA,polyw.b)   
# plt.figure(103)
# a2w=polya2w.plot()
# plt.title('region of A^2W')
# plt.xlim([-5,5])
# plt.ylim([-5,5])

polya3wA=np.matmul(polya2w.A,AI)
polya3w=Polytope(polya3wA,polyw.b)  
# plt.figure(104)
# a3w=polya3w.plot()
# plt.title('region of A^3W')
# plt.xlim([-5,5])
# plt.ylim([-5,5])

polya4wA=np.matmul(polya3w.A,AI)
polya4w=Polytope(polya4wA,polyw.b)  
# plt.figure(105)
# a4w=polya4w.plot()
# plt.title('region of A^4W')
# plt.xlim([-5,5])
# plt.ylim([-5,5])

pmw=polyi-polyw
# plt.figure(201)
# amw=pmw.plot()
# plt.title('region of PMW')
# plt.xlim([-pi,pi])
# plt.ylim([-tdl,tdl])

pmaw=pmw-polyaw
# plt.figure(202)
# amaw=pmaw.plot()
# plt.title('region of PMAW')
# plt.xlim([-pi,pi])
# plt.ylim([-tdl,tdl])

pma2w=pmaw-polya2w
# plt.figure(203)
# ama2w=pma2w.plot()
# plt.title('region of PMA2W')
# plt.xlim([-pi,pi])
# plt.ylim([-tdl,tdl])

pma3w=pma2w-polya3w
# plt.figure(204)
# ama3w=pma3w.plot()
# plt.title('region of PMA3W')
# plt.xlim([-pi,pi])
# plt.ylim([-tdl,tdl])

pma4w=pma3w-polya4w
# plt.figure(205)
# ama4w=pma4w.plot()
# plt.title('region of PMA4W')
# plt.xlim([-pi,pi])
# plt.ylim([-tdl,tdl])

p = 5#20 # number of available types of measurements
Rk=np.eye(p)*R
np.random.seed(0)

ul=[]#ul for u list
fl=[]#fl for f list
xl=[]#xl for x list
for t in range(N):
    u = cp.Variable(p)
    x1=np.matmul(A,x0)+B*u[0]
    x2=np.matmul(A,x1)+B*u[1]
    x3=np.matmul(A,x2)+B*u[2]
    x4=np.matmul(A,x3)+B*u[3]
    x5=np.matmul(A,x4)+B*u[4]
    Jterminal=cp.quad_form(x5,Qf)#np.matmul(x5,np.matmul(Qf,x5))#
    #Jx=np.matmul(x4,np.matmul(Q,x4))+np.matmul(x3,np.matmul(Q,x3))+np.matmul(x2,np.matmul(Q,x2))+np.matmul(x1,np.matmul(Q,x1))
    Jx=cp.quad_form(x4,Q)+cp.quad_form(x3,Q)+cp.quad_form(x2,Q)+cp.quad_form(x1,Q)
    Ju=cp.quad_form(u, Rk)
    obj = cp.Minimize(Jterminal+Jx+Ju)#(x**2 + y**2+z**2)#
    #y = xx[1]#cp.Variable(nonneg=True)
    #y = cp.Variable(nonneg=True)
    #z = cp.Variable()
    p1m1=np.array([[1],[-1]])#p1m1 for plus 1 minus 1
    pmwbr=pmw.b.reshape((pmw.b.shape[0],))
    pmawbr=pmaw.b.reshape((pmaw.b.shape[0],))
    pma2wbr=pma2w.b.reshape((pma2w.b.shape[0],))
    pma3wbr=pma3w.b.reshape((pma3w.b.shape[0],))
    pma4wbr=pma4w.b.reshape((pma4w.b.shape[0],))
    constraints=[pmw.A@x1<=pmwbr,pmaw.A@x2<=pmawbr,pma2w.A@x3<=pma2wbr,pma3w.A@x4<=pma3wbr,pma4w.A@x5<=pma4wbr]
    #constraints=[x1[3]>=-pi,x1[3]<=pi,x2[3]>=-pi,x2[3]<=pi,x3[3]>=-pi,x3[3]<=pi,x4[3]>=-pi,x4[3]<=pi,x5[3]>=-pi,x5[3]<=pi]#,F5@x5<=b5]
    #p1m1@x4[3]<=b43,F5@x5<=b5]
    prob = cp.Problem(obj, constraints)
    prob.solve()#(qcp=True)  # Returns the optimal value.
    print("status:", prob.status)
    print("optimal value", prob.value)
    print("optimal var", u.value)#, y.value)
    uv=u.value
    ntheta=np.random.normal(scale=1e-3)#ntheta for noise theta
    nthetadot=np.random.normal(scale=0.4)##nthetadot for noise theta dot
    w=np.array([ntheta,nthetadot])#the noise term w!
    x1v=np.matmul(A,x0)+B*uv[0] + w
    x2v=np.matmul(A,x1v)+B*uv[1]
    x3v=np.matmul(A,x2v)+B*uv[2]
    x4v=np.matmul(A,x3v)+B*uv[3]
    x5v=np.matmul(A,x4v)+B*uv[4]
    x0=x1v
    xl.append(x0)
    fl.append(prob.value)
    ul.append(uv[0])
    y, _ = decode(params, state, None, x1v)
    fig = plt.figure()
    plt.imshow(y.dist.mean)
    print('figure' + str(t) + ' saved')
    plt.savefig('MPC/control_step' + str(t) + '.png')
###############################################
###############################################
###############################################
###############################################
###############################################
print(xl[-1])
xl = np.array(xl)
m = np.zeros([N-10])
std = np.zeros([N-10])
for i in range(N-10):
    m[i] = xl[i:i+10,0].mean()
    std[i] = np.std(xl[i:i+10, 0]) 

iteration = np.arange(len(xl) - 10) + 1
plt.plot(iteration, m, color='r')
plt.fill_between(iteration, m - std, m + std, alpha=0.5, color='r')
plt.xlabel('# of Iterations')
plt.ylabel('Theta')
plt.show()
plt.savefig('MPC/std.png')