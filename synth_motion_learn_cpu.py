from __future__ import print_function, division
import numpy as np, matplotlib.pyplot as plt, os, sys, gzip, pickle, time
from theano import tensor as T, shared, function, config
#from theano.sandbox.cuda.dnn import dnn_conv, dnn_pool
from theano.tensor.nnet.corr import CorrMM
from theano.tensor.signal.downsample import max_pool_2d
#from theano.sandbox.rng_mrg import MRG_RandomStreams; rng = MRG_RandomStreams(123456);
floatX = config.floatX;  flt = eval('np.'+floatX);  np.set_printoptions(precision=4) #, linewidth=150)

mom=0.9;  wdecay=0.001;  lr0=0.005;  nepoch=50;  batsz=128

# [# filter, filter_size, padding, stride]
nconv = [[32,9,0,2],[64,5,0,2],[128,3,1,1],[128,3,1,2],[128,3,1,1]]
#Map sz: 80->72-36-> 32-16    ->  16->       16-8     ->8          ->1

with np.load('moving_circles_dataset_spacing15_10k.npz') as data: 
  Yhost = data['params'].astype(floatX)
  Xhost = data['videos'].astype(floatX) # ~2GB
Xhost *= -1./255;  Xhost += 1
nsample,nframe,height,width = Xhost.shape
X = shared(Xhost)
Y = shared(Yhost)

if len(sys.argv)==1:
  winit = lambda n,d,h,w: (np.random.randn(n,d,h,w)/np.sqrt(d*h*w)).astype(floatX)
  binit = lambda n: np.zeros(n, floatX)
  net = {'W': [winit(nh[0],nv[0],nh[1],nh[1]) for nv,nh in zip([[nframe]]+nconv[:-1],nconv)],
         'b': [binit(nh[0]) for nh in nconv],
         'Wout': np.random.randn(8*8*nconv[-1][0],Yhost.shape[1]) / np.sqrt(8*8*nconv[-1][0]),
         'bout': binit(Yhost.shape[1]),
         'nconv': nconv, 'mom':mom, 'wdecay':wdecay, 'lr':lr0,
         'loss': np.zeros((nepoch,2,Yhost.shape[1]),floatX)
        }
  start_epoch = 0
else:
  with open(sys.argv[1],'rb') as f: net = pickle.load(f)
  nepoch = int(sys.argv[2])
  start_epoch = net['loss'].shape[0]
  if start_epoch < nepoch:
    net['loss'] = np.concatenate((net['loss'],np.zeros((nepoch-start_epoch,2,Yhost.shape[1]),floatX)))

params_host = net['W'] + net['b'] + [net['Wout'], net['bout']]
print('# params:', sum([np.prod(p.shape) for p in params_host]))
W = [shared(w) for w in net['W']];  Wout = shared(net['Wout'])
b = [shared(b) for b in net['b']];  bout = shared(net['bout'])
params = W + b + [Wout,bout]
dparams= [ shared( np.zeros_like(p) ) for p in params_host ]

#conv = lambda x,w,n: dnn_conv(x, w, border_mode='valid' if n[2]==0 else (n[2],n[2]))
conv = lambda x,w,n: CorrMM('valid' if n[2]==0 else (n[2],n[2]))(x,w)
x0xx = lambda x: x.dimshuffle('x',0,'x','x')               #, subsample=(n[3],n[3]))

x = T.ftensor4();  y = T.fmatrix();  h = [];  lr = T.fscalar();  idx = T.lscalar()
for i,n in enumerate(nconv):
  h.append( T.nnet.relu( conv(h[-1] if i else x, W[i], n) + x0xx(b[i]) ) )
  #if n[3]>1: h.append( dnn_pool(h[-1],(n[3],n[3]),(n[3],n[3])) )
  if n[3]>1: h.append( max_pool_2d(h[-1],(n[3],n[3]), ignore_border=True) )
o = T.dot(h[-1].reshape((batsz, nconv[-1][0] * 8 * 8)), Wout) + bout
L = T.sqr(y-o).sum(0) #.5 * T.sqr(y-o).sum()
gparams = T.grad(.5*L.sum(), params)
updates = []
for p,g,d in zip(params,gparams,dparams):
  dnew = mom * d - lr * (g + wdecay * p)
  updates += [[p,p+dnew],[d,dnew]]

f_train = function([idx,lr], L, givens={x: X[idx:idx+batsz], y: Y[idx:idx+batsz]}, updates=updates)
f_valid = function([idx],    L, givens={x: X[idx:idx+batsz], y: Y[idx:idx+batsz]})
f_prop  = function([idx],    o, givens={x: X[idx:idx+batsz]})

print('[Train: x,y,dx,dy,speed], [Valid: x,y,dx,dy,speed]')
lr=.1*lr0/batsz;  ntrain = nsample - (nsample>>3);  dt=time.time()
for epoch in range(start_epoch,nepoch):
  if epoch==1: lr*=10.
  ltrain,lvalid = np.zeros((2,Yhost.shape[1]), floatX);  #lr*=0.99; 
  for batch in range(0,ntrain,batsz):
    ltrain += f_train(batch,lr)
  if np.any(np.isnan(ltrain)): print('Loss is NaN at epoch', epoch); sys.exit(1)
  ltrain = np.sqrt( ltrain / ntrain )
  for batch in range(ntrain,nsample,batsz):
    lvalid += f_valid(batch)
  lvalid = np.sqrt( lvalid / (nsample - ntrain) )
  net['loss'][epoch] = ltrain,lvalid
  print(ltrain,lvalid)

if start_epoch < nepoch:
  dt=time.time()-dt;  print('Training time: %.1fs' % dt)
  for phost,pdev in zip(params_host, params): phost[:] = pdev.get_value()
  with open('synth_motion_learn_net.pkl','wb') as f: pickle.dump(net, f, pickle.HIGHEST_PROTOCOL)

sample = [Xhost[ntrain:ntrain+batsz], Yhost[ntrain:ntrain+batsz], f_prop(ntrain)]
with open('synth_motion_learn_sample.pkl','wb') as f: pickle.dump(sample, f, pickle.HIGHEST_PROTOCOL)
#plt.plot(net['loss'].reshape(nepoch,10))
plt.gca().set_color_cycle('rgbycrgbyc')
plt.plot(net['loss'][:,0]);  plt.plot(net['loss'][:,1], '--');
plt.legend('t_x t_y t_dx t_dy t_v v_x v_y v_dx v_dy v_v'.split(), ncol=2)
plt.savefig('synth_motion_learn_curve.pdf')
plt.figure(figsize=(8.5,11))
for i in range(5):
  plt.subplot(511+i);  plt.imshow(1. - sample[0][i].transpose(1,0,2).reshape(80,640))
  plt.axis('off');     plt.title(np.array((sample[1][i], sample[2][i])))
plt.savefig('synth_motion_sample_y.pdf')
