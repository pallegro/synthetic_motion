from pylab import *
import time, subprocess
from matplotlib import animation, rcParams, verbose

nsample=10<<10;  nframe=8;  spacing=.15;  figsz=1;

params = rand(nsample,5) * [.5,.5,2*pi,0.,.5] + [.25,.25,0,0,0];
params[:,3] = sin(params[:,2])
params[:,2] = cos(params[:,2])
I = zeros((nsample,nframe,(80*figsz)**2), 'uint8')

fig,ax = subplots(figsize=(figsz,figsz), frameon=False)
p, = ax.plot([],[], 'ko', markersize=4*figsz)
ax.set_xlim(0,1);  ax.set_ylim(0,1);  ax.set_axis_off();  fig.tight_layout(pad=0);
subplots_adjust(left=0, right=1, top=1, bottom=0)
fig.canvas.draw();  buf = fig.canvas.buffer_rgba()

for sample,(x,y,dx,dy,s) in enumerate(params):
  for frame in range(nframe):
    i = (frame/nframe-.5) * s + arange(-1,1,spacing)
    p.set_data(x+i*dx, y+i*dy)
    fig.canvas.draw();
    I[sample,frame] = np.frombuffer(buf, np.uint8)[::4]

I.shape = nsample,nframe,80*figsz,80*figsz
savez('moving_circles_dataset_spacing15_10k.npz', params=params, videos=I)
'''
proc = subprocess.Popen(['ffmpeg', '-f','rawvideo', '-vcodec','rawvideo', '-s','%dx%d' % (80*figsz,80*figsz),
                         '-pix_fmt','gray', '-r','8', '-i', 'pipe:','-vcodec',rcParams['animation.codec'],
                         '-metadata', 'title=Moving circles', '-y','moving_circles_dataset_spacing15.mp4'],
                        shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE)
out, err = proc.communicate(I.tobytes());  print(out,err);
'''