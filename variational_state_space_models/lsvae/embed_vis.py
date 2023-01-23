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
for i,x in enumerate(grid_samples()):
    fig = plt.figure()
    plt.imshow(x['images'])
    plt.savefig(str(i)+'_'+'|'.join(str(x['states']).split(' '))+'.jpg')

    dist, _ = encode(params, state, None, x)
    print(np.linalg.inv(dist.conc))
    samples = dist.multi_sample(next(rng), 10)
    for s in samples:
        if i == 1 or i == 93 or i == 64:
            b_x.append(s[0])
            b_y.append(s[1])
            b_cx.append(x['states'][0])
            b_cy.append(x['states'][1])
        xs.append(s[0])
        ys.append(s[1])
        color_x.append(x['states'][0])
        color_y.append(x['states'][1])
b_cx.append(max(color_x))
b_cy.append(max(color_y))
b_cx.append(min(color_x))
b_cy.append(min(color_y))

y,state = decode(params,state,None,np.array([0., 3]))

fig = plt.figure()
plt.imshow(y.dist.mean)
plt.savefig('test.png')

fig = plt.figure()
plt.scatter(xs, ys, c=bivariateColor(color_x, color_y),s=18)
plt.scatter(b_x,b_y,c=bivariateColor(b_cx,b_cy)[:-2],s=288)
plt.scatter(b_x,b_y,c=np.zeros_like(b_x),s=10)
plt.xlabel('x1')
plt.ylabel('x2')
plt.savefig('space.png')
