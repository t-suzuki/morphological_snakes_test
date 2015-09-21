#!env python
# Implementation of "A morphological approach to curvature-based evolution of curves and surfaces," TPAMI2013, Pablo Marquez-Neila, et al.
#
# License: Public Domain

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import skimage.io
import skimage.transform
import skimage.filter
import skimage.draw
import skimage.color

def neighbors(u, y, x):
    ys, xs = np.mgrid[y-1:y+2, x-1:x+2]
    ys[ys < 0] = 0
    ys[ys > u.shape[0] - 1] = u.shape[0] - 1
    xs[xs < 0] = 0
    xs[xs > u.shape[1] - 1] = u.shape[1] - 1
    return  u[ys, xs].ravel()

def base_B(u, y, x, p):
    m = neighbors(u, y, x)
    return np.any(m != p)

def base_B2(u, y, x, p):
    m = neighbors(u, y, x)
    if p == 0: m = 1 - m
    return max([m[0]*m[8], m[1]*m[7], m[2]*m[6], m[3]*m[5]]) == 0

def _morph_SI_or_IS(u, SI):
    res = np.array(u)
    for y in range(u.shape[0]):
        for x in range(u.shape[1]):
            if u[y, x] == SI and base_B2(u, y, x, SI):
                res[y, x] = 1 - SI
    return res

def morph_IS(u): return _morph_SI_or_IS(u, 0)
def morph_SI(u): return _morph_SI_or_IS(u, 1)

def _morph_erode_or_dilate(u, erode):
    res = np.array(u)
    for y in range(u.shape[0]):
        for x in range(u.shape[1]):
            if u[y, x] == erode and base_B(u, y, x, erode):
                res[y, x] = 1 - erode
    return res

def morph_erode(u): return _morph_erode_or_dilate(u, 1)
def morph_dilate(u): return _morph_erode_or_dilate(u, 0)

def gradient(u):
    yp = np.arange(u.shape[0])
    ym = np.arange(u.shape[0])
    yp[:-1] = yp[1:]
    ym[1:] = ym[:-1]
    xp = np.arange(u.shape[1])
    xm = np.arange(u.shape[1])
    xp[:-1] = xp[1:]
    xm[1:] = xm[:-1]
    return np.dstack([u[yp, :] - u[ym, :], u[:, xp] - u[:, xm]]) * 0.5

def MorphologicalGAC(g_map, u_initial, nu, theta=0.8, mu=1, n_iter=5, plot_iteration=False):
    '''Morphological version of Geodesic Active Contour Eq. (29)'''

    # morphological version of Eq. (27)
    def step_baloon(u_in):
        u_out = (morph_erode if nu < 0 else morph_dilate)(u_in)
        mask = g_map < theta
        u_out[mask] = u_in[mask]
        return u_out

    # data term Eq. (29)
    dg = gradient(g_map)
    def step_data(u_in):
        du = gradient(u_in)
        dot = du[:, :, 0]*dg[:, :, 0] + du[:, :, 1]*dg[:, :, 1]
        u_out = np.zeros_like(u_in)
        u_out[dot > 0] = 1
        u_out[dot == 0] = u_in[dot == 0]
        return u_out

    # morphological version of Eq. (28)
    op = [morph_SI, morph_IS]
    def step_smooth(u_in):
        u_out = u_in
        for j in range(mu):
            u_out = op[0](op[1](u_out))
            op[:] = op[::-1]
        return u_out

    if plot_iteration:
        fig, axs = plt.subplots(3, n_iter)

    u_res = np.array(u_initial)
    for i in range(n_iter):
        u_res = step_baloon(u_res)
        if plot_iteration: axs[0, i].imshow(u_res)
        u_res = step_data(u_res)
        if plot_iteration: axs[1, i].imshow(u_res)
        u_res = step_smooth(u_res)
        if plot_iteration: axs[2, i].imshow(u_res)

    return u_res


def MorphologicalACWE(I_map, u_initial, nu, lmd1=1.0, lmd2=1.0, mu=1, n_iter=5, plot_iteration=False):
    '''Morphological version of Active Contours Without Edges Eq. (32)'''

    def step_baloon(u_in):
        if nu == 0: return u_in
        return (morph_erode if nu < 0 else morph_dilate)(u_in)

    def step_data(u_in):
        du = gradient(u_in)
        mag = np.sqrt(du[:, :, 0]**2.0 + du[:, :, 1]**2.0)
        c1 = I_map[u_in == 1].mean()
        c2 = I_map[u_in == 0].mean()
        diff = lmd1*(I_map - c1)**2.0 - lmd2*(I_map - c2)**2.0
        u_out = np.zeros_like(u_in)
        u_out[diff < 0] = 1
        u_out[diff == 0] = u_in[diff == 0]
        return u_out

    op = [morph_SI, morph_IS]
    def step_smooth(u_in):
        u_out = u_in
        for j in range(mu):
            u_out = op[0](op[1](u_out))
            op[:] = op[::-1]
        return u_out

    if plot_iteration:
        fig, axs = plt.subplots(3, n_iter)

    u_res = np.array(u_initial)
    for i in range(n_iter):
        u_res = step_baloon(u_res)
        if plot_iteration: axs[0, i].imshow(u_res)
        u_res = step_data(u_res)
        if plot_iteration: axs[1, i].imshow(u_res)
        u_res = step_smooth(u_res)
        if plot_iteration: axs[2, i].imshow(u_res)

    return u_res

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default=None,
            help='input file')
    parser.add_argument('--sigma', default=1.0, type=float,
            help='Gaussian sigma')
    parser.add_argument('--alpha', default=10.0, type=float,
            help='Edge detector alpha')
    parser.add_argument('--nu', default=1, type=int,
            help='nu<0: erode base, nu>0: dilate base')
    parser.add_argument('--iter', default=15, type=int,
            help='number of iteration')
    parser.add_argument('--lmd1', default=1.0, type=float,
            help='lambda 1(inside) parameter')
    parser.add_argument('--lmd2', default=1.0, type=float,
            help='lambda 2(outside) parameter')
    parser.add_argument('--method', default='MACWE', type=str,
            help='MGAC or MACWE')
    args = parser.parse_args()

    if args.input is not None:
        img = skimage.io.imread(args.input)
        div = 4
    else:
        img = skimage.data.lena()
        div = 4

    # input image.
    if len(img.shape) == 3:
        img = skimage.color.rgb2gray(img)
    img = skimage.transform.resize(img, (img.shape[0]/div, img.shape[1]/div))
    img = (img - img.min())/ img.ptp()

    # prepare edge detector g(x)
    gaussian_img = skimage.filter.gaussian_filter(img, args.sigma)
    gx = skimage.filter.hprewitt(gaussian_img)
    gy = skimage.filter.vprewitt(gaussian_img)
    g_map = 1.0/np.sqrt(1.0 + args.alpha*np.sqrt(gx**2.0 + gy**2.0))
    g_map = (g_map - g_map.min())/g_map.ptp()

    initial_u = np.zeros_like(img, dtype=np.int32)
    if args.nu <= 0:
        print 'erode-driven'
        initial_u[1:-1, 1:-1] = 1
    else:
        print 'dilate-driven'
        initial_u[img.shape[0]*5/9:img.shape[0]*6/9, img.shape[1]*5/9:img.shape[1]*6/9] = 1

    if args.method == 'MGAC':
        print 'Morphological GAC'
        final_u = MorphologicalGAC(g_map, initial_u, args.nu, n_iter=args.iter, plot_iteration=True)
    elif args.method == 'MACWE':
        print 'Morphological ACWE'
        final_u = MorphologicalACWE(img, initial_u, args.nu, lmd1=args.lmd1, lmd2=args.lmd2, n_iter=args.iter, plot_iteration=True)
    else:
        print 'unknown method', args.method
        sys.exit(-1)

    fig, axs = plt.subplots(2, 2)
    ax = axs[0][0]; ax.imshow(img, vmin=0, vmax=1, cmap='gray')
    ax = axs[0][1]; ax.imshow(g_map, vmin=0, vmax=1, cmap='hot');
    ax = axs[1][0]; ax.imshow(initial_u, cmap='gray')
    ax = axs[1][1]; ax.imshow(final_u, cmap='gray')
    plt.show()
