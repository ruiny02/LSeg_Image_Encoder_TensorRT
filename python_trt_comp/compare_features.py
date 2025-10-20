#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_features_all.py

1) PT vs TRT metrics table (all sizes)
2) Cross-Weight metrics table
3) PT vs TRT heatmap (size=480)
4) PT size×weight heatmaps (PT만)
"""
import os
import numpy as np
import argparse
import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA
import torch
import torch.nn.functional as F

def cosine_similarity(a, b):
    ta = torch.from_numpy(a).flatten().float().unsqueeze(0)
    tb = torch.from_numpy(b).flatten().float().unsqueeze(0)
    return float(F.cosine_similarity(ta, tb))

def l2_distance(a, b):
    return float(np.linalg.norm(a.flatten() - b.flatten()))

def reduce_feature_map(fm):
    # fm: (B×)C×H×W  -> 배치 차원이 있으면 제거
    if fm.ndim == 4:
        # 가정: (1, C, H, W)
        fm = fm.squeeze(0)
    # 이제 fm.ndim == 3 이 되어야 함
    C, H, W = fm.shape
    X = fm.reshape(C, -1).T  # (H*W)×C
    pca = PCA(n_components=1, svd_solver='randomized')
    comp = pca.fit_transform(X).squeeze().reshape(H, W)
    mn, mx = comp.min(), comp.max()
    return (comp-mn)/(mx-mn) if mx>mn else comp

def load_and_reduce(path):
    arr = np.load(path)
    return reduce_feature_map(arr)

def find_sizes(sample_dir):
    return sorted(int(os.path.splitext(f)[0])
                  for f in os.listdir(sample_dir) if f.endswith(".npy"))

def plot_pt_vs_trt_heatmap(outputs_dir, backbones, weights, images, size):
    """ size=480 한정, 각 (backbone,weight)별 한 행에 [img_py, img_trt, ...] """
    modes = ["pt", "trt"]
    entries = [(b,w) for b in backbones for w in weights[b]]
    nrows = len(entries)
    ncols = len(images) * 2

    fig = plt.figure(figsize=(ncols, nrows))
    gs = GridSpec(nrows, ncols, figure=fig, wspace=0.05, hspace=0.05)

    for r, (b,w) in enumerate(entries):
        for i, img in enumerate(images):
            for m, mode in enumerate(modes):
                c = i*2 + m
                ax = fig.add_subplot(gs[r, c])
                path = os.path.join(outputs_dir, mode, b, w, img, f"{size}.npy")
                if os.path.exists(path):
                    ax.imshow(load_and_reduce(path), cmap="viridis")
                ax.set_xticks([]); ax.set_yticks([])
                # 첫 행에만 상단에 이미지+mode 표시
                if r == 0:
                    ax.set_title(f"{img} ({mode.upper()})", pad=2, fontsize=8)
                # 첫 열에만 y축에 backbone/weight 표시
                if c == 0:
                    ax.set_ylabel(f"{b}/{w}", rotation=0, labelpad=20, fontsize=8)

    fig.suptitle(f"PT vs TRT (size={size})", y=0.92, fontsize=12)
    plt.tight_layout(rect=[0,0,1,0.90])
    plt.show()

def plot_pt_size_weight_heatmaps(outputs_dir, backbones, weights, images, sizes):
    """ PT만, 각 (backbone,weight)별 한 행에 [sz1, sz2, ...] """
    entries = [(b,w) for b in backbones for w in weights[b]]
    nrows = len(entries)
    # 한 행에 [cat(sz1,sz2...), cat2(sz1,sz2...), cat3(...)]
    ncols = len(sizes) * len(images)

    fig, axs = plt.subplots(nrows, ncols,
                             figsize=(ncols, nrows),
                             squeeze=False,
                             gridspec_kw={'wspace':0.05,'hspace':0.05})

    for r, (b,w) in enumerate(entries):
        for img_idx, img in enumerate(images):
            for size_idx, sz in enumerate(sizes):
                col = img_idx * len(sizes) + size_idx
                ax = axs[r, col]
                path = os.path.join(outputs_dir, "pt", b, w, img, f"{sz}.npy")
                if os.path.exists(path):
                    ax.imshow(load_and_reduce(path), cmap="viridis")
                ax.set_xticks([]); ax.set_yticks([])
                # 첫 행엔 이미지 이름+size
                if r == 0:
                    ax.set_title(f"{img} | sz={sz}", pad=2, fontsize=8)
                # 각 row의 첫 (img0,sz0) 에만 백본/웨이트 레이블
                if img_idx == 0 and size_idx == 0:
                    ax.set_ylabel(f"{b}/{w}", rotation=0, labelpad=20, fontsize=8)

    fig.suptitle("PT size×weight heatmaps", y=0.92, fontsize=12)
    plt.tight_layout(rect=[0,0,1,0.90])
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs_dir", default="outputs/feature_maps",
                        help="root of pt/ and trt/ folders")
    parser.add_argument("--visualize", action="store_true",
                        help="show heatmaps")
    args = parser.parse_args()

    pt_root  = os.path.join(args.outputs_dir, "pt")
    trt_root = os.path.join(args.outputs_dir, "trt")

    # 백본, 웨이트, 이미지, 사이즈 자동 탐색
    backbones = sorted(d for d in os.listdir(pt_root)
                       if os.path.isdir(os.path.join(pt_root, d)))
    weights = {b: sorted(os.listdir(os.path.join(pt_root, b)))
               for b in backbones}
    # 첫 번째 (backbone,weight) 디렉토리에서 이미지 폴더
    sample_imgs = os.listdir(os.path.join(pt_root, backbones[0], weights[backbones[0]][0]))
    images = sorted(sample_imgs)
    # 첫 이미지에서 사이즈 폴더
    size_dir = os.path.join(pt_root, backbones[0], weights[backbones[0]][0], images[0])
    sizes = find_sizes(size_dir)

    # 1) PT vs TRT metrics (모든 size)
    rec_ptvstrt = []
    for b in backbones:
        for w in weights[b]:
            for img in images:
                for sz in sizes:
                    p = os.path.join(pt_root, b, w, img, f"{sz}.npy")
                    t = os.path.join(trt_root, b, w, img, f"{sz}.npy")
                    if os.path.exists(p) and os.path.exists(t):
                        fpt = np.load(p)
                        ftrt = np.load(t)
                        cos = cosine_similarity(fpt, ftrt)
                        l2  = l2_distance(fpt, ftrt)
                        rec_ptvstrt.append({
                            "Backbone": b, "Weight": w,
                            "Image": img, "Size": sz,
                            "Cosine_PTvTRT": cos,
                            "L2_PTvTRT": l2
                        })
    df1 = pd.DataFrame(rec_ptvstrt)
    print("\n## ▶ PT vs TRT 비교")
    print(df1.to_markdown(index=False))

    # 2) Cross-Weight metrics (같은 백본 내 모든 페어, 모든 size)
    rec_cross = []
    for b in backbones:
        for w1, w2 in combinations(weights[b], 2):
            for img in images:
                for sz in sizes:
                    p1 = os.path.join(pt_root, b, w1, img, f"{sz}.npy")
                    p2 = os.path.join(pt_root, b, w2, img, f"{sz}.npy")
                    t1 = os.path.join(trt_root, b, w1, img, f"{sz}.npy")
                    t2 = os.path.join(trt_root, b, w2, img, f"{sz}.npy")
                    if all(os.path.exists(x) for x in (p1,p2,t1,t2)):
                        f1 = np.load(p1); f2 = np.load(p2)
                        c_pt = cosine_similarity(f1, f2)
                        l_pt = l2_distance(f1, f2)
                        f1t = np.load(t1); f2t = np.load(t2)
                        c_tr = cosine_similarity(f1t, f2t)
                        l_tr = l2_distance(f1t, f2t)
                        rec_cross.append({
                            "Backbone": b,
                            "Weight Pair": f"{w1} vs {w2}",
                            "Image": img, "Size": sz,
                            "Cosine_PT": c_pt, "L2_PT": l_pt,
                            "Cosine_TRT": c_tr, "L2_TRT": l_tr
                        })
    df2 = pd.DataFrame(rec_cross)
    print("\n## ▶ Cross-Weight 비교")
    print(df2.to_markdown(index=False))

    # 3) 시각화
    if args.visualize:
        FIXED = 480 if 480 in sizes else sizes[-1]
        plot_pt_vs_trt_heatmap(args.outputs_dir, backbones, weights, images, FIXED)
        plot_pt_size_weight_heatmaps(args.outputs_dir, backbones, weights, images, sizes)

if __name__ == "__main__":
    main()
