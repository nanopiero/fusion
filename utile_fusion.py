#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Le 03/06/2024
@author: lepetit
# fonctions utiles pour la génération
# de données à fusionner
"""
from random import randint
import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import copy 

##############################
########## with JIT ##########
##############################
from numba import jit
from numpy.random import randint


@jit(nopython=True)
def simu_rec(image, L,l,  fields=0):
    channels,size,size2=image.size()
    rec= torch.zeros(channels,size,size2)
    #out = 0*(image.clone())
    vertical=np.random.binomial(1,0.5)==1
    if vertical:
        width=l
        height=L
    else:
        width=L
        height=l

    top=randint(0, size-height)
    left=randint(0, size-width)
    rec[fields,top:top+height,left:left+width]=  np.random.uniform(0,0.4)   #0.1
    image=image + rec
    return image

def simu_noisy_rec(image, L,l,  fields=0):
    channels,size,size2=image.size()
    rec= torch.zeros(channels,size,size2)
    #out = 0*(image.clone())
    vertical=np.random.binomial(1,0.5)==1
    if vertical:
        width=l
        height=L
    else:
        width=L
        height=l

    top=randint(0, size-height)
    left=randint(0, size-width)
    rec[fields,top:top+height,left:left+width]= np.random.uniform(0,0.4)  #0.1
    noise=torch.randn(channels,size,size2)
    noise=noise*(noise>0).float()
    rec=noise*rec
    image=image + rec
    return image

@jit(nopython=True)
def get_equation(coords_1, coords_2):
  # rustine pour éviter les pbs. lon lat étant données à 10-5 près:
  eps = 10**-5
  if coords_2[0] == coords_1[0] :
      m = (coords_2[1] - coords_1[1]) / (eps + coords_2[0] - coords_1[0])
  elif coords_2[1] == coords_1[1] :
      m = (eps + coords_2[1] - coords_1[1]) / (coords_2[0] - coords_1[0])
  else:
      m = (coords_2[1] - coords_1[1]) / (coords_2[0] - coords_1[0])
  a =  -m
  b =  1
  c =  m*coords_1[0]  - coords_1[1]
  return a,b,c

@jit(nopython=True)
def get_dists(coords_1, coords_2, matrix):

  """
  Compute intersection length between
  [coords_1, coords_2] and pixel_ij
  should have: coords_1[0] < coords_2[0]
  """
  a, b, c = get_equation(coords_1, coords_2)
  # print("equation :", a, b ,c)
  N,M = matrix.shape
  # Indx = np.arange(0,M).reshape(1,M).repeat(N,axis = 0)
  # Indy = np.arange(N-1,-1,-1).reshape(N,1).repeat(M,axis = 1)

  Indx = np.arange(0,M).repeat(N).reshape(M,N).transpose((1,0))
  Indy = np.arange(N-1,-1,-1).repeat(M).reshape(N,M)

  # print(Indx == np.arange(0,M).repeat(N).reshape(M,N).transpose((1,0)))
  # print(Indy == np.arange(N-1,-1,-1).repeat(M).reshape(N,M))
  # raise Exception('')

  inds_1 = (np.ceil(coords_1[0]), np.ceil(coords_1[1]))
  inds_2 = (np.ceil(coords_2[0]), np.ceil(coords_2[1]))

  coord_l = min(coords_1[0], coords_2[0])
  coord_r = max(coords_1[0], coords_2[0])
  coord_d = min(coords_1[1], coords_2[1])
  coord_u = max(coords_1[1], coords_2[1])

  eps = 10**-6
  ind_l = int(np.floor(coord_l))
  ind_r = int(np.floor(coord_r - eps))
  ind_d = int(np.floor(coord_d))
  ind_u = int(np.floor(coord_u - eps))


  # for each pixel i,j: the sign of Mld[i,j]
  # gives the relative position of the left/down corner
  # wrt the line ax + by + c = 0
  Mld = (a * Indx + b * Indy + c)
  Mrd = (a * (Indx + 1) + b * Indy + c)
  Mlu = (a * Indx + b * (Indy + 1) + c)
  Mru = (a * (Indx + 1) + b * (Indy + 1) + c)

  # intersections with l/r edges :
  Xl = (Indx + 0.) #.astype(float)
  Xr = (Indx + 1.) #.astype(float)
  Yl = (- a * Indx - c) / b
  Yr = (- a * (Indx + 1) - c) / b


  # intersections with u/d edges :
  Xd = (- b * Indy - c) / a
  Xu = (- b * (Indy + 1) - c) / a
  Yd = (Indy + 0.) #.astype(float)
  Yu = (Indy + 1.) #.astype(float)

  # case of pixels 1 & 2
  if a >= 0:
    Xl[N - 1 - ind_u,ind_l] = coord_l
    Xu[N - 1 - ind_u,ind_l] = coord_l

    Xr[N - 1 - ind_d,ind_r] = coord_r
    Xd[N - 1 - ind_d,ind_r] = coord_r

    Yu[N - 1 - ind_u,ind_l] = coord_u
    Yl[N - 1 - ind_u,ind_l] = coord_u

    Yd[N - 1 - ind_d,ind_r] = coord_d
    Yr[N - 1 - ind_d,ind_r] = coord_d


  else:
    Xl[N - 1 - ind_d,ind_l] = coord_l
    Xd[N - 1 - ind_d,ind_l] = coord_l

    Xr[N - 1 - ind_u,ind_r] = coord_r
    Xu[N - 1 - ind_u,ind_r] = coord_r

    Yu[N - 1 - ind_u,ind_r] = coord_u
    Yr[N - 1 - ind_u,ind_r] = coord_u

    Yd[N - 1 - ind_d,ind_l] = coord_d
    Yl[N - 1 - ind_d,ind_l] = coord_d


  # Building distance matrix
  Dists = 0. * Xl
  # lu : path between lu, ld & lu, ru
  Mask = ((Mlu * Mld < 0) * (Mlu * Mru <= 0))
  Dists += np.sqrt((Xl - Xu)**2 + (Yl - Yu)**2) * Mask

  # lr : path between lu, ld & ru, rd
  Mask = (Mlu * Mld < 0) * (Mru * Mrd < 0)
  Dists += np.sqrt((Xl - Xr)**2 + (Yl - Yr)**2) * Mask

  # ld : path between lu, ld & ld, rd
  Mask = (Mlu * Mld < 0) * (Mld * Mrd <= 0)
  Dists += np.sqrt((Xl - Xd)**2 + (Yl - Yd)**2) * Mask

  # ur : path between lu, ru & ru, rd
  Mask = (Mlu * Mru <= 0) * (Mru * Mrd < 0)
  Dists += np.sqrt((Xu - Xr)**2 + (Yu - Yr)**2) * Mask

  # ud : path between lu, ru & ld, rd
  Mask = (Mlu * Mru <= 0) * (Mld * Mrd <= 0)
  Dists += np.sqrt((Xu - Xd)**2 + (Yu - Yd)**2) * Mask

  # rd : path between ld, rd & ru, rd
  Mask = (Mld * Mrd <= 0) * (Mru * Mrd < 0)
  Dists += np.sqrt((Xr - Xd)**2 + (Yr - Yd)**2) * Mask

  # clean outside the segment :
  Mask = (Indx >= ind_l) * (Indx <= ind_r)
  Dists *= Mask

  Mask = (Indy >= ind_d) * (Indy <= ind_u)
  Dists *= Mask

  return Dists


@jit(nopython=True)
def pseudo_meshgrid(size):
  b = np.arange(0,size).repeat(size).reshape(1,size,size)
  a = np.transpose(b, (0,2,1))
  return  a, b


@jit(nopython=True)
def simu_moving_disc(image, a, b):
  nsteps, size, _ = image.shape

  # Initialize centers and radii arrays
  centers = np.zeros((nsteps, 2))
  radii = np.zeros(nsteps)

  # Choose k
  k = np.random.randint(0,size)

  # Generate the kth center and radius
  radius_k = np.abs(np.random.normal(10, 8))
  center_k = radius_k + (size - radius_k) * np.random.random(2)

  # Generate advection speed and radius increment
  advection_speed = np.random.normal(0, 3, 2)
  radius_increment = np.random.normal(0, 8/nsteps)

  # Fill centers and radii arrays
  abs_centers = center_k[0] + (np.arange(nsteps) - k) * advection_speed[0]
  ord_centers = center_k[1] + (np.arange(nsteps) - k) * advection_speed[1]
  radii = radius_k  +  (np.arange(nsteps) - k) * radius_increment
  radii[radii <= 0] = 0

  distances_to_centers = (a - abs_centers.reshape((nsteps, 1, 1)))**2 + \
                         (b - ord_centers.reshape((nsteps, 1, 1)))**2

  discs =  1. * (distances_to_centers < radii.reshape(nsteps, 1, 1)**2)
  # discs =  (0.39 - 0.36*distance_to_centers/radii**2)*(distances_to_centers < radii**2)

  # apply a random intensity
  discs *= np.random.uniform(0.1,1.)
  image = image + discs
  return image

@jit(nopython=True)
def resize_channel(channel, new_size):
    x = np.linspace(0, 1, channel.shape[0])
    y = np.linspace(0, 1, channel.shape[1])
    x_new = np.linspace(0, 1, new_size)
    y_new = np.linspace(0, 1, new_size)
    return np.interp(x_new[:, None] + y_new[None, :], x, np.interp(y_new, y, channel))

@jit(nopython=True)
def spatialized_gt(ndiscs=5, size=64, nsteps=60):
  image=np.zeros((nsteps, 64, 64))
  a, b = pseudo_meshgrid(size)
  for i in range(ndiscs):
    image = simu_moving_disc(image, a, b)

  return image




@jit(nopython=True)
def create_cmls_filter(S, npairs = 10):
  # step 1 : on crée un filtre (batch pas encore formé)
  # associé à npairs cmls distribués au hasard dans l'image

  # init filter
  filter = np.zeros((npairs, S, S))

  distx = np.random.randint(0, 32, (npairs,))
  disty = np.random.randint(-15, 16, (npairs,))
  pairs = np.zeros((4,npairs))

  for i in range(npairs):

    coli0 = np.random.randint(0, S - distx[i], (1,)).item()
    rowi0 = np.random.randint(max(0, 0 - disty[i]), min(S, S - disty[i]), (1,)).item()
    coli1 = coli0 + distx[i]
    rowi1 = rowi0 + disty[i]
    xi0_local = np.random.rand(1).item()
    xi1_local = (distx[i] + np.random.rand(1)).item()

    xi0_global = coli0 + xi0_local
    xi1_global = coli0 + xi1_local

    if rowi1 > rowi0:
        rowi_max = rowi1
        rowi_min = rowi0
        yi0_local = rowi1 - rowi0 + np.random.rand(1).item()
        yi1_local =  np.random.rand(1).item()
        yi0_global = (S - 1 - rowi1) + yi0_local
        yi1_global = (S - 1 - rowi1) + yi1_local

    else :
        rowi_max = rowi0
        rowi_min = rowi1
        yi1_local = rowi0 - rowi1 + np.random.rand(1).item()
        yi0_local =  np.random.rand(1).item()
        yi0_global = (S - 1 - rowi0) + yi0_local
        yi1_global = (S - 1 - rowi0) + yi1_local

    # fill the filters (would be much better with sparse representation)
    cropi = np.zeros((rowi_max+1 - rowi_min, coli1+1 - coli0))
    filter[i, rowi_min:rowi_max+1, coli0:coli1+1] = \
                get_dists((xi0_local, yi0_local), (xi1_local, yi1_local), cropi)
    pairs[:,i] = [xi0_global/S, yi0_global/S, xi1_global/S, yi1_global/S]

  return pairs, filter


##############################
########## on GPU   ##########
##############################
def point_gt(images, npoints=10):
  bs, nsteps, S, _ = images.shape
  flat_images = images.view(bs, nsteps, S * S)
  # Randomly sample M indices for each image in the batch
  indices = torch.randint(0, S * S, (bs, npoints), device=images.device)

  # Gather the values from these indices for all images
  sampled_values = torch.gather(flat_images, 2, indices.unsqueeze(dim=1).repeat([1,nsteps,1]))

  # Calculate coordinates from indices
  rows = indices // S
  cols = indices % S

  # Normalize coordinates to be between 0 and 1
  ys = (1 - rows.float()/S) - 1/(2*S)
  xs = cols.float()/S + 1/(2*S)
  # print(normalized_rows.shape)
  # print(sampled_values.shape)
  # Stack the normalized coordinates with the values
  result = torch.cat((xs.unsqueeze(1),
                      ys.unsqueeze(1),
                      sampled_values), dim=1)

  return result


def segment_gt(images, pairs, filters):
  bs, nsteps, S, _ = images.shape
  nanfilters = copy.deepcopy(filters)
  nanfilters[nanfilters == 0] = torch.nan
  filtered_images = images.unsqueeze(dim=2) * nanfilters.unsqueeze(dim=1)
  sampled_values = torch.nanmean(filtered_images, dim=(3,4))
  result = torch.cat((pairs, sampled_values), dim=1)
  return result

def make_noisy_images(images):
    nbatch, nchannels, S, _ = images.shape

    # Step 1: Extract channels n°5 to 60, with a step of 5 (12 channels)
    extracted_images = images[:, torch.arange(4, 60, 5), :, :]  # Selects the 5th, 10th, ..., 60th channels

    # Step 2: Build a 25 x 25 Gaussian kernel with random std in [0,5]
    # and center taken in a random place around the square center (5 pixels max)
    kernel_size = 25
    central_square_size = 7
    std = torch.rand(nbatch, device=images.device) * 2.5

    center_x = (kernel_size - central_square_size) // 2 + torch.randint(0, central_square_size, (nbatch,), device=images.device)
    center_y = (kernel_size - central_square_size) // 2 + torch.randint(0, central_square_size, (nbatch,), device=images.device)

    x = torch.arange(kernel_size, dtype=torch.float32, device=images.device).repeat(nbatch, kernel_size, 1)
    y = x.transpose(1, 2)

    center_x = center_x.view(-1, 1, 1)
    center_y = center_y.view(-1, 1, 1)
    std = std.view(-1, 1, 1)

    gaussian_kernel = torch.exp(-((x - center_x) ** 2 + (y - center_y) ** 2) / (2 * std ** 2))
    gaussian_kernel[gaussian_kernel < 0.1] = 0
    gaussian_kernel  /= gaussian_kernel.sum(dim=(1, 2), keepdim=True)
    gaussian_kernel  = gaussian_kernel.unsqueeze(0)
    print('shape of gk:', gaussian_kernel.shape)
    # gaussian_kernel = gaussian_kernel.view(nbatch, 1, kernel_size, kernel_size)

    # Step 3: Apply the Gaussian kernel to all channels using conv2d
    transposed_images = extracted_images.permute(1, 0, 2, 3)  # (12, nbatch, S, S)

    noisy_images = F.conv2d(transposed_images, gaussian_kernel, padding='same') #, groups=nbatch)
    # print(noisy_images.shape)
    # Step 4: Binarize the output
    threshold =  0.4 * torch.rand(1) #torch.finfo(torch.float32).eps  # Tiny threshold
    print(threshold)
    binarized_images = (noisy_images > threshold).float()

    # Step 5: Re-transpose the dimensions back to (nbatch, nchannels, S, S)
    final_images = binarized_images.permute(1, 0, 2, 3)  # (nbatch, 12, S, S)

    return final_images




##############################
########## VIZ ###############
##############################

def set_tensor_values2(X, extracted_data):
    """
    Set values in tensor X using coordinates and values extracted from another tensor.

    Args:
    X (torch.Tensor): Target tensor where values need to be set, shape (N, nc, S, S).
    extracted_data (torch.Tensor): Data containing normalized coordinates and values, shape (N, 2+M, npoints).
    S (int): Size of the spatial dimension of X.

    Returns:
    torch.Tensor: Updated tensor X with new values set at specified coordinates.
    """
    N, nc, S, _ = X.shape
    N, Mp2, npoints = extracted_data.shape
    M = Mp2 - 2

    # Extract normalized coordinates and values
    xs = extracted_data[:, 0, :]
    ys = extracted_data[:, 1, :]
    values = extracted_data[:, 2:, :]

    # Convert normalized coordinates back to original scale
    # ys = (1 - rows.float()/S) - 1/(2*S)
    # xs = cols.float()/S + 1/(2*S)

    rows = ((1 - (ys + 1/(2*S))) * S).long()
    cols = ((xs - 1/(2*S)) * S).long()

    # Use the coordinates to set the values in X
    for i in range(N):
        for j in range(npoints):
            X[i, :, rows[i, j], cols[i, j]] = values[i, :, j] + (values[i, :, j] > -1)

    return X


import matplotlib.gridspec as gridspec
def plot_images(images, noisy_images, point_measurements, segment_measurements):
    # Set up the figure with GridSpec
    fig = plt.figure(figsize=(18, 24))
    gs = gridspec.GridSpec(12, 7, width_ratios=[1, 1, 1, 1, 1, 1, 2])  # Last column twice as wide

    # Manually create axes array for uniform handling as before
    axs = [fig.add_subplot(gs[i, j]) for i in range(12) for j in range(7)]

    # Hide all primary spines and ticks
    for ax in axs:
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(axis='both', which='both', left=False, right=False, top=False, bottom=False, labelleft=False, labelbottom=False)

    # Image and noisy image plots
    for i in range(12):
        image_indices = [5*i, 5*i+1, 5*i+2, 5*i+3, 5*i+4, 5*i]
        noisy_index = i
        for j in range(6):
            ax = axs[i*7 + j]
            img = noisy_images[noisy_index] if j == 5 else images[image_indices[j]]
            img_normalized = (img - np.min(img)) / (np.max(img) - np.min(img) + 0.000001)
            ax.imshow(img_normalized, cmap='gray', aspect='auto')
            ax.axis('off')

    # Point and Segment measurements plots
    for row in range(12):
        ax_main = axs[row * 7 + 6]  # Last column in each row
        if row < 2:  # First two rows for point measurements
            for idx in range(3) if row == 0 else range(2):
                ax = ax_main.inset_axes([0, 1 - (idx+1)/3, 1, 1/3])
                ax.plot(point_measurements[2:, idx + row*3], marker='.', markevery=(4, 5), markeredgewidth=2, markeredgecolor='black')
                label = f"Pluvio {idx+1 + row*3}"
                ax.set_ylim([-0.1, 1.5])
                coord1 = f"x={point_measurements[0, idx + row*3]:.2f}"  # First coordinate on a new line
                coord2 = f"y={point_measurements[1, idx + row*3]:.2f}"  # Second coordinate on another new line
                full_label = f"{label}\n{coord1}\n{coord2}"  # Combine into one string with two newlines
                ax.set_ylabel(full_label, rotation=0, labelpad=0, fontsize=6)
                ax.yaxis.set_label_coords(0.05, 0.4)
                ax.tick_params(axis='both', which='both', left=False, bottom=False, labelleft=False, labelbottom=False)
                for spine in ax.spines.values():
                    spine.set_visible(False)

        elif 2 <= row < 6:  # Next four rows for segment measurements
            for idx in range(3):
                actual_idx = 3 * (row - 2) + idx
                if actual_idx < 10:  # Ensure we don't exceed the 10 graphs
                    ax = ax_main.inset_axes([0, 1 - (idx+1)/3, 1, 1/3])
                    ax.plot(segment_measurements[4:, actual_idx], marker='.', markevery=(4, 5), markeredgewidth=1, markeredgecolor='black')
                    ax.set_ylim([-0.1, 1.5])
                    label = f"CML {actual_idx+1}"
                    coord_text = f"x1={segment_measurements[0, actual_idx]:.2f}, y1={segment_measurements[1, actual_idx]:.2f}\nx2={segment_measurements[2, actual_idx]:.2f}, y2={segment_measurements[3, actual_idx]:.2f}"
                    full_label = f"{label}\n{coord_text}"
                    ax.set_ylabel(full_label, rotation=0, labelpad=0, fontsize=6)
                    ax.yaxis.set_label_coords(0.05, 0.4)
                    ax.tick_params(axis='both', which='both', left=False, bottom=False, labelleft=False, labelbottom=False)
                    for spine in ax.spines.values():
                        spine.set_visible(False)
    # plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.1, wspace=0.05)  # Adjust overall spacing

    plt.tight_layout()
    plt.show()