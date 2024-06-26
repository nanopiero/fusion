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
  """
  creates npairs of pairs of coordinates
  creates npairs images of size S x S
  where :
      - pixels out of the segment defined by 
  the pairs of coordinates are zero
      - pixels in that segment contain the distance
      of the intersection between the segment and the pixel
  """
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
########## Datasets ##########
##############################

from torch.utils.data import Dataset

class FusionDataset(Dataset):
    def __init__(self, length_dataset=6400, npairs=10, nsteps=60, ndiscs=5, size_image=64):
        """
        Args:
              I need a pytorch dataset that will simply embed two numpy function that generates random tensors. 
              These functions, called spatialized_gt and create_cmls_filter are @jit decorated.
        """
        
        self.length_dataset = length_dataset
        self.npairs = npairs
        self.nsteps = nsteps
        self.ndiscs = ndiscs
        self.size_image = size_image

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, idx):
        image = spatialized_gt(ndiscs=self.ndiscs, size=self.size_image, nsteps=self.nsteps)
        pairs, filter = create_cmls_filter(self.size_image, npairs = self.npairs)

        return image, pairs, filter


"""
%%timeit -r 2 -n 1
# comparaison avec dataset pytorch : 3 fois plus lent si num_workers=4
S = 64
for i in range(100):
  for i in range(64):
    # in the dataset :
    image = spatialized_gt(ndiscs=5, size=S, nsteps=60)
    pairs, filter = create_cmls_filter(S, npairs = 10)


%%timeit -r 2 -n 1
for i, data in enumerate(loader):
  pass
"""




##############################
########## on GPU   ##########
##############################

def segment_gt(images, pairs, filters, use_fcn=False, split=None):
  """
  split : list of tuples (min link id, max link id + 1)) 
  """
    
  bs, nsteps, S, _ = images.shape
  _, nlinks, _, _ = filters.shape

  if split is None :
      filtered_images = images.unsqueeze(dim=2) * \
                    filters.unsqueeze(dim=1)     
      sampled_values = torch.sum(filtered_images,\
                        dim=(3,4))
      denom = torch.sum(filters.unsqueeze(dim=1),\
                        dim=(3,4))
      sampled_values /= denom 
      segment_measurements = torch.cat((pairs, sampled_values), dim=1)
        
    
      if not use_fcn:
        return segment_measurements, None
      else:
        sampled_values += 0.1
        filters = filters.unsqueeze(1) > 0
        filters = filters * sampled_values.view(bs, nsteps, nlinks, 1, 1)
        filters = filters.sum(dim=2)
        
      return segment_measurements, filters

  else :
      list_segments = []
      for (l0, l1) in split:
          partial_filters = filters[:,l0:l1,...]
          
          filtered_images = images.unsqueeze(dim=2) * \
                            partial_filters.unsqueeze(dim=1)     
          sampled_values = torch.sum(filtered_images,\
                            dim=(3,4))
          denom = torch.sum(partial_filters.unsqueeze(dim=1),\
                            dim=(3,4))
          sampled_values /= denom 
          segment_measurements = torch.cat((pairs, sampled_values), dim=1)
            
        
          if not use_fcn:
            list_segments.append((segment_measurements, None))
          else:
            sampled_values += 0.1
            partial_filters = partial_filters.unsqueeze(1) > 0 
            partial_filters = partial_filters * sampled_values.view(bs, nsteps, l1 - l0, 1, 1)
            partial_filters = partial_filters.sum(dim=2)

            list_segments.append((segment_measurements, partial_filters))
              
      return list_segments
      


"""
def segment_gt(images, pairs, filters, use_fcn=False):
  bs, nsteps, S, _ = images.shape
  _, nlinks, _, _ = filters.shape

  filtered_images = images.unsqueeze(dim=2) * \
                    filters.unsqueeze(dim=1)

  sampled_values = torch.sum(filtered_images,\
                    dim=(3,4))
  denom = torch.sum(filters.unsqueeze(dim=1),\
                    dim=(3,4))
  sampled_values /= denom 
  segment_measurements = torch.cat((pairs, sampled_values), dim=1)
    

  if not use_fcn:
    return segment_measurements, None
  else:
    sampled_values += 0.1
    filters = filters.unsqueeze(1)
    filters = filters * sampled_values.view(bs, nsteps, nlinks, 1, 1)
    filters = filters.sum(dim=2)
  return segment_measurements, filters


def segment_gt(images, pairs, filters, use_fcn=False):
  bs, nsteps, S, _ = images.shape
  _, nlinks, _, _ = filters.shape

  filters[filters == 0] = torch.nan
  filtered_images = images.unsqueeze(dim=2) * \
                    filters.unsqueeze(dim=1)
  # on ajoute 0.1 pour distinguer du cas == 0
  sampled_values = torch.nanmean(filtered_images,\
                    dim=(3,4))
  segment_measurements = torch.cat((pairs, sampled_values), dim=1)
  filters[filters != filters] = 0

  if not use_fcn:
    return segment_measurements, None
  else:
    sampled_values += 0.1
    filters = filters.unsqueeze(1)
    filters = filters * sampled_values.view(bs, nsteps, nlinks, 1, 1)
    filters = filters.sum(dim=2)
  return segment_measurements, filters

"""

def generate_indices_rows_and_columns(images, npoints):
  bs, nsteps, S, _ = images.shape
  weights = torch.ones(S**2).expand(bs, -1).to(images.device)
  indices = torch.multinomial(weights, num_samples=npoints, replacement=False) #.to(images.device)

  # Calculate coordinates from indices
  rows = indices // S
  cols = indices % S

  # Gather the values from these indices for all images
  indices = indices.unsqueeze(dim=1).repeat([1,nsteps,1])
  return indices, rows, cols


def indices_to_sampled_values(images, indices):
  bs, nsteps, S, _ = images.shape
  flat_images = images.view(bs, nsteps, S * S)

  # Gather the values from these indices for all images
  sampled_values = torch.gather(flat_images, 2, indices)
  return sampled_values


def get_point_measurements(rows, cols, sampled_values, S=64):
  # Normalize coordinates to be between 0 and 1
  ys = (1 - rows.float()/S) - 1/(2*S)
  xs = cols.float()/S + 1/(2*S)

  # Stack the normalized coordinates with the values
  point_measurements = torch.cat((xs.unsqueeze(1),
                                  ys.unsqueeze(1),
                                  sampled_values), dim=1)
  return point_measurements


def point_gt(images, npoints=10, use_fcn=False, split=None): # nb_pluvios_ Split: (n0,n1,n2,..., nr). rq : n_points = Sum ni
  bs, nsteps, S, _ = images.shape

  indices, rows, cols = generate_indices_rows_and_columns(images, npoints)
  sampled_values = indices_to_sampled_values(images, indices)

  if split is None:
    point_measurements = get_point_measurements(rows, cols, sampled_values, S)

    if not use_fcn:
      return point_measurements, None, (indices, rows, cols)

    else:
      # Difference with point_gt:
      point_measurements_fcn = -0.1 * torch.ones(images.numel(), device=images.device)
      indices_batch = torch.arange(bs).repeat(60)
      # indice du premier élément de la i ème image pour le premier time step dans images.flatten()
      idx_i000=(torch.arange(bs, device = images.device) * nsteps).view(bs,1).expand(bs,nsteps)
      # indices du premier élément de la i ème image pour le premier time step j dans images.flatten()
      idx_ij00=idx_i000 + torch.arange(nsteps, device = images.device).view(1,nsteps).expand(bs,nsteps)
      # indices à conserver :
      idx_ijkl = S**2 * idx_ij00.unsqueeze(-1) + indices
      point_measurements_fcn[idx_ijkl.flatten()] = sampled_values.flatten()

      point_measurements_fcn = point_measurements_fcn.view(bs, nsteps, S, S)

      return point_measurements, point_measurements_fcn, (indices, rows, cols)
  # splitting
  else:
    pos = 0
    splitted_point_measurements = []
    for np in split:
      point_measurements = get_point_measurements(rows[:, pos:pos + np],
                                                  cols[:, pos:pos + np],
                                                  sampled_values[:, :, pos:pos + np],
                                                  S)

      splitted_point_measurements.append((point_measurements,
                                          None,
                                          (indices[:, :, pos:pos + np], rows[:,pos:pos + np], cols[:, pos:pos + np])))
      pos += np

    if not use_fcn :
        return splitted_point_measurements

    else :
      pos = 0
      splitted_point_measurements_fcn = []

      for i, np in enumerate(split):
        split_indices = indices[:, :, pos:pos + np]

        # Difference with point_gt:
        point_measurements_fcn = -0.1 * torch.ones(images.numel(), device=images.device)
        indices_batch = torch.arange(bs).repeat(60)
        # indice du premier élément de la i ème image pour le premier time step dans images.flatten()
        idx_i000=(torch.arange(bs, device = images.device) * nsteps).view(bs,1).expand(bs,nsteps)
        # indices du premier élément de la i ème image pour le premier time step j dans images.flatten()
        idx_ij00=idx_i000 + torch.arange(nsteps, device = images.device).view(1,nsteps).expand(bs,nsteps)
        # indices à conserver :
        idx_ijkl = S**2 * idx_ij00.unsqueeze(-1) + split_indices
        point_measurements_fcn[idx_ijkl.flatten()] = sampled_values[:, :, pos:pos + np].flatten()

        splitted_point_measurements_fcn.append((splitted_point_measurements[i][0], point_measurements_fcn.view(bs, nsteps, S, S),
                                                (split_indices, rows[:,pos:pos + np], cols[:, pos:pos + np])))
        pos += np

      return splitted_point_measurements_fcn


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

    x = torch.arange(kernel_size, dtype=torch.float32, \
                                  device=images.device)\
                                  .repeat(nbatch, kernel_size, 1)
    y = x.transpose(1, 2)

    center_x = center_x.view(-1, 1, 1)
    center_y = center_y.view(-1, 1, 1)
    std = std.view(-1, 1, 1)

    gaussian_kernel = torch.exp(-((x - center_x) ** 2 + (y - center_y) ** 2) / (2 * std ** 2))
    gaussian_kernel[gaussian_kernel < 0.1] = 0
    gaussian_kernel  /= gaussian_kernel.sum(dim=(1, 2), keepdim=True)
    gaussian_kernel  = gaussian_kernel.unsqueeze(1)

    # gaussian_kernel = gaussian_kernel.view(nbatch, 1, kernel_size, kernel_size)

    # Step 3: Apply the Gaussian kernel to all channels using conv2d
    transposed_images = extracted_images.permute(1, 0, 2, 3)  # (12, nbatch, S, S)
    noisy_images = F.conv2d(transposed_images, gaussian_kernel, padding='same', groups=nbatch)
    # print(noisy_images.shape)
    # Step 4: Binarize the output
    threshold =  0.4 * torch.rand(1,\
                       device=images.device) #torch.finfo(torch.float32).eps  # Tiny threshold
    binarized_images = (noisy_images > threshold).float()

    # Step 5: Re-transpose the dimensions back to (nbatch, nchannels, S, S)
    final_images = binarized_images.permute(1, 0, 2, 3)  # (nbatch, 12, S, S)

    return final_images




"""

def segment_gt(images, pairs, filters, use_fcn=False):
  bs, nsteps, S, _ = images.shape
  _, nlinks, _, _ = filters.shape

  filters[filters == 0] = torch.nan
  filtered_images = images.unsqueeze(dim=2) * \
                    filters.unsqueeze(dim=1)
  # on ajoute 0.1 pour distinguer du cas == 0
  sampled_values = torch.nanmean(filtered_images,\
                    dim=(3,4)) 
  segment_measurements = torch.cat((pairs, sampled_values), dim=1)
  filters[filters != filters] = 0

  if not use_fcn:
    return segment_measurements, None
  else:
    sampled_values += 0.1
    filters = filters.unsqueeze(1)
    filters = filters * sampled_values.view(bs, nsteps, nlinks, 1, 1)
    filters = filters.sum(dim=2)
  return segment_measurements, filters





def generate_indices_rows_and_columns(images, npoints):
  bs, nsteps, S, _ = images.shape
  weights = torch.ones(S**2).expand(bs, -1).to(images.device)
  indices = torch.multinomial(weights, num_samples=npoints, replacement=False) #.to(images.device)

  # Calculate coordinates from indices
  rows = indices // S
  cols = indices % S

  # Gather the values from these indices for all images
  indices = indices.unsqueeze(dim=1).repeat([1,nsteps,1])
  return indices, rows, cols



def indices_to_sampled_values(images, indices):
  bs, nsteps, S, _ = images.shape
  flat_images = images.view(bs, nsteps, S * S)

  # Gather the values from these indices for all images
  sampled_values = torch.gather(flat_images, 2, indices)
  return sampled_values




def get_point_measurements(rows, cols, sampled_values, S=64):
  # Normalize coordinates to be between 0 and 1
  ys = (1 - rows.float()/S) - 1/(2*S)
  xs = cols.float()/S + 1/(2*S)

  # Stack the normalized coordinates with the values
  point_measurements = torch.cat((xs.unsqueeze(1),
                      ys.unsqueeze(1),
                      sampled_values), dim=1)
  return point_measurements





def point_gt(images, npoints=10, use_fcn=False):
  bs, nsteps, S, _ = images.shape

  indices, rows, cols = generate_indices_rows_and_columns(images, npoints)

  sampled_values = indices_to_sampled_values(images, indices)
  point_measurements = get_point_measurements(rows, cols, sampled_values, S)
  
  if not use_fcn:
    return point_measurements, None, (indices, rows, cols)

  else:
    # Difference with point_gt:
    point_measurements_fcn = -0.1 * torch.ones(images.numel(), device=images.device)
    indices_batch = torch.arange(bs).repeat(60)
    # indice du premier élément de la i ème image pour le premier time step dans images.flatten()
    idx_i000=(torch.arange(bs, device = images.device) * nsteps).view(bs,1).expand(bs,nsteps)
    # indices du premier élément de la i ème image pour le premier time step j dans images.flatten()
    idx_ij00=idx_i000 + torch.arange(nsteps, device = images.device).view(1,nsteps).expand(bs,nsteps)
    # indices à conserver :
    idx_ijkl = S**2 * idx_ij00.unsqueeze(-1) + indices 
    point_measurements_fcn[idx_ijkl.flatten()] = sampled_values.flatten()

    point_measurements_fcn = point_measurements_fcn.view(bs, nsteps, S, S)

    return point_measurements, point_measurements_fcn, (indices, rows, cols)


# Test code point_gt avec use_fcn
bs = 3
S = 5
npoints = 2
nsteps = 7

images = (torch.rand(bs, nsteps, S, S) > 0.5) + (torch.rand(bs, nsteps, S, S) > 0.5)
images = images.float().to('cuda:0')


bs, nsteps, S, _ = images.shape
flat_images = images.view(bs, nsteps, S * S)
# Randomly sample M indices for each image in the batch
indices = torch.randint(0, S * S, (bs, npoints), \
                        device=images.device)

weights = torch.ones(S**2).expand(bs, -1)
indices = torch.multinomial(weights, num_samples=npoints, replacement=False).to(images.device)
indices = indices.unsqueeze(dim=1).repeat([1,nsteps,1])

# Gather the values from these indices for all images
sampled_values = torch.gather(flat_images, 2, indices)
print(sampled_values.shape)


point_measurements = -0.1 * torch.ones(images.numel()).to(device)
indices_batch = torch.arange(bs).repeat(60)
# indice du premier élément de la i ème image pour le premier time step dans images.flatten()
idx_i000=(torch.arange(bs, device = images.device) * nsteps).view(bs,1).expand(bs,nsteps)
# indices du premier élément de la i ème image pour le premier time step j dans images.flatten()
idx_ij00=idx_i000 + torch.arange(nsteps, device = images.device).view(1,nsteps).expand(bs,nsteps)
# indices à conserver :
idx_ijkl = S**2 * idx_ij00.unsqueeze(-1) + indices 
point_measurements[idx_ijkl.flatten()] = sampled_values.flatten()

point_measurements = point_measurements.view(bs, nsteps, S, S)
"""

