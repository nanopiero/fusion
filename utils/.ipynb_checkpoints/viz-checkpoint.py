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


def plot_images_10pts_20seg(images, noisy_images, point_measurements, segment_measurements):
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
            # img_normalized = (img - np.min(img)) / (np.max(img) - np.min(img) + 0.000001)
            ax.imshow(img, cmap='gray', aspect='auto')
            ax.axis('off')

    # Point and Segment measurements plots
    for row in range(12):
        ax_main = axs[row * 7 + 6]  # Last column in each row
        if row < 4:  # First four rows for point measurements
            for idx in range(3) if row in [0,1,2] else range(1):
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

        elif 4 <= row < 11:  # Next 7 rows for segment measurements
            for idx in range(3) if row in [4,5,6,7,8,9,10] else range(1):
                actual_idx = 3 * (row - 4) + idx
                if actual_idx < 20:  # Ensure we don't exceed the 20 graphs
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









def plot_results_10pts_20seg(images, noisy_images, point_measurements, segment_measurements, images_pred, images_pred_5min_mask, point_measurements_pred, segment_measurements_pred):
    # Set up the figure with GridSpec
    fig = plt.figure(figsize=(18, 48))
    gs = gridspec.GridSpec(24, 7, width_ratios=[1, 1, 1, 1, 1, 1, 2])  # Last column twice as wide

    # Manually create axes array for uniform handling as before
    axs = [fig.add_subplot(gs[i, j]) for i in range(24) for j in range(7)]

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
            ax = axs[i*14 + j]
            img = noisy_images[noisy_index] if j == 5 else images[image_indices[j]]
            # img_normalized = (img - np.min(img)) / (np.max(img) - np.min(img) + 0.000001)
            ax.imshow(img, cmap='gray', aspect='auto')
            ax.axis('off')

    # Outputs and masked output plots
    for i in range(12):
        image_indices = [5*i, 5*i+1, 5*i+2, 5*i+3, 5*i+4, 5*i]
        noisy_index = i
        for j in range(6):
            ax = axs[7 + i*14 + j]
            img = images_pred_5min_mask[noisy_index] if j == 5 else images_pred[image_indices[j]]
            # img_normalized = (img - np.min(img)) / (np.max(img) - np.min(img) + 0.000001)
            ax.imshow(img, cmap='gray', aspect='auto')
            ax.axis('off')

    # Point and Segment measurements plots
    for row in range(12):
        ax_main = axs[row * 7 + 6]  # Last column in each row
        if row < 4:  # First four rows for point measurements
            for idx in range(3) if row in [0,1,2] else range(1):
                ax = ax_main.inset_axes([0, 1 - (idx+1)/3, 1, 1/3])
                ax.plot(point_measurements[2:, idx + row*3], marker='.', markevery=(4, 5), markeredgewidth=2, markeredgecolor='black')
                ax.plot(point_measurements_pred[2:, idx + row*3], marker='.', markevery=(4, 5), markeredgewidth=2, markeredgecolor='black', color='red')
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

        elif 4 <= row < 11:  # Next 7 rows for segment measurements
            for idx in range(3) if row in [4,5,6,7,8,9,10] else range(1):
                actual_idx = 3 * (row - 4) + idx
                if actual_idx < 20:  # Ensure we don't exceed the 20 graphs
                    ax = ax_main.inset_axes([0, 1 - (idx+1)/3, 1, 1/3])
                    ax.plot(segment_measurements[4:, actual_idx], marker='.', markevery=(4, 5), markeredgewidth=1, markeredgecolor='black')
                    ax.plot(segment_measurements_pred[4:, actual_idx], marker='.', markevery=(4, 5), markeredgewidth=1, markeredgecolor='black',color='red')
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

def moving_average(data, window_size):
    """
    Smooth a list of points using a moving average.

    Parameters:
    - data (list): The input list of points.
    - window_size (int): The size of the moving average window.

    Returns:    - smoothed_data (list): The smoothed list of points.
    """
    if window_size <= 0:
        raise ValueError("Window size must be greater than 0.")

    half_window = window_size // 2
    smoothed_data = []

    for i in range(len(data)):
        start_index = max(0, i - half_window)
        end_index = min(len(data), i + half_window + 1)
        window_values = data[start_index:end_index]
        average = sum(window_values) / len(window_values)
        smoothed_data.append(average)

    return smoothed_data
