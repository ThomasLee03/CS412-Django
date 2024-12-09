from statistics import mode
from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.urls import reverse
from django.contrib.auth.mixins import LoginRequiredMixin
from matplotlib import pyplot as plt

from cs412 import settings
from .models import (
    Researcher,
    Image,
    ImageGenerated,
    Mask,
    CorruptedImage,
    Paper,
    PaperWithResearcher,
    PaperImage,
    PaperWithGeneratedImage,
    ImageWithGenerated,
    PaperWithCorruptedImage,
    PaperWithMask
)
from itertools import groupby
from .forms import UploadImageForm, CreatePaperForm, ImputationMethodForm, ImputeImageForm, UploadImageForm
from django.views.generic import CreateView, UpdateView, DetailView, DeleteView, ListView, FormView
from django.urls import reverse_lazy
from io import BytesIO
from django.core.files.base import ContentFile
from PIL import Image as PILImage
import numpy as np
from scipy.stats import mode
from math import log10, sqrt
from skimage.metrics import structural_similarity as ssim
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth import login, authenticate, logout
from django.contrib import messages
import cvxpy as cp
def svd_imputation(M, mask, rank=50, max_iters=100, tol=1e-4):
    """
    Perform SVD-based imputation to fill in missing pixels in an image.
    
    Parameters:
    - M: 2D or 3D NumPy array (height x width x channels), the corrupted image matrix
    - mask: 2D NumPy array (height x width), where 1 indicates known pixels, and 0 indicates corrupted pixels
    - rank: int, the rank for the low-rank approximation
    - max_iters: int, the maximum number of iterations
    - tol: float, the convergence tolerance
    
    Returns:
    - M: The image with imputed values in place of corrupted pixels
    """
    # Convert to float for imputation
    M = M.astype(float)
    M_orig = M.copy()

    # Initialize unknown values in the corrupted areas with random values
   # M[mask == 0] = np.random.rand(*M[mask == 0].shape) * 255

    # Repeat until convergence or maximum iterations
    for i in range(max_iters):
        M_prev = M.copy()  # Store the previous image for convergence checking

        # Perform SVD on each color channel separately
        for channel in range(M.shape[2]):
            # Get the SVD of the current channel
            U, S, Vt = np.linalg.svd(M[..., channel], full_matrices=False)
            
            # Keep only the top 'rank' singular values for low-rank approximation
            S = np.diag(S[:rank])
            U = U[:, :rank]
            Vt = Vt[:rank, :]

            # Reconstruct the channel with the low-rank approximation
            M[..., channel] = U @ S @ Vt

        # Re-impose known pixel values (those indicated by mask == 1)
        M[mask == 1] = M_orig[mask == 1]

        # Check for convergence
        if np.linalg.norm(M - M_prev) < tol:
            print(f"Converged in {i + 1} iterations.")
            break

    else:
        print("Reached the maximum number of iterations.")

    

    return M

def PCAbestSSIMPSNR(corrupted_image_array, original_image_array, corrupted_image_mat, mask):
    rankL = []
    PSNRL = []
    SSIML = []
    maxPSNR = 0
    bestRankPSNR = 0
    maxSSIM = 0
    bestRankSSIM = -1

    MPSNRbest = np.zeros(corrupted_image_array.shape)
    MSSIMbest = np.zeros(corrupted_image_array.shape)

    for rank in range(1, 101, 10):  # Loop through rank values
        M = svd_imputed_image = svd_imputation(corrupted_image_mat, mask, rank=rank, max_iters=50, tol=1e-4)
        PSNR_value = PSNR(original_image_array, M)
        print(f"PSNR value is {PSNR_value} dB with {rank} Rank")
        SSIM_value = compute_color_ssim(original_image_array, M)
        print(f"SSIM value is {SSIM_value} with {rank} Rank")

        if PSNR_value > maxPSNR:
            MPSNRbest = M
            maxPSNR = PSNR_value
            bestRankPSNR = rank
        if SSIM_value > maxSSIM:
            MSSIMbest = M
            maxSSIM = SSIM_value
            bestRankSSIM = rank

        SSIML.append(SSIM_value)
        rankL.append(rank)
        PSNRL.append(PSNR_value)

    # Ensure images are in uint8 format before saving
    MPSNRbest = np.clip(MPSNRbest, 0, 255).astype(np.uint8)
    MSSIMbest = np.clip(MSSIMbest, 0, 255).astype(np.uint8)

    return MPSNRbest, MSSIMbest, rankL, PSNRL, SSIML


def PSNR(original, compressed):
    """
    Calculate the Peak Signal-to-Noise Ratio (PSNR) between two images.
    """
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:  # No noise in the signal
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def compute_color_ssim(image1, image2):
    """
    Compute SSIM (Structural Similarity Index) between two color images.
    """
    if image1.shape != image2.shape:
        raise ValueError("Input images must have the same dimensions.")

    ssim_values = []
    for channel in range(3):  # Loop over RGB channels
        ssim_value, _ = ssim(image1[..., channel], image2[..., channel], full=True, data_range=255)
        ssim_values.append(ssim_value)

    avg_ssim_value = np.mean(ssim_values)
    return avg_ssim_value


# Utility function to save an image from a buffer
def save_image_from_buffer(image, filename):
    if isinstance(image, PILImage.Image):  # Ensure image is a PIL.Image instance
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return ContentFile(buffer.getvalue(), name=filename)
    else:
        raise TypeError("The 'image' argument must be a PIL.Image instance.")

def generate_corruption(original_image):
    pil_image = PILImage.open(original_image.image_file)
    image_matrix = np.array(pil_image)

    # Generate a 2D mask for corruption
    mask = np.random.choice([0, 1], size=image_matrix.shape[:2], p=[0.3, 0.7])  # 30% corrupted
    mask_expanded = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

    # Corrupt the image
    corrupted_image_matrix = np.where(mask_expanded == 1, image_matrix, 0)
    corrupted_image = PILImage.fromarray(corrupted_image_matrix.astype(np.uint8))

    # Create mask image
    mask_image = PILImage.fromarray((mask * 255).astype(np.uint8))  # Scale mask to 0-255

    return corrupted_image, mask_image



def column_imputation(image_array, mask_array, impute_strategy="mean"):
    """
    Impute corrupted pixels in an image column by column using a specified strategy.
    """
    # Convert to float to handle NaN values
    image_array = image_array.astype(float)
    image_array[mask_array == 0] = np.nan  # Set corrupted pixels to NaN based on mask

    # Impute each column independently
    for col in range(image_array.shape[1]):
        column_data = image_array[:, col]
        corrupted_pixels = np.isnan(column_data)

        if impute_strategy == "mean":
            column_impute_value = np.nanmean(column_data)
        elif impute_strategy == "median":
            column_impute_value = np.nanmedian(column_data)
        elif impute_strategy == "mode":
            column_impute_value = mode(column_data, nan_policy="omit").mode[0]
        else:
            raise ValueError("Unknown imputation strategy: choose 'mean', 'median', or 'mode'")

        column_data[corrupted_pixels] = column_impute_value

    # Clip values to valid image range and convert to uint8
    image_array = np.clip(image_array, 0, 255).astype(np.uint8)
    return image_array
# PCA-based Imputation Function
from sklearn.decomposition import PCA
def pca_imputation(M, n_components=50):
    """
    Perform PCA-based imputation to reduce and reconstruct the image.

    Parameters:
    - M: 3D NumPy array (image matrix).
    - n_components: Number of principal components to keep.

    Returns:
    - Reconstructed image matrix.
    """
    M_pca = M.copy().astype(float)
    height, width, channels = M_pca.shape

    for channel in range(channels):
        # Reshape the 2D image into a 2D array (height x width)
        reshaped_channel = M_pca[:, :, channel]

        # Fit PCA and transform
        pca = PCA(n_components=n_components)
        reduced = pca.fit_transform(reshaped_channel)

        # Reconstruct the channel
        reconstructed_channel = pca.inverse_transform(reduced)

        # Update the original channel with the reconstructed values
        M_pca[:, :, channel] = reconstructed_channel

    return np.clip(M_pca, 0, 255).astype(np.uint8)



import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend for non-GUI use
import matplotlib.pyplot as plt
import io
import base64
from io import BytesIO
from collections import defaultdict
def imputation_comparison(request):
    # Get the researcher profile for the logged-in user
    researcher = request.user.researcher_profile

    # Fetch the images associated with the researcher
    original_images = Image.objects.filter(researcher=researcher)
    
    bar_graphs = []

    for original_image in original_images:
        # Initialize lists for PSNR, SSIM values, and imputation method labels
        psnr_values = []
        ssim_values = []
        labels = []

        # Retrieve the generated images for the current original image
        generated_images = ImageGenerated.objects.filter(image=original_image)  # Use the foreign key 'image'

        # Loop through the generated images and extract their values
        for generated_image in generated_images:
            psnr_values.append(generated_image.psnr_value)
            ssim_values.append(generated_image.ssim_value)
            labels.append(generated_image.imputation_method)

        # Check if any generated images are available
        if psnr_values:
            # Create PSNR graph
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(labels, psnr_values, color='blue')
            ax.set_title(f'PSNR')
            ax.set_xlabel('Imputation Method')
            ax.set_ylabel('PSNR')
            plt.xticks(fontsize=6)
            # Save PSNR graph to buffer
            buf = BytesIO()
            plt.savefig(buf, format="png")
            plt.close(fig)
            psnr_graph_content = ContentFile(buf.getvalue(), name=f'{original_image.image_file.name}_psnr.png')

            # Create SSIM graph
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(labels, ssim_values, color='green')
            ax.set_title(f'SSIM')
            ax.set_xlabel('Imputation Method')
            ax.set_ylabel('SSIM')
            plt.xticks(fontsize=6)
            # Save SSIM graph to buffer
            buf = BytesIO()
            plt.savefig(buf, format="png")
            plt.close(fig)
            
            ssim_graph_content = ContentFile(buf.getvalue(), name=f'{original_image.image_file.name}_ssim.png')

            # Append the generated data (graphs and original image URL)
            bar_graphs.append({
                'image_file': original_image.image_file.url,  # Use the URL of the original image
                'psnr_graph': psnr_graph_content,
                'ssim_graph': ssim_graph_content,
            })

    # Pass the generated bar graphs to the template
    return render(request, 'FinalProject/imputation_comparison.html', {'bar_graphs': bar_graphs})


def preprocess_with_imputation(M, mask, impute_strategy='mean'):
    """
    Perform imputation on the image using mean, median, or mode.

    Parameters:
    - M: 3D NumPy array (image matrix).
    - mask: 2D NumPy array where 1 indicates known pixels and 0 indicates corrupted pixels.
    - impute_strategy: 'mean', 'median', or 'mode' for imputation strategy.

    Returns:
    - Imputed image matrix.
    """
    M = M.astype(float)
    M[mask == 0] = np.nan

    for col in range(M.shape[1]):
        for channel in range(3):  # Loop over RGB channels
            column_data = M[:, col, channel]
            corrupted_pixels = np.isnan(column_data)

            if impute_strategy == 'mean':
                column_impute_value = np.nanmean(column_data)
            elif impute_strategy == 'median':
                column_impute_value = np.nanmedian(column_data)
            elif impute_strategy == 'mode':
                mode_result = mode(column_data, nan_policy='omit')
                column_impute_value = mode_result.mode if np.isscalar(mode_result.mode) else mode_result.mode[0]
            else:
                raise ValueError("Unknown imputation strategy: choose 'mean', 'median', or 'mode'")

            # Replace NaN values in the column with the imputed value
            column_data[corrupted_pixels] = column_impute_value

    return np.clip(M, 0, 255).astype(np.uint8)




class ImputeImageView(LoginRequiredMixin, FormView):
    template_name = "FinalProject/select_image.html"
    form_class = ImputeImageForm
    success_url = reverse_lazy("dashboard")  # Redirect after successful imputation
    def get_login_url(self) -> str:
        '''return the url of the login page'''
        return reverse('login2')
    def get_form_kwargs(self):
        """Override to pass the logged-in user to the form."""
        kwargs = super().get_form_kwargs()
        kwargs['user'] = self.request.user  # Pass the logged-in user to the form
        return kwargs
    def form_valid(self, form):
        # Get selected original image and imputation method
        original_image = form.cleaned_data["original_image"]
        mask = original_image.mask
        corrupted_image = original_image.corrupted_image
        imputation_method = form.cleaned_data["imputation_method"]
        pca_preprocessing = form.cleaned_data["pca_preprocessing"] 
        
        # Load the corrupted image and mask as NumPy arrays
        corrupted_image_array = np.array(PILImage.open(corrupted_image.corrupted_file).convert("RGB"))
        mask_array = np.array(PILImage.open(mask.mask_file).convert("L")) // 255  # Convert mask to binary (0, 1)
        # Calculate PSNR and SSIM by comparing the original image and the inpainted image
        original_image_array = np.array(PILImage.open(original_image.image_file))  # Load original image
        # Perform PCA-based imputation if the user selects PCA
        if imputation_method == "pca":
            corrupted_image_mat = np.array(corrupted_image_array)
            original_image_array = np.array(PILImage.open(original_image.image_file))

            # Call PCA-based imputation method and track PSNR and SSIM over PCA ranks
            MPSNRbest, MSSIMbest, rankL, PSNRL, SSIML = PCAbestSSIMPSNR(corrupted_image_array, original_image_array, corrupted_image_mat, mask_array)

            # Save the PSNR-based imputed image
            psnr_buffer = BytesIO()
            psnr_image = PILImage.fromarray(MPSNRbest)
            psnr_image.save(psnr_buffer, format="PNG")
            psnr_content_file = ContentFile(psnr_buffer.getvalue(), name="PCA_PSNR_imputed.png")

            # Calculate PSNR and SSIM for PSNR best image
            psnr_value = PSNR(np.array(PILImage.open(original_image.image_file)), MPSNRbest)
            ssim_value_psnr = compute_color_ssim(np.array(PILImage.open(original_image.image_file)), MPSNRbest)

            psnr_imputed_image_instance = ImageGenerated.objects.create(
                image_file=psnr_content_file,
                imputation_method="PCA_PSNR",
                image = original_image,
                researcher=original_image.researcher,
                ssim_value=ssim_value_psnr,
                psnr_value=psnr_value
            )

            # Save the SSIM-based imputed image
            ssim_buffer = BytesIO()
            ssim_image = PILImage.fromarray(MSSIMbest)
            ssim_image.save(ssim_buffer, format="PNG")
            ssim_content_file = ContentFile(ssim_buffer.getvalue(), name="PCA_SSIM_imputed.png")

            # Calculate PSNR and SSIM for SSIM best image
            psnr_value_ssim = PSNR(np.array(PILImage.open(original_image.image_file)), MSSIMbest)
            ssim_value_ssim = compute_color_ssim(np.array(PILImage.open(original_image.image_file)), MSSIMbest)

            ssim_imputed_image_instance = ImageGenerated.objects.create(
                image_file=ssim_content_file,
                imputation_method="PCA_SSIM",
                image=original_image,
                researcher=original_image.researcher,
                ssim_value=ssim_value_ssim,
                psnr_value=psnr_value_ssim
            )

            # Generate the two separate plots for PCA convergence (PSNR and SSIM vs Rank)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))  # Adjusted the figure size for better fitting

            # PSNR Plot
            ax1.plot(rankL, PSNRL, label="PSNR", color='blue', marker='o')
            ax1.set_xlabel("PCA Rank")
            ax1.set_ylabel("PSNR")
            ax1.set_title("PSNR vs. PCA Rank")
            ax1.legend()

            # SSIM Plot
            ax2.plot(rankL, SSIML, label="SSIM", color='green', marker='x')
            ax2.set_xlabel("PCA Rank")
            ax2.set_ylabel("SSIM")
            ax2.set_title("SSIM vs. PCA Rank")
            ax2.legend()

            # Save the plots as PNG and convert to base64
            img_buf = io.BytesIO()
            plt.savefig(img_buf, format='png')
            img_buf.seek(0)
            img_str = base64.b64encode(img_buf.getvalue()).decode('utf8')

            # Render the results page with the plots and images
            return render(
                self.request,
                "FinalProject/imputed_image.html",
                {
                    "original_image": original_image,
                    "corrupted_image": corrupted_image,
                    "psnr_imputed_image": psnr_imputed_image_instance,
                    "ssim_imputed_image": ssim_imputed_image_instance,
                    "imputation_method": "PCA",
                    "psnr_value": psnr_value,
                    "ssim_value": ssim_value_ssim,
                    "pca_plot": img_str,  # Pass the base64 plot string to the template
                },
            )
        elif imputation_method == "total_variation":
            # Perform Total Variation Inpainting

            # Initialize CVXPY variables for the three color channels
            variables = []
            constraints = []
            for i in range(3):  # For each color channel
                U = cp.Variable(shape=(corrupted_image_array.shape[0], corrupted_image_array.shape[1]))
                variables.append(U)
                constraints.append(cp.multiply(mask_array, U) == cp.multiply(mask_array, corrupted_image_array[:, :, i]))

            # Define the TV minimization problem
            prob = cp.Problem(cp.Minimize(cp.tv(*variables)), constraints)
            prob.solve(verbose=True, solver=cp.SCS)

            # Get the reconstructed image
            rec_arr = np.zeros_like(corrupted_image_array)
            for i in range(3):
                rec_arr[:, :, i] = variables[i].value
            rec_arr = np.clip(rec_arr, 0, 255).astype(np.uint8)

            # Save the reconstructed image as a PNG
            rec_image = PILImage.fromarray(rec_arr)
            buffer = BytesIO()
            rec_image.save(buffer, format="PNG")
            content_file = ContentFile(buffer.getvalue(), name="TV_inpainting_imputed.png")
           
            

            # Calculate PSNR and SSIM
            psnr_value = PSNR(original_image_array, rec_arr)  # Compare original with reconstructed
            ssim_value = compute_color_ssim(original_image_array, rec_arr)  # Compare original with reconstructed

            # Calculate PSNR and SSIM for the inpainted image
            psnr_value = PSNR(original_image_array, rec_arr)
            ssim_value = compute_color_ssim(original_image_array, rec_arr)

            # Save the imputed image instance
            imputed_image_instance = ImageGenerated.objects.create(
                image_file=content_file,
                imputation_method="TV Inpainting",
                researcher=original_image.researcher,
                ssim_value=ssim_value,
                image=original_image,
                psnr_value=psnr_value
            )

            # Render the results page
            return render(
                self.request,
                "FinalProject/imputed_image.html",
                {
                    "original_image": original_image,
                    "corrupted_image": corrupted_image,
                    "imputed_image": imputed_image_instance,
                    "imputation_method": "Total Variation Inpainting",
                    "psnr_value": psnr_value,
                    "ssim_value": ssim_value,
                },
            )
        elif pca_preprocessing:
            # Apply PCA reconstruction to the imputed image
            imputed_image = column_imputation(corrupted_image_array, mask_array, imputation_method)

            # If PCA preprocessing is selected, apply PCA

            MPSNRbest, MSSIMbest, rankL, PSNRL, SSIML = PCAbestSSIMPSNR(corrupted_image_array, original_image_array, imputed_image, mask_array)
            # Save the PSNR-based imputed image
            psnr_buffer = BytesIO()
            psnr_image = PILImage.fromarray(MPSNRbest)
            psnr_image.save(psnr_buffer, format="PNG")
            psnr_content_file = ContentFile(psnr_buffer.getvalue(), name="PCA_PSNR_imputed.png")

            # Calculate PSNR and SSIM for PSNR best image
            psnr_value = PSNR(np.array(PILImage.open(original_image.image_file)), MPSNRbest)
            ssim_value_psnr = compute_color_ssim(np.array(PILImage.open(original_image.image_file)), MPSNRbest)

            psnr_imputed_image_instance = ImageGenerated.objects.create(
                image_file=psnr_content_file,
                imputation_method=f"{imputation_method} + PCA_PSNR",
                image = original_image,
                researcher=original_image.researcher,
                ssim_value=ssim_value_psnr,
                psnr_value=psnr_value
            )

            # Save the SSIM-based imputed image
            ssim_buffer = BytesIO()
            ssim_image = PILImage.fromarray(MSSIMbest)
            ssim_image.save(ssim_buffer, format="PNG")
            ssim_content_file = ContentFile(ssim_buffer.getvalue(), name="PCA_SSIM_imputed.png")

            # Calculate PSNR and SSIM for SSIM best image
            psnr_value_ssim = PSNR(np.array(PILImage.open(original_image.image_file)), MSSIMbest)
            ssim_value_ssim = compute_color_ssim(np.array(PILImage.open(original_image.image_file)), MSSIMbest)

            ssim_imputed_image_instance = ImageGenerated.objects.create(
                image_file=ssim_content_file,
                imputation_method=f"{imputation_method} + PCA_SSIM",
                image=original_image,
                researcher=original_image.researcher,
                ssim_value=ssim_value_ssim,
                psnr_value=psnr_value_ssim
            )

            # Generate the two separate plots for PCA convergence (PSNR and SSIM vs Rank)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))  # Adjusted the figure size for better fitting

            # PSNR Plot
            ax1.plot(rankL, PSNRL, label="PSNR", color='blue', marker='o')
            ax1.set_xlabel("PCA Rank")
            ax1.set_ylabel("PSNR")
            ax1.set_title("PSNR vs. PCA Rank")
            ax1.legend()

            # SSIM Plot
            ax2.plot(rankL, SSIML, label="SSIM", color='green', marker='x')
            ax2.set_xlabel("PCA Rank")
            ax2.set_ylabel("SSIM")
            ax2.set_title("SSIM vs. PCA Rank")
            ax2.legend()

            # Save the plots as PNG and convert to base64
            img_buf = io.BytesIO()
            plt.savefig(img_buf, format='png')
            img_buf.seek(0)
            img_str = base64.b64encode(img_buf.getvalue()).decode('utf8')

            # Render the results page with the plots and images
            return render(
                self.request,
                "FinalProject/imputed_image.html",
                {
                    "original_image": original_image,
                    "corrupted_image": corrupted_image,
                    "psnr_imputed_image": psnr_imputed_image_instance,
                    "ssim_imputed_image": ssim_imputed_image_instance,
                    "imputation_method": "PCA",
                    "psnr_value": psnr_value,
                    "ssim_value": ssim_value_ssim,
                    "pca_plot": img_str,  # Pass the base64 plot string to the template
                },
            )
        else:
            imputed_image_array = column_imputation(
                np.array(PILImage.open(corrupted_image.corrupted_file).convert("RGB")),
                np.array(PILImage.open(mask.mask_file).convert("L")),  # Convert mask to grayscale
                impute_strategy=imputation_method
            )
            # Save the imputed image
            imputed_image = PILImage.fromarray(imputed_image_array)
            buffer = BytesIO()
            imputed_image.save(buffer, format="PNG")
            content_file = ContentFile(buffer.getvalue(), name=f"imputed.png")

            # Calculate PSNR and SSIM
            psnr_value = PSNR(
                np.array(PILImage.open(original_image.image_file).convert("RGB")),
                imputed_image_array
            )
            ssim_value = compute_color_ssim(
                np.array(PILImage.open(original_image.image_file).convert("RGB")),
                imputed_image_array
            )

            # Create an ImageGenerated instance for the imputed image
            imputed_image_instance = ImageGenerated.objects.create(
                image_file=content_file,
                imputation_method=imputation_method,
                researcher=original_image.researcher,
                ssim_value=ssim_value,
                image=original_image,
                psnr_value=psnr_value
                
            )
            # Render the results page
            return render(
                self.request,
                "FinalProject/imputed_image.html",
                {
                    "original_image": original_image,
                    "corrupted_image": corrupted_image,
                    "imputed_image": imputed_image_instance,
                    "imputation_method": imputation_method,
                    "psnr_value": psnr_value,
                    "ssim_value": ssim_value,
                },
            )

def apply_imputation(request):
    if request.method == "POST":
        form = ImputationMethodForm(request.POST)
        if form.is_valid():
            corrupted_image_id = form.cleaned_data["corrupted_image_id"]
            imputation_method = form.cleaned_data["imputation_method"]
            pca_preprocessing = form.cleaned_data["pca_preprocessing"]  # Get the PCA preprocessing checkbox status

            # Fetch the corrupted image
            corrupted_image = get_object_or_404(CorruptedImage, pk=corrupted_image_id)

            # Find the original image that references this corrupted image
            original_image = get_object_or_404(Image, corrupted_image=corrupted_image)

            # Fetch the mask associated with the original image
            mask = original_image.mask

            if not mask:
                return redirect("import_image")  # Handle the case where there's no mask

            # Load corrupted image and mask into numpy arrays
            corrupted_image_array = np.array(PILImage.open(corrupted_image.corrupted_file))
            mask_array = np.array(PILImage.open(mask.mask_file)) // 255  # Convert mask to binary (0, 1)

            corrupted_image_mat = np.array(corrupted_image_array)

            if imputation_method == "pca":
                # Apply PCA-based imputation and get best images for PSNR and SSIM
                original_image_array = np.array(PILImage.open(original_image.image_file))

                MPSNRbest, MSSIMbest, rankL, PSNRL, SSIML = PCAbestSSIMPSNR(corrupted_image_array, original_image_array, corrupted_image_mat, mask_array)

                # Save the PSNR-based imputed image
                psnr_buffer = BytesIO()
                psnr_image = PILImage.fromarray(MPSNRbest)
                psnr_image.save(psnr_buffer, format="PNG")
                psnr_content_file = ContentFile(psnr_buffer.getvalue(), name="PCA_PSNR_imputed.png")

                # Calculate PSNR and SSIM for PSNR best image
                psnr_value = PSNR(np.array(PILImage.open(original_image.image_file)), MPSNRbest)
                ssim_value_psnr = compute_color_ssim(np.array(PILImage.open(original_image.image_file)), MPSNRbest)

                psnr_imputed_image_instance = ImageGenerated.objects.create(
                    image=original_image,  # Link to the original image
                    image_file=psnr_content_file,
                    imputation_method="PCA_PSNR",
                    researcher=original_image.researcher,
                    ssim_value=ssim_value_psnr,
                    psnr_value=psnr_value
                )

                # Save the SSIM-based imputed image
                ssim_buffer = BytesIO()
                ssim_image = PILImage.fromarray(MSSIMbest)
                ssim_image.save(ssim_buffer, format="PNG")
                ssim_content_file = ContentFile(ssim_buffer.getvalue(), name="PCA_SSIM_imputed.png")

                # Calculate PSNR and SSIM for SSIM best image
                psnr_value_ssim = PSNR(np.array(PILImage.open(original_image.image_file)), MSSIMbest)
                ssim_value_ssim = compute_color_ssim(np.array(PILImage.open(original_image.image_file)), MSSIMbest)

                ssim_imputed_image_instance = ImageGenerated.objects.create(
                    image=original_image,  # Link to the original image
                    image_file=ssim_content_file,
                    imputation_method="PCA_SSIM",
                    researcher=original_image.researcher,
                    ssim_value=ssim_value_ssim,
                    psnr_value=psnr_value_ssim
                )

                # Generate PCA convergence graphs for PSNR and SSIM vs Rank
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))  # Adjusted size for better fit

                # PSNR vs PCA Rank
                ax1.plot(rankL, PSNRL, label="PSNR", color='blue', marker='o')
                ax1.set_xlabel("PCA Rank")
                ax1.set_ylabel("PSNR")
                ax1.set_title("PSNR vs. PCA Rank")
                ax1.legend()

                # SSIM vs PCA Rank
                ax2.plot(rankL, SSIML, label="SSIM", color='green', marker='x')
                ax2.set_xlabel("PCA Rank")
                ax2.set_ylabel("SSIM")
                ax2.set_title("SSIM vs. PCA Rank")
                ax2.legend()

                # Save the plots as PNG and convert to base64 for embedding
                img_buf = io.BytesIO()
                plt.savefig(img_buf, format='png')
                img_buf.seek(0)
                img_str = base64.b64encode(img_buf.getvalue()).decode('utf8')

                # Render the results page with the images and graphs
                return render(
                    request,
                    "FinalProject/imputed_image.html",
                    {
                        "original_image": original_image,
                        "corrupted_image": corrupted_image,
                        "psnr_imputed_image": psnr_imputed_image_instance,
                        "ssim_imputed_image": ssim_imputed_image_instance,
                        "imputation_method": "PCA",
                        "psnr_value": psnr_value,
                        "ssim_value": ssim_value_ssim,
                        "pca_plot": img_str,  # Pass the graph as base64 to template
                    },
                )
            elif pca_preprocessing:
                # Apply PCA reconstruction to the imputed image
                imputed_image = column_imputation(corrupted_image_array, mask_array, imputation_method)

                # If PCA preprocessing is selected, apply PCA
                imputed_image = pca_imputation(imputed_image, 50)


                # Calculate PSNR and SSIM for the PCA-reconstructed image
                psnr_value = PSNR(np.array(PILImage.open(original_image.image_file)), imputed_image)
                ssim_value = compute_color_ssim(np.array(PILImage.open(original_image.image_file)), imputed_image)

                # Save the PCA-reconstructed image
                buffer = BytesIO()
                pca_image = PILImage.fromarray(imputed_image)
                pca_image.save(buffer, format="PNG")
                content_file = ContentFile(buffer.getvalue(), name="PCA_imputed.png")

                # Save the imputed image instance
                imputed_image_instance = ImageGenerated.objects.create(
                    image_file=content_file,
                    imputation_method=f"{imputation_method} + PCA",
                    researcher=original_image.researcher,
                    ssim_value=ssim_value,
                    image=original_image,
                    psnr_value=psnr_value
                )

                # Render the results page for PCA preprocessing
                return render(
                    request,
                    "FinalProject/imputed_image.html",
                    {
                        "original_image": original_image,
                        "corrupted_image": corrupted_image,
                        "imputed_image": imputed_image_instance,
                        "imputation_method": f"{imputation_method} + PCA",
                        "psnr_value": psnr_value,
                        "ssim_value": ssim_value,
                    },
                )
            elif imputation_method == "total_variation":
                # Perform Total Variation Inpainting

                # Initialize CVXPY variables for the three color channels
                variables = []
                constraints = []
                for i in range(3):  # For each color channel
                    U = cp.Variable(shape=(corrupted_image_array.shape[0], corrupted_image_array.shape[1]))
                    variables.append(U)
                    constraints.append(cp.multiply(mask_array, U) == cp.multiply(mask_array, corrupted_image_array[:, :, i]))

                # Define the TV minimization problem
                prob = cp.Problem(cp.Minimize(cp.tv(*variables)), constraints)
                prob.solve(verbose=True, solver=cp.SCS)

                # Get the reconstructed image
                rec_arr = np.zeros_like(corrupted_image_array)
                for i in range(3):
                    rec_arr[:, :, i] = variables[i].value
                rec_arr = np.clip(rec_arr, 0, 255).astype(np.uint8)

                # Save the reconstructed image as a PNG
                rec_image = PILImage.fromarray(rec_arr)
                buffer = BytesIO()
                rec_image.save(buffer, format="PNG")
                content_file = ContentFile(buffer.getvalue(), name="TV_inpainting_imputed.png")
            
                # Calculate PSNR and SSIM by comparing the original image and the inpainted image
                original_image_array = np.array(PILImage.open(original_image.image_file))  # Load original image

                # Calculate PSNR and SSIM
                psnr_value = PSNR(original_image_array, rec_arr)  # Compare original with reconstructed
                ssim_value = compute_color_ssim(original_image_array, rec_arr)  # Compare original with reconstructed

                # Calculate PSNR and SSIM for the inpainted image
                psnr_value = PSNR(corrupted_image_array, rec_arr)
                ssim_value = compute_color_ssim(corrupted_image_array, rec_arr)

                # Save the imputed image instance
                imputed_image_instance = ImageGenerated.objects.create(
                    image_file=content_file,
                    imputation_method="TV Inpainting",
                    researcher=original_image.researcher,
                    ssim_value=ssim_value,
                    image=original_image,
                    psnr_value=psnr_value
                )

                # Render the results page
                return render(
                    "FinalProject/imputed_image.html",
                    {
                        "original_image": original_image,
                        "corrupted_image": corrupted_image,
                        "imputed_image": imputed_image_instance,
                        "imputation_method": "Total Variation Inpainting",
                        "psnr_value": psnr_value,
                        "ssim_value": ssim_value,
                    },
                )
            else:
                # Apply column imputation for other methods (mean, median, mode)
                imputed_image_array = column_imputation(corrupted_image_array, mask_array, imputation_method)

                # Save the imputed image
                buffer = BytesIO()
                imputed_image = PILImage.fromarray(imputed_image_array)
                imputed_image.save(buffer, format="PNG")
                content_file = ContentFile(buffer.getvalue(), name=f"{imputation_method}_imputed.png")

                # Calculate PSNR and SSIM
                original_image_array = np.array(PILImage.open(original_image.image_file))
                psnr_value = PSNR(original_image_array, imputed_image_array)
                ssim_value = compute_color_ssim(original_image_array, imputed_image_array)

                imputed_image_instance = ImageGenerated.objects.create(
                    image_file=content_file,
                    imputation_method=imputation_method,
                    researcher=original_image.researcher,
                    image=original_image,
                    ssim_value=ssim_value,
                    psnr_value=psnr_value
                )

                # Render the results page for non-PCA imputation
                return render(
                    request,
                    "FinalProject/imputed_image.html",
                    {
                        "original_image": original_image,
                        "corrupted_image": corrupted_image,
                        "imputed_image": imputed_image_instance,
                        "imputation_method": imputation_method,
                        "psnr_value": psnr_value,
                        "ssim_value": ssim_value,
                    },
                )

    return redirect("import_image")


class ImputedImageDetailView(LoginRequiredMixin, DetailView):
    def get_login_url(self) -> str:
        '''return the url of the login page'''
        return reverse('login2')
    model = ImageGenerated
    template_name = "FinalProject/imputed_image.html"
    context_object_name = "imputed_image"

# DeleteView for Image
class ImageDeleteView(LoginRequiredMixin, DeleteView):
    def get_login_url(self) -> str:
        '''return the url of the login page'''
        return reverse('login2')
    model = Image
    template_name = "FinalProject/delete_image.html"
    success_url = reverse_lazy("dashboard")  # Redirect to the dashboard after deletion

    def get_queryset(self):
        # Ensure only the images belonging to the logged-in user are deletable
        researcher = self.request.user.researcher_profile
        return Image.objects.filter(researcher=researcher)

# DeleteView for ImageGenerated
class ImageGeneratedDeleteView(LoginRequiredMixin, DeleteView):
    def get_login_url(self) -> str:
        '''return the url of the login page'''
        return reverse('login2')
    model = ImageGenerated
    template_name = "FinalProject/delete_generated_image.html"
    success_url = reverse_lazy("dashboard")  # Redirect to the dashboard after deletion

    def get_queryset(self):
        # Ensure only the generated images belonging to the logged-in user are deletable
        researcher = self.request.user.researcher_profile
        return ImageGenerated.objects.filter(researcher=researcher)


class ImputationCreateView(LoginRequiredMixin, CreateView):
    def get_login_url(self) -> str:
        '''return the url of the login page'''
        return reverse('login2')
    model = ImageGenerated
    form_class = ImputationMethodForm
    template_name = "FinalProject/imputation_form.html"

    def form_valid(self, form):
        # Get the corrupted image from the form data
        corrupted_image_id = form.cleaned_data["corrupted_image_id"]
        imputation_method = form.cleaned_data["imputation_method"]

        # Fetch the corrupted image object
        corrupted_image = get_object_or_404(Image, pk=corrupted_image_id)

        # Placeholder logic to create the imputed image
        # Replace this with your actual imputation logic
        self.object = ImageGenerated.objects.create(
            image_file=corrupted_image.image_file,
            imputation_method=imputation_method,
            researcher=corrupted_image.researcher,
        )

        # Redirect to the detail view for the imputed image
        return redirect(
            reverse_lazy(
                "view_imputed_image",
                kwargs={"pk": self.object.pk},
            )
        )
    

class PaperDetailView(LoginRequiredMixin, DetailView):
    def get_login_url(self) -> str:
        '''return the url of the login page'''
        return reverse('login2')
    model = Paper
    template_name = 'FinalProject/view_paper.html'
    context_object_name = 'paper'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['processed_content'] = self.object.render_content()
        return context
    
def login_view(request):
    if request.method == "POST":
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            messages.success(request, "You have successfully logged in.")
            return redirect('dashboard')  # Redirect to your dashboard or desired page
        else:
            messages.error(request, "Invalid credentials. Please try again.")
    else:
        form = AuthenticationForm()

    return render(request, 'FinalProject/login.html', {'form': form})

def logout_view(request):
    logout(request)
    return redirect('login2')

# CreateView for uploading and generating images
class ImageImportCreateView(LoginRequiredMixin, CreateView):
    def get_login_url(self) -> str:
        '''return the url of the login page'''
        return reverse('login2')
    model = Image
    form_class = UploadImageForm
    template_name = "FinalProject/import_image.html"

    def form_valid(self, form):
        # Ensure the user has a related Researcher profile
        researcher = self.request.user.researcher_profile
        if not researcher:
            return redirect("login")

        # Check if the image file is actually uploaded
        if not form.cleaned_data['image_file']:
            raise ValueError("No image file uploaded.")

        # Set the researcher field on the form instance
        form.instance.researcher = researcher

        # Save the form instance (this creates the original image)
        original_image = form.save()

        # Generate corrupted image and mask
        corrupted_image, mask_image = generate_corruption(original_image)

        # Save the corrupted image
        corrupted_content = save_image_from_buffer(corrupted_image, "corrupted_image.png")
        corrupted_image_instance = CorruptedImage.objects.create(
            corrupted_file=corrupted_content,
            researcher=researcher  # Set the researcher field
        )

        # Save the mask as another image
        mask_content = save_image_from_buffer(mask_image, "corruption_mask.png")
        mask_instance = Mask.objects.create(
            mask_file=mask_content,
            researcher=researcher  # Set the researcher field
        )

        # Link the corrupted image and mask to the original image
        original_image.corrupted_image = corrupted_image_instance
        original_image.mask = mask_instance
        original_image.save()

        # Redirect to a success page
        self.success_url = reverse_lazy("view_generated_image", kwargs={"pk": original_image.pk})
        return super().form_valid(form)

class PaperSearchView(LoginRequiredMixin, ListView):
    def get_login_url(self) -> str:
        '''return the url of the login page'''
        return reverse('login2')
    model = Paper
    template_name = 'FinalProject/paper_search.html'
    context_object_name = 'papers'
    paginate_by = 10

    def get_queryset(self):
        queryset = Paper.objects.all()

        # Filter by title and researcher
        title_query = self.request.GET.get('title', '')
        researcher_query = self.request.GET.get('researcher', '')

        if title_query:
            queryset = queryset.filter(title__icontains=title_query)
        if researcher_query:
            queryset = queryset.filter(
                paperwithresearcher__researcher__first_name__icontains=researcher_query
            ) | queryset.filter(
                paperwithresearcher__researcher__last_name__icontains=researcher_query
            )

        # Order by creation date (if specified)
        order_by = self.request.GET.get('order_by', 'time_created')  # Default to 'created_at'
        if order_by == 'date':
            queryset = queryset.order_by('time_created')  # Order by creation date
        else:
            queryset = queryset.order_by('-time_created')  # Default descending order by creation date

        return queryset.distinct()

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['request'] = self.request  # Explicitly pass request
        return context

# DetailView for displaying images
class ImageGeneratedDetailView(LoginRequiredMixin, DetailView):
    def get_login_url(self) -> str:
        '''return the url of the login page'''
        return reverse('login2')
    model = Image
    template_name = "FinalProject/generated_image_display.html"
    context_object_name = "original_image"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        original_image = self.object

        context.update({
            "corrupted_image": original_image.corrupted_image,
            "mask": original_image.mask,
        })
        return context


class PaperCreateView(LoginRequiredMixin, CreateView):
    def get_login_url(self) -> str:
        '''return the url of the login page'''
        return reverse('login2')
    model = Paper
    form_class = CreatePaperForm
    template_name = "FinalProject/create_paper.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        researcher = self.request.user.researcher_profile
        context["images"] = Image.objects.filter(researcher=researcher)
        context["generated_images"] = ImageGenerated.objects.filter(researcher=researcher)
        context["corrupted_images"] = CorruptedImage.objects.all()
        context["masks"] = Mask.objects.all()
        return context

    def form_valid(self, form):
        # Save the paper instance
        paper = form.save()

        # Get the current researcher
        researcher = self.request.user.researcher_profile

        # Create a PaperWithResearcher instance to link the paper with the researcher
        PaperWithResearcher.objects.create(researcher=researcher, paper=paper)

        # Clear existing relationships to avoid duplicates
        PaperImage.objects.filter(paper=paper).delete()
        PaperWithGeneratedImage.objects.filter(paper=paper).delete()
        PaperWithCorruptedImage.objects.filter(paper=paper).delete()
        PaperWithMask.objects.filter(paper=paper).delete()

        # Handle regular images
        order_counter = 1
        images = self.request.POST.getlist('images')
        for image_id in images:
            PaperImage.objects.create(paper=paper, image_id=image_id, order=order_counter)
            order_counter += 1

        # Handle generated images
        order_counter = 1
        generated_images = self.request.POST.getlist('generated_images')
        for gen_image_id in generated_images:
            PaperWithGeneratedImage.objects.create(paper=paper, generated_image_id=gen_image_id, order=order_counter)
            order_counter += 1

        # Handle corrupted images
        order_counter = 1
        corrupted_images = self.request.POST.getlist('corrupted_images')
        for corr_image_id in corrupted_images:
            PaperWithCorruptedImage.objects.create(paper=paper, corrupted_image_id=corr_image_id, order=order_counter)
            order_counter += 1

        # Handle masks
        order_counter = 1
        masks = self.request.POST.getlist('masks')
        for mask_id in masks:
            PaperWithMask.objects.create(paper=paper, mask_id=mask_id, order=order_counter)
            order_counter += 1

        return super().form_valid(form)

    def get_success_url(self):
        return reverse_lazy("dashboard")

class PaperUpdateView(LoginRequiredMixin, UpdateView):
    def get_login_url(self) -> str:
        '''return the url of the login page'''
        return reverse('login2')
    model = Paper
    form_class = CreatePaperForm
    template_name = "FinalProject/edit_paper.html"

    def get_success_url(self):
        # Return the URL to the paper detail view after successful update
        return reverse('paper_detail', kwargs={'pk': self.object.pk})

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        researcher = self.request.user.researcher_profile

        # Ensure the paper belongs to the current researcher
        if not PaperWithResearcher.objects.filter(researcher=researcher, paper=self.object).exists():
            return redirect("dashboard")

        context["images"] = Image.objects.filter(researcher=researcher)
        context["generated_images"] = ImageGenerated.objects.filter(researcher=researcher)

        # Filter corrupted images and masks to show only those related to the logged-in researcher
        context["corrupted_images"] = CorruptedImage.objects.filter(researcher=researcher)
        context["masks"] = Mask.objects.filter(researcher=researcher)
        
        # Linked images, generated images, corrupted images, and masks to be checked in the form
        context["linked_images"] = {img.image.id: True for img in self.object.paperimage_set.all()}
        context["linked_generated_images"] = {gen_img.generated_image.id: True for gen_img in self.object.paperwithgeneratedimage_set.all()}
        context["linked_corrupted_images"] = {corr_img.corrupted_image.id: True for corr_img in self.object.paperwithcorruptedimage_set.all()}
        context["linked_masks"] = {mask.mask.id: True for mask in self.object.paperwithmask_set.all()}

        return context
    def form_valid(self, form):
        paper = form.save()

        # Clear existing relationships to avoid duplicates
        PaperImage.objects.filter(paper=paper).delete()
        PaperWithGeneratedImage.objects.filter(paper=paper).delete()
        PaperWithCorruptedImage.objects.filter(paper=paper).delete()
        PaperWithMask.objects.filter(paper=paper).delete()

        # Handle regular images
        order_counter = 1
        images = self.request.POST.getlist('images')
        for image_id in images:
            PaperImage.objects.create(paper=paper, image_id=image_id, order=order_counter)
            order_counter += 1

        # Handle generated images
        order_counter = 1
        generated_images = self.request.POST.getlist('generated_images')
        for gen_image_id in generated_images:
            PaperWithGeneratedImage.objects.create(paper=paper, generated_image_id=gen_image_id, order=order_counter)
            order_counter += 1

        # Handle corrupted images
        order_counter = 1
        corrupted_images = self.request.POST.getlist('corrupted_images')
        for corr_image_id in corrupted_images:
            PaperWithCorruptedImage.objects.create(paper=paper, corrupted_image_id=corr_image_id, order=order_counter)
            order_counter += 1

        # Handle masks
        order_counter = 1
        masks = self.request.POST.getlist('masks')
        for mask_id in masks:
            PaperWithMask.objects.create(paper=paper, mask_id=mask_id, order=order_counter)
            order_counter += 1

        return super().form_valid(form)







class PaperDeleteView(LoginRequiredMixin, DeleteView):
    def get_login_url(self) -> str:
        '''return the url of the login page'''
        return reverse('login2')
    model = Paper
    template_name = "FinalProject/delete_paper.html"
    success_url = reverse_lazy("dashboard")

    def get_queryset(self):
        researcher = self.request.user.researcher_profile
        return Paper.objects.filter(paperwithresearcher__researcher=researcher).distinct()



class CorruptedImageListView(LoginRequiredMixin, ListView):
    def get_login_url(self) -> str:
        '''return the url of the login page'''
        return reverse('login2')
    model = CorruptedImage
    template_name = 'FinalProject/corrupted_image_list.html'
    context_object_name = 'corrupted_images'

class CorruptedImageDetailView(LoginRequiredMixin, DetailView):
    def get_login_url(self) -> str:
        '''return the url of the login page'''
        return reverse('login2')
    model = CorruptedImage
    template_name = 'FinalProject/corrupted_image_detail.html'
    context_object_name = 'corrupted_image'

class MaskListView(LoginRequiredMixin, ListView):
    def get_login_url(self) -> str:
        '''return the url of the login page'''
        return reverse('login2')
    model = Mask
    template_name = 'FinalProject/mask_list.html'
    context_object_name = 'masks'

class MaskDetailView(LoginRequiredMixin, DetailView):
    def get_login_url(self) -> str:
        '''return the url of the login page'''
        return reverse('login2')
    model = Mask
    template_name = 'FinalProject/mask_detail.html'
    context_object_name = 'mask'


class DashboardView(LoginRequiredMixin, ListView):
    def get_login_url(self) -> str:
        '''return the url of the login page'''
        return reverse('login2')
    template_name = "FinalProject/dashboard.html"
    context_object_name = "papers"

    def get_queryset(self):
        researcher = self.request.user.researcher_profile
        return Paper.objects.filter(paperwithresearcher__researcher=researcher)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        researcher = self.request.user.researcher_profile

        # Get images uploaded by the researcher
        context["images"] = Image.objects.filter(researcher=researcher)
        
        # Get all generated images related to the researcher
        context["generated_images"] = ImageGenerated.objects.filter(researcher=researcher)

        # Get corrupted images associated with the researcher's images
        context["corrupted_images"] = CorruptedImage.objects.filter(
            corrupted_images__researcher=researcher
        ).distinct()

        # Get masks associated with the researcher's images
        context["masks"] = Mask.objects.filter(
            masked_images__researcher=researcher
        ).distinct()

        return context




