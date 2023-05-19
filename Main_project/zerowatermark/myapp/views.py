from django.shortcuts import render, redirect
from django.contrib import messages
from myapp.models import UploadedFile
from django.conf import settings
from django.templatetags.static import static
import shutil
import os
import cv2
import numpy as np
import pywt
from skimage import io, exposure, color, transform
from Crypto.Random import get_random_bytes
import random


# Load the four original images in grayscale mode
image1 = cv2.imread('E:/S4/Project/Main Project/images/1.jpeg', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('E:/S4/Project/Main Project/images/2.jpeg', cv2.IMREAD_GRAYSCALE)
image3 = cv2.imread('E:/S4/Project/Main Project/images/3.jpeg', cv2.IMREAD_GRAYSCALE)
image4 = cv2.imread('E:/S4/Project/Main Project/images/4.jpeg', cv2.IMREAD_GRAYSCALE)

# Resize the images to a fixed size
M = 512
image1 = cv2.resize(image1, (M, M))
image2 = cv2.resize(image2, (M, M))
image3 = cv2.resize(image3, (M, M))
image4 = cv2.resize(image4, (M, M))

# Normalize the images
image1 = cv2.normalize(image1.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
image2 = cv2.normalize(image2.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
image3 = cv2.normalize(image3.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
image4 = cv2.normalize(image4.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

# Fuse the images
fused1 = cv2.addWeighted(image1, 0.5, image2, 0.5, 0)
fused2 = cv2.addWeighted(image3, 0.5, image4, 0.5, 0)

final_fused = cv2.addWeighted(fused1, 0.5, fused2, 0.5, 0)

# Save fused image
cv2.imwrite('E:/S4/Project/Main Project/images/newcode_fused_image.jpg', final_fused * 255)
print(final_fused)

# Load fused image
image = cv2.imread('E:/S4/Project/Main Project/images/newcode_fused_image.jpg', 0)

# Extract effective area of size N x N
N = 256
start_x = int((image.shape[0] - N) / 2)
start_y = int((image.shape[1] - N) / 2)
effective_area = image[start_x:start_x + N, start_y:start_y + N]

# Save effective area image
cv2.imwrite('E:/S4/Project/Main Project/images/new_effective_area.jpg', effective_area)

# Load effective area image
image = cv2.imread('E:/S4/Project/Main Project/images/new_effective_area.jpg', 0)

# Obtain the approximate coefficient image of low frequency using the 2D discrete wavelet transform
coeffs = pywt.dwt2(image, 'haar')
cA, (cH, cV, cD) = coeffs

# Save approximate coefficient image
cv2.imwrite('E:/S4/Project/Main Project/images/new_approx_coeff.jpg', cA)

# Load approximate coefficient image
image = cv2.imread('E:/S4/Project/Main Project/images/new_approx_coeff.jpg', 0)

# Decompose image into blocks of size n x n
n = 16
blocks = [image[i:i+n, j:j+n] for i in range(0, image.shape[0], n) for j in range(0, image.shape[1], n)]

# Save blocks as a numpy array
np.save('E:/S4/Project/Main Project/images/blocks.npy', blocks)

# Load blocks
blocks = np.load('E:/S4/Project/Main Project/images/blocks.npy', allow_pickle=True)

# Obtain a 2-norm matrix of size k x k
k = 4
norms = []
for block in blocks:
    norm = np.linalg.norm(block, 2)
    norms.append(norm)
norms = np.array(norms).reshape(int(image.shape[0]/n), int(image.shape[1]/n))

# Save 2-norm matrix
np.save('E:/S4/Project/Main Project/images/norms.npy', norms)

# Convert norms to uint8 type and normalize the pixel values
norms = cv2.normalize(norms, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# Apply Otsu's thresholding
_, feature_image = cv2.threshold(norms, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

feature_image = cv2.resize(feature_image,(256,256))

# Save feature image
cv2.imwrite('E:/S4/Project/Main Project/images/new_feature_image.jpg', feature_image)
print(feature_image)

def generate_shared_images():
    logo_image = read_latest_uploaded_image()
    if logo_image is not None:
        # Get the size of the image
        height, width, _ = logo_image.shape

        # Generate two random bitmaps of the same size as the logo image
        key_size = height * width
        key1 = random.randint(0, 2**key_size - 1)
        key2 = key1 ^ random.randint(0, 2**key_size - 1)
        bitmap1 = np.unpackbits(np.frombuffer(key1.to_bytes((key_size + 7) // 8, byteorder='big'), dtype=np.uint8))
        bitmap2 = np.unpackbits(np.frombuffer(key2.to_bytes((key_size + 7) // 8, byteorder='big'), dtype=np.uint8))
        bitmap1 = bitmap1[:height*width].reshape(height, width)
        bitmap2 = bitmap2[:height*width].reshape(height, width)

        # Generate two shared images using non-extended visual cryptography
        shared1 = cv2.merge((cv2.bitwise_and(logo_image[:,:,0], bitmap1 * 255), cv2.bitwise_and(logo_image[:,:,1], bitmap1 * 255), cv2.bitwise_and(logo_image[:,:,2], bitmap1 * 255)))
        shared2 = cv2.merge((cv2.bitwise_and(logo_image[:,:,0], bitmap2 * 255), cv2.bitwise_and(logo_image[:,:,1], bitmap2 * 255), cv2.bitwise_and(logo_image[:,:,2], bitmap2 * 255)))

        # Save shared images
        cv2.imwrite('E:/S4/Project/Main Project/images/new_shared1.jpg', shared1)
        cv2.imwrite('E:/S4/Project/Main Project/images/new_shared2.jpg', shared2)

        return shared1, shared2
    else:
        return None, None

def read_latest_uploaded_image():
    try:
        latest_uploaded_file = UploadedFile.objects.latest('uploaded_at')
        image_data = np.frombuffer(latest_uploaded_file.file.read(), np.uint8)
        logo_image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        return logo_image
    except:
        return None

def generate_zero_watermark():
    # Load feature image
    feature_image = cv2.imread('E:/S4/Project/Main Project/images/new_feature_image.jpg')

    # Obtain binary representation of feature image
    feature_image_bin = np.unpackbits(feature_image)

    # Load shared images
    shared2 = cv2.imread('E:/S4/Project/Main Project/images/new_shared2.jpg')

    # shared2_reshaped = shared2.reshape(-1, 1)[:feature_image_bin.size].reshape(feature_image_bin.shape)
    feature_image = cv2.resize(feature_image, (512, 512))
    shared2 = cv2.resize(shared2, (512, 512))

    # zero_watermark = cv2.bitwise_xor(feature_image_bin, shared2_reshaped)
    zero_watermark = cv2.bitwise_xor(feature_image, shared2)

    # Save zero-watermark image to local directory
    local_zero_watermark_path = 'E:/S4/Project/Main Project/images/new_zero_watermark.jpg'
    cv2.imwrite(local_zero_watermark_path, zero_watermark)

    # Save zero-watermark image to static directory
    static_zero_watermark_path = os.path.join(settings.STATICFILES_DIRS[0], 'images/new_zero_watermark.jpg')
    shutil.copyfile(local_zero_watermark_path, static_zero_watermark_path)

    return local_zero_watermark_path, static_zero_watermark_path

def upload(request):
    context = {
        'zero_watermark_path': None,
        'show_popup': False,
        'show_image': False,
    }

    if request.method == 'POST':
        file = request.FILES.get('file')
        if file:
            uploaded_file = UploadedFile(file=file)
            uploaded_file.save()
            messages.success(request, 'File uploaded successfully!')
            latest_shared_images = generate_shared_images()
            local_zero_watermark_path, static_zero_watermark_path = generate_zero_watermark()

            if static_zero_watermark_path:
                # Use the static() method to get the static URL for the image
                context['zero_watermark_path'] = static(static_zero_watermark_path)
                context['show_popup'] = True

    latest_shared_images = generate_shared_images()
    context['shared1'] = latest_shared_images[0]
    context['shared2'] = latest_shared_images[1]

    if request.POST.get('ok_button'):
        # Use the static() method to get the static URL for the image
        context['show_image'] = True
        context['zero_watermark_path'] = static('images/new_zero_watermark.jpg')

    return render(request, 'upload.html', context)

