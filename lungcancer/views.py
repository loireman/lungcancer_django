import os
from django.conf import settings
from django.http import HttpResponse
import cv2
import os
import numpy as np
import pydicom as dicom
from skimage import measure, segmentation
import scipy.ndimage as ndimage
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage

def index(request):
    class_names = [
    'Не діагностовано',  # 0
    'Діагностовано',  # 1
    ]
    
    loaded_model = load_model('models/model-1680359862.h5')

    # load weights into new model
    loaded_model.trainable = True
    print("Loaded model from disk")
    datagen = ImageDataGenerator(rescale=1. / 255)

    # Load the scans in given folder path
    def load_scan(path):
        slices = [dicom.read_file((path),force = True)]
            
        return slices

    def get_pixels_hu(scans):
        image = np.stack([s.pixel_array for s in scans])
        # Convert to int16 (from sometimes int16), 
        # should be possible as values should always be low enough (<32k)
        image = image.astype(np.int16)

        # Set outside-of-scan pixels to 0
        # The intercept is usually -1024, so air is approximately 0
        image[image == -2000] = 0
        
        # Convert to Hounsfield units (HU)
        intercept = scans[0].RescaleIntercept
        slope = scans[0].RescaleSlope
        
        if slope != 1:
            image = slope * image.astype(np.float64)
            image = image.astype(np.int16)
            
        image += np.int16(intercept)
        
        return np.array(image, dtype=np.int16)

    def generate_markers(image):
        #Creation of the internal Marker
        marker_internal = image < -400
        marker_internal = segmentation.clear_border(marker_internal)
        marker_internal_labels = measure.label(marker_internal)
        areas = [r.area for r in measure.regionprops(marker_internal_labels)]
        areas.sort()
        if len(areas) > 2:
            for region in measure.regionprops(marker_internal_labels):
                if region.area < areas[-2]:
                    for coordinates in region.coords:                
                        marker_internal_labels[coordinates[0], coordinates[1]] = 0
        marker_internal = marker_internal_labels > 0
        #Creation of the external Marker
        external_a = ndimage.binary_dilation(marker_internal, iterations=10)
        external_b = ndimage.binary_dilation(marker_internal, iterations=55)
        marker_external = external_b ^ external_a
        #Creation of the Watershed Marker matrix
        marker_watershed = np.zeros((512, 512), dtype=np.int)
        marker_watershed += marker_internal * 255
        marker_watershed += marker_external * 128
        
        return marker_internal, marker_external, marker_watershed

    def seperate_lungs(image):
        #Creation of the markers as shown above:
        marker_internal, marker_external, marker_watershed = generate_markers(image)
        
        #Creation of the Sobel-Gradient
        sobel_filtered_dx = ndimage.sobel(image, 1)
        sobel_filtered_dy = ndimage.sobel(image, 0)
        sobel_gradient = np.hypot(sobel_filtered_dx, sobel_filtered_dy)
        sobel_gradient *= 255.0 / np.max(sobel_gradient)
        
        #Watershed algorithm
        watershed = segmentation.watershed(sobel_gradient, marker_watershed)
        
        #Reducing the image created by the Watershed algorithm to its outline
        outline = ndimage.morphological_gradient(watershed, size=(3,3))
        outline = outline.astype(bool)
        
        #Performing Black-Tophat Morphology for reinclusion
        #Creation of the disk-kernel and increasing its size a bit
        blackhat_struct = [[0, 0, 1, 1, 1, 0, 0],
                        [0, 1, 1, 1, 1, 1, 0],
                        [1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1],
                        [0, 1, 1, 1, 1, 1, 0],
                        [0, 0, 1, 1, 1, 0, 0]]
        
        blackhat_struct = ndimage.iterate_structure(blackhat_struct, 8)
        #Perform the Black-Hat
        outline += ndimage.black_tophat(outline, structure=blackhat_struct)
        
        #Use the internal marker and the Outline that was just created to generate the lungfilter
        lungfilter = np.bitwise_or(marker_internal, outline)
        #Close holes in the lungfilter
        #fill_holes is not used here, since in some slices the heart would be reincluded by accident
        lungfilter = ndimage.morphology.binary_closing(lungfilter, structure=np.ones((7,7)), iterations=3)
        
        #Apply the lungfilter (note the filtered areas being assigned -2000 HU)
        segmented = np.where(lungfilter == 1, image, -2000*np.ones((512, 512)))

        #### nodule
        lung_nodule_1 = np.bitwise_or(marker_internal, image)
        lung_nodule = np.where(lungfilter == 1, lung_nodule_1, np.zeros((512, 512)))

        
        return segmented, lung_nodule, lungfilter, outline, watershed, sobel_gradient, marker_internal, marker_external, marker_watershed


    def preprocess_image(filename):
        test_patient_scans = load_scan(filename)
        test_patient_images = get_pixels_hu(test_patient_scans)
        data = []
        img = test_patient_images[0]
        seg_img = seperate_lungs(img)[0]
        sobel_gradient = seperate_lungs(img)[5]
        img_watershed = seperate_lungs(img)[8]
        new_img = cv2.resize(seg_img, (224, 224))
        new_img = np.expand_dims(new_img,axis = -1)
        data.append(new_img)

        return np.array(data), img, seg_img, sobel_gradient, img_watershed
    
    if request.method == 'POST':

            uploaded_file = request.FILES['document']
            fs = FileSystemStorage('lungcancer/upload/')
            fs.save(uploaded_file.name, uploaded_file)

            data, rawdata, segdata, sobeldata, watersheddata = preprocess_image('lungcancer/upload/' + uploaded_file.name)

            plt.imsave('lungcancer/upload/myimage1.png', rawdata, cmap='gray')
            plt.imsave('lungcancer/upload/myimage2.png', segdata, cmap='gray')
            plt.imsave('lungcancer/upload/myimage3.png', sobeldata, cmap='gray')
            plt.imsave('lungcancer/upload/myimage4.png', watersheddata, cmap='gray')

            sub_generator = datagen.flow(data, shuffle=False)
            sub_generator.reset()

            predictions = loaded_model.predict(sub_generator)
            print(predictions[0][0])
            predictions_res = predictions[0].round().astype(int)  # multiple categories

            result = class_names[predictions_res[0]]

            os.remove('lungcancer/upload/' + uploaded_file.name)

            return render(request, 'index.html', {'result': result, 'response': 1, 'percent': (predictions[0][0] * 100).round(2)})
    return render(request, 'index.html')


def serve_photo(request, filename):
    # Construct the path to the image file on disk
    file_path = os.path.join(settings.BASE_DIR, 'lungcancer', 'upload', filename)

    # Open the image file in binary mode
    with open(file_path, 'rb') as f:
        # Create an HTTP response with the contents of the image file
        response = HttpResponse(f.read(), content_type='image/png')

    # Return the HTTP response
    return response