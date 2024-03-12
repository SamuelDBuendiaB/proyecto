import cv2
#importa libreria opencv
import numpy as np
#importa libreria numpy de opencv

def gabor_filter(image, theta=0, frequency=0.6, sigma_x=1, sigma_y=1):
    kernel = cv2.getGaborKernel((21, 21), sigma_x, sigma_y, frequency, 0.5, theta)
    filtered_image = cv2.filter2D(image, cv2.CV_8UC3, kernel)
    return filtered_image

def extract_fingerprint_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    gabor_features = []
    for theta in orientations:
        filtered_image = gabor_filter(image, theta=theta)
        gabor_features.append(filtered_image)
    return gabor_features, image

def find_minutiae(image):
    # imagen binario
    _, binarized = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Mustra la imagen luego de adelgazarla
    cv2.imshow('Thinned Image', binarized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # toma la imagen adelgazada binariamentes y la converte en el esqueleto de esta
    thinning = cv2.ximgproc.thinning(binarized)
    skeleton = thinning.copy()
    skeleton = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)

    # encuentra el contorno si cambian por RETR_External nunca muestra cosas en rojo no se porque
    contours, _ = cv2.findContours(thinning, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # dibuja el contorno
    cv2.drawContours(skeleton, contours, -1, (255, 255, 255), 1)
    
    # muestra el controno
    cv2.imshow('Contours', skeleton)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # filtro que se supone que busca los delta
    delta_minutiae = []
    for contour in contours:
        # busca en el controno de arriba linea 34
        perimeter = cv2.arcLength(contour, True)
        # para buscar en el contorno tama;a de como el perimtro entre mas le suban mas rojo les dara max 1 creo 
        approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
        if len(approx) == 3:  # Delta tiene 3 vertices
            # verifica el rango del contorno y revisa si este esta dentro del parametro
            area = cv2.contourArea(contour)
            if area >= 20 and area <= 800:  # ajusta el perimetro del area como el tama;o al que puede acceder
                # verifica que el rectangulo de la imagen sea cerca a 1
                x, y, w, h = cv2.boundingRect(contour)
                # divide el tama;o de la imagen con su altura
                aspect_ratio = float(w) / h
                if aspect_ratio >= 0.5 and aspect_ratio <= 2.5:  #ajusta el aspect ratio de la imagen 
                    # checka si el contorno esta como en el centro del area
                    M = cv2.moments(contour)
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    if cv2.pointPolygonTest(contour, (cx, cy), False) == 1:
                        delta_minutiae.append(contour)

    # colorea las minutae en rojo
    cv2.drawContours(skeleton, delta_minutiae, -1, (0, 0, 255), 1)

    return skeleton

# muestra imagenes
def display_images(images, titles):
    for i, (image, title) in enumerate(zip(images, titles)):
        cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    #ruta de la imagen
    image_path = 'imagen\img3.png'
    #manda la imagen al void de sacar el gabor filter
    gabor_features, original_image = extract_fingerprint_features(image_path)
    #manda imagen a encontrar la minutiae
    minutiae_image = find_minutiae(original_image)
    #muestra imagenes las de gabor y la minutae con rayos
    display_images(gabor_features + [minutiae_image], [f'Gabor Filtered Image {i+1}' for i in range(len(gabor_features))] + ['Minutiae'])

if __name__ == "__main__":
    #ejecuta main
    main()
