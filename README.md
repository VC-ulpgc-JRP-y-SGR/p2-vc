## Práctica 2. Funciones básicas de OpenCV.

## Tarea 1
#### Realiza la cuenta de píxeles blancos por filas, determina el máximo para filas y columnas (uno para cada) y muestra el número de valores que superan en cada caso 0.95*máximo.


##### Método 1: Rotar la imagen

Una de las formas más faciles de realizar esta tarea es mantener todo el código del reduce tal como esta pero rotando la imagen 90 grados de forma que al aplicar la transformación el histograma resultante sea horizontal.

Para esto se puede usar la función rotate de opencv pasandole como segundo parámetro ROTATE_90_COUNERCLOCKWISE.

```python
image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
col_counts = cv2.reduce(image, 0, cv2.REDUCE_SUM, dtype=cv2.CV_32SC1)

#Normaliza en base al número de filas y al valor máximo del píxel (255)
#El resultado será el número de píxeles blancos por columna
cols = col_counts[0] / (255 * image.shape[1])

#Muestra dicha cuenta gráficamente
plt.figure()
plt.subplot(1, 2, 1)
plt.axis("off")
plt.title("Imagen")
plt.imshow(image, cmap='gray') 

plt.subplot(1, 2, 2)
plt.title("Histograma vertical")
plt.xlabel("Columnas")
plt.ylabel("% píxeles")
plt.plot(cols)
#Rango en x definido por las columnas
plt.xlim([0, image.shape[1]])
```

Un problema de este acercamiento es que cada vez que ejecutemos el código al no ser que volvamos a cargar la imagen de nuevo se rotara todo el rato 90 grados.

##### Método 2: Modificar los parámetros de reduce


## Tarea 2
#### Elige otra imagen, muestra el contenido de alguna de las imágenes resultado de Sobel antes y después de ajustar la escala

Para la imagen de este ejercicio se ha escogido una imagen de dos mariposas que hemos guardado como '*mariposa.jpg*'. Como se nos pide que mostremos el resultado de esta imagen de aplicarle Sobel antes y después del escalado, lo primero que hacemos es hacer una conversión a grises de la imagen original que se encuentra en formate BGR.

```python
# Lee imagen de archivo
img_m = cv2.imread('mariposa.jpg') 
# Conversión a grises de la original en BGR
gris_m = cv2.cvtColor(img_m, cv2.COLOR_BGR2GRAY)
```

Tras esto, se pasa a hacer un suavizado de la imagen haciendo uso de un filtro Gaussiano de 3x3. 

La aplicación de este filtro es una técnica común en el procesamiento de imágenes ya que reduce el ruido de las imágenes así como los detalles innecesarios de estas (cambios de intesidad que no interesan, etc). Además, proporciona más estabilidad a la hora de hacer el cálculo del gradiente, lo que ayudará a que Sobel consiga una detección de bordes más robusta y precisa.

```python
# Gaussiana para suavizar la imagen original
ggris_m = cv2.GaussianBlur(gris_m, (3, 3), 0)
```

Ahora sí, se procede a aplicar Sobel en direcciones Horizontales y Verticales por separado a partir de la imagen resultado de aplicar el filtro Gaussiano y se combinan en una misma imagen.

```python
# Calcula en ambas direcciones (horizontal y vertical)
sobelx_m = cv2.Sobel(ggris_m, cv2.CV_64F, 1, 0)  # x
sobely_m = cv2.Sobel(ggris_m, cv2.CV_64F, 0, 1)  # y
# Combina ambos resultados
sobel_m = cv2.add(sobelx_m, sobely_m)
```

Como último paso, mostramos tanto el resultado sin escalar y el resultado tras aplicarle un ajuste en la escala de grises de la imagen. Esto garantiza que todos los valores estén dentro del rango de 0 a 255, lo que es más adecuado para la visualización. Al escalar los valores, los detalles se vuelven más visibles porque los valores se distribuyen en un rango que es fácilmente interpretable para el ojo humano.

```python
plt.figure()
plt.subplot(1, 2, 1)
plt.axis("off")
plt.title('Combinada Escalada')
# Para visualizar convierte a escala manejable en una imagen de grises
plt.imshow(cv2.convertScaleAbs(sobel_m), cmap='gray') 

plt.subplot(1, 2, 2)
plt.axis("off")
plt.title('Combinada Sin Escalar')
plt.imshow(sobel_m, cmap='gray') # Sin convertir escala
plt.show()
```

## Tarea 3
####  Aplica umbralizado a la imagen resultante de Sobel (valores 0 a 255 y convertida a 8 bits por ejemplo sobel8 = np.uint8(sobel)), y posteriormente realiza el conteo por filas y columnas similar al realizado en el ejemplo con la salida de Canny. Calcula los máximos por filas y columnas, y determina las filas y columnas por encima del 0.95*máximo. Remarca con alguna primitiva gráfica dichas filas y columnas sobre la imagen ¿Cómo se comparan los resultados obtenidos a partir de Sobel y Canny?

##### Sobel:

En esta tarea lo primero que tenemos que hacer es aplicar Sobel tanto en las horizontales como en las verticales y sumar ambos tensores resultantes.

```python
image = cv2.imread('./mariposa.jpg')
image = cv2.cvtColor(img_m, cv2.COLOR_BGR2GRAY)

# Smooth the image
smoothed_image = cv2.GaussianBlur(image, (5, 5), 0)

# Apply Sobel edge detection
sobel_x = cv2.Sobel(smoothed_image, cv2.CV_64F, 1, 0)
sobel_y = cv2.Sobel(smoothed_image, cv2.CV_64F, 0, 1)

# Calculate the magnitude of the gradient
sobel_magnitude = cv2.add(sobel_x, sobel_y)
```

Después aplicaremos el threshold sobre la imagen creada por el cv2.add y la transformaremos a el formato uint8.

```python
_, thresholded_image = cv2.threshold(sobel_magnitude, threshold_value, 255, cv2.THRESH_BINARY)
thresholded_image = np.uint8(thresholded_image)
```

A continuación mostraremos por pantalla las aplicaciones en cada eje y el resultado final.

```python
plt.title('Resultado de Sobel')
plt.imshow(thresholded_image, cmap='gray')
plt.show()

plt.figure()
plt.subplot(1, 3, 1)
plt.axis("off")
plt.title('Verticales')
plt.imshow(sobel_x, cmap='gray')

plt.subplot(1, 3, 2)
plt.axis("off")
plt.title('Horizontales')
plt.imshow(sobel_y, cmap='gray')

plt.subplot(1, 3, 3)
plt.axis("off")
plt.title('Combinado')
plt.imshow(thresholded_image, cmap='gray')
plt.show()
```

Ahora lo que tenemos que hacer es sacar aquellas columnas y filas mayores al 0.95 del máximo pico en los mismos ejes. Para ello primero averiguaremos el máximo y lo múltiplicaremos por esta cifra. Luego con la función np.where encontraremos aquellos puntos mayores que ese valor.

```python
# Find the maximum values in rows and columns
max_row_value = np.max(row_sums)
max_col_value = np.max(col_sums)

# Determine rows and columns above 95% of the maximum
row_threshold = 0.95 * max_row_value
col_threshold = 0.95 * max_col_value

# Find rows and columns that meet the threshold
highlighted_rows = np.where(row_sums >= row_threshold)
highlighted_cols = np.where(col_sums >= col_threshold)
```

Por último pintaremos rectas horizontales y verticales en cada uno de esos puntos sobre la imagen horizontal y la mostraremos por pantalla.

```python
plt.show()

painted_image = image.copy()

for row in highlighted_rows[0]:
    cv2.line(painted_image, (0, row), (image.shape[1], row), (0, 0, 255), 2)

for col in highlighted_cols[0]:
    cv2.line(painted_image, (col, 0), (col, image.shape[0]), (0, 0, 255), 2)

# Display the image with highlighted rows and columns
plt.title('Columnas y filas por encima de 0.95 del máximo')
plt.imshow(cv2.cvtColor(painted_image, cv2.COLOR_BGR2RGB))
plt.show()


show_histogram_x(thresholded_image)
show_histogram_y(thresholded_image)
```

También mostramos los histogramas para comprobar si los picos del mismo coinciden con la posición de estas rectas.

##### Canny:

En el caso de Canny hacemos exactamente el mismo procedimiento lo único que cambia es la primera parte, es decir, la aplicación del filtro.

```python
# Apply Sobel edge detection
sobel_x = cv2.Canny(smoothed_image, 100, 200)
sobel_y = cv2.Canny(smoothed_image, 100, 200)
```


##### Conclusiones

Ambos resultados obtenidos son diferentes aunque es cierto que en la detección de ciertas partes como el ala de la mariposa ambos coinciden, quizá se deba a que este es uno de los bordes mas pronunciados de toda la imagen.

## Tarea 4:
#### Asumiendo que quieren mostrar a personas que no forman parte del curso de VC el comportamiento de una o varias fuciones de las vistas hasta este momento aplicadas sobre la entrada de la webcam. ¿Cuál(es) escogerían?

Se ha escogido usar el filtro de Sobel para detectar los bordes de las imágenes. Para mostrar su funcionamiento, se ha decidido capturar las imágenes que muestra la cámara y aplicar en cada frame el filtro, aplicando un valor de umbral aleatorio, de forma que parece da una sensación de "dibujo" en el filtro.

Empezamos capturando las imágenes de la cámara y leyéndolas.

```python
import random 

cam = cv2.VideoCapture(0)

while True:
    ret, image = cam.read()
```

Como vamos a usar Sobel con el método Canny, aplicamos a la imagen capturada un filtro Gaussiano para suavizar la imagen y, con un número aleatorio de umbral entre 30 y 50, aplicamos el método de Canny para detectar los bordes y lo combinamos en una misma imagen, convirtiéndo la imagen de magnitud del gradiente a un formato de 8 bits.

```python
    smoothed_image = cv2.GaussianBlur(image, (5, 5), 0)

    r = random.randint(30, 50)
    # Apply Sobel edge detection
    sobel_x = cv2.Canny(smoothed_image, 0, r)
    sobel_y = cv2.Canny(smoothed_image, 0, r)

    # Calculate the magnitude of the gradient
    sobel_magnitude = cv2.add(sobel_x, sobel_y)

    # Convert the magnitude image to 8-bit
    sobel_magnitude = np.uint8(sobel_magnitude)
```

Tras esto, se aplica un umbral a la imagen de magnitud del gradiente para obtener una imagen binaria, donde los píxeles por encima del umbral se establecen a 255 (blanco) y los píxeles por debajo del umbral se establecen a 0 (negro).

```python
    # Define a threshold value
    threshold_value = 0

    # Threshold the Sobel magnitude image
    _, thresholded_image = cv2.threshold(sobel_magnitude, threshold_value, 255, cv2.THRESH_BINARY)
```

Por último, mostramos la imagen.

```python
    cv2.imshow("SKETCHY", thresholded_image)

    if cv2.waitKey(20) == 27:
        break

cam.release()
cv2.destroyAllWindows()
```

## Tarea 5:
#### Tras ver los vídeos [My little piece of privacy](https://www.niklasroy.com/project/88/my-little-piece-of-privacy), [Messa di voce](https://youtu.be/GfoqiyB1ndE?feature=shared) y [Virtual air guitar](https://youtu.be/FIAmyoEpV5c?feature=shared) propongan (los componentes de cada grupo) una reinterpretación del procesamiento de imágenes con las técnicas vistas o que conozcan.


En esta tarea hemos decidido crear tracker de pelotas de tennis y cubos de rubik. Esta idea ha sido inspirada en My little piece of privacy. 

Nos parece bastante interesante la propuesta pues el tracking de determinado tipo de objetos a través de visión por computador puede ser de extrema utilidad en el dominio de varios negocios, como por ejemplo, en deportes como podría ser el tennis.

Para conseguir este objetivo previamente hemos sacado varias fotos de la pelota de tennis con sombra, sin sombra y intentado extraer la gama de colores posibles que esta pueda llegar a tener. El formato usado para esto ha sido el rgb y hemos usado un color picker para poder averiguar estos colores.

Una vez realizado el estudio previo hemos usado la función inRange para obtener la mascara en ese rango de colores y después la hemos aplicado a la imagen horizontal con un bitwise_and

```python
ret, image = cam.read()

color_pelota_0 = np.array([40, 95, 110])
color_pelota_1 = np.array([60, 255, 255])
# Create a mask that selects pixels within the specified color range


mask = cv2.inRange(image, color_pelota_0, color_pelota_1)

# Use the mask to extract the pixels within the color range
result = cv2.bitwise_and(image, image, mask=mask)
```

Lo siguiente que hemos echo es al igual que en las anteriores prácticas extraer el 0.95 del maximo valor de las columnas y filas y sacar las correspondientes columnas y filas por encima de ese valor.

```python
# Determine rows and columns above 95% of the maximum
row_threshold = 0.95 * max_row_value
col_threshold = 0.95 * max_col_value

highlighted_rows = np.where(row_sums >= row_threshold)
highlighted_cols = np.where(col_sums >= col_threshold)
```

Para asegurar que solo trackee en caso de haber el objeto determinado y no algún otro dentro de ese rango de color hemos puesto una condición que al no ser que el valro máximo supere un umbral no vamos a trackear el objeto. Después hemos buscado el máximo en el array resultante del where y por último agregamos la posición dada por los máximos a una lista de puntos.

```python
if(max_row_value > 6000 and max_col_value > 6000):
    # Highlight rows and columns on the image
    position_x = max(highlighted_rows[0])
    position_y = max(highlighted_cols[0])

    center = (position_y, position_x)
    position_stack.append(center)

    # Define the color of the circle in BGR format (blue in this example)
    color = (255, 0, 0)  # (Blue, Green, Red)
```

Por último pintaremos cada uno de estos puntos como una linea interconecctada usando la fución cv2 line durante el recorrido de toda la lista de puntos.

```python
if len(position_stack) > 2:
    # Draw lines connecting interpolated points
    for i in range(0, len(position_stack) - 1):
        cv2.line(image, position_stack[i], position_stack[i+1], (0, 0, 255), thickness=5)
```