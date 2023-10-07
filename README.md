## Práctica 2. Funciones básicas de OpenCV.

## Tarea 1
#### Realiza la cuenta de píxeles blancos por filas, determina el máximo para filas y columnas (uno para cada) y muestra el número de valores que superan en cada caso 0.95*máximo.


##### Método 1: Rotar la imagen


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

##### Canny:

##### Sobel:

##### Conclusiones


## Tarea 4:
#### Asumiendo que quieren mostrar a personas que no forman parte del curso de VC el comportamiento de una o varias fuciones de las vistas hasta este momento aplicadas sobre la entrada de la webcam. ¿Cuál(es) escogerían?



## Tarea 5:
#### Tras ver los vídeos [My little piece of privacy](https://www.niklasroy.com/project/88/my-little-piece-of-privacy), [Messa di voce](https://youtu.be/GfoqiyB1ndE?feature=shared) y [Virtual air guitar](https://youtu.be/FIAmyoEpV5c?feature=shared) propongan (los componentes de cada grupo) una reinterpretación del procesamiento de imágenes con las técnicas vistas o que conozcan.

