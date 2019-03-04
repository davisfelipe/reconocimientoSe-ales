# <center>Señales de Transito<center>
| Nombres | Codigos | Correo |
|:------------------------------:|:-----------:|:----------------------------------:|
| David Felipe Peña Huertas | 20132020018 | dfpenah@correo.udistrital.edu.co |
| Anthony Jason Vargas Sepulveda | 20132020115 | ajvargass@correo.udistrital.edu.co |
    
## Resumen
En estes codigo se mostrara el reconocimiento de señales de transito a partir del stream por camara, para la selección de las imagenes se tomaron las siguientes:
* Limite de Velocidad
* Paradero SITP
* Pare
* Prohibido Parquear
Posteriormente a partir de imagenes tomadas y redimensionadas a 128x128 y 64x64 se desarrollaro 3 modelos para la prediccion de dichas imagenes los cuales se tomar asi:
* Modelo solo a partir de la dimension y el color de la imagen
* Modelo a partir del ancho largo y canal
* Modelo a partir de promedio de diferentes modelos entrenados
