Propuesta de proyecto en visión artificial

  ------------------------------------------------------------------------
  Título               Desarrollo de un modelo de deep learning para el
                       conteo y detección de animales en manadas densas a
                       partir de imágenes aéreas
  -------------------- ---------------------------------------------------
  Organización/Grupo   Proyecto Guacamaya[^1], CINFONIA
  de investigación     

  Experto              Isai Daniel Chacon Silva. Magister en Ingeniería
                       Biomédica, Universidad de los Andes. Investigador,
                       Laboratorio de Microsoft AI for Good.
  ------------------------------------------------------------------------

**1. Descripción.**

Este proyecto se enmarca en el área de biodiversidad y conservación del
medio ambiente, específicamente en la gestión de conflictos entre la
vida silvestre y el ganado en áreas protegidas de África subsahariana.
El aumento acelerado de la población humana en esta región ha generado
una expansión paralela en la cantidad de ganado, lo cual trae como
consecuencia conflictos por el uso de recursos como pastizales y fuentes
de agua. Estos conflictos impactan negativamente tanto en la
conservación de la biodiversidad como en la sostenibilidad de las
comunidades rurales.

La conservación de la biodiversidad en África, además de preservar
ecosistemas únicos, es clave para mitigar el cambio climático y promover
la sostenibilidad económica de las comunidades locales. En este
contexto, la capacidad de monitorear con precisión la densidad del
ganado y la fauna silvestre es fundamental para alcanzar un equilibrio
entre los objetivos de conservación y los intereses de las comunidades
locales.

El conteo manual de ganado y fauna silvestre en imágenes aéreas
representa un reto significativo. Métodos tradicionales, como la
observación desde aeronaves o el conteo visual de fotografías, son
propensos a errores, especialmente en situaciones de manadas densas, y
consumen una gran cantidad de tiempo y recursos. Las arquitecturas
tradicionales de redes neuronales convolucionales (CNN), si bien han
mostrado potencial, presentan limitaciones para contar animales en
manadas densas debido a factores como la oclusión mutua, los fondos
complejos, las variaciones de escala y la distribución no uniforme de
los individuos. Esto genera imprecisiones en la detección y el conteo.
La falta de herramientas automáticas y confiables impide el monitoreo
frecuente y eficiente, lo que limita la capacidad de tomar decisiones
informadas sobre la gestión de recursos.

El impacto del proyecto será multidimensional:

• Ambiental: Mejora de la conservación de ecosistemas al gestionar de
manera sostenible los conflictos entre ganado y fauna silvestre.

• Social: Apoyo a las comunidades rurales mediante un mejor manejo de
sus recursos, minimizando los conflictos.

• Tecnológico: Avance en el desarrollo de redes neuronales adaptadas al
conteo denso en imágenes aéreas, con aplicaciones que se pueden
generalizar más allá del contexto inicial.

El objetivo principal es diseñar un modelo basado en aprendizaje
profundo que permita detectar y contar de manera precisa y eficiente
animales en imágenes aéreas de manadas densas, considerando los desafíos
inherentes al problema (oclusiones, fondos complejos, variaciones de
escala). Esto sentará las bases para un monitoreo más eficiente y una
gestión sostenible de la biodiversidad y los recursos naturales.

**2. Conjunto de datos (*dataset*).**

![](media/image1.png){width="4.534027777777778in"
height="3.020138888888889in"}Los datos disponibles para este proyecto
son públicos y accesibles a través del enlace:
https://dataverse.uliege.be/file.xhtml?fileId=11098&version=1.0. Esta
base de datos está diseñada específicamente para tareas de conteo de
animales en manadas densas utilizando imágenes aéreas y ya se encuentra
organizada en tres conjuntos: entrenamiento (*train*), validación
(*val*) y prueba (*test*). A continuación, se muestran algunas imágenes
de este *dataset*:

![](media/image2.png){width="4.479166666666667in"
height="2.9895833333333335in"}

Los datos están bajo la licencia CC BY-NC-SA 4.0
(Attribution-NonCommercial-ShareAlike 4.0 International). Esto implica
las siguientes condiciones:

- Atribución. Se debe dar el crédito adecuado al autor original, incluir
  un enlace a la licencia y señalar si se realizaron modificaciones.

- No Comercial. Los datos no pueden ser utilizados con fines
  comerciales.

- Compartir Igual. Si los estudiantes generan nuevos trabajos derivados,
  estos deben ser distribuidos bajo la misma licencia.

- No contiene información sensible o confidencial. Los datos son
  completamente públicos y adecuados para el uso académico y de
  investigación.

**3. Validación de resultados.**

Se evaluará la capacidad del modelo para detectar y contar con precisión
el número total de animales en cada imagen, independientemente de la
especie, así como la capacidad del modelo para clasificar correctamente
a los animales detectados entre las especies disponibles: búfalo,
elefante, kob, topi, facocero (warthog) y waterbuck. Para este
propósito, se utilizarán métricas clave que permitan medir su
efectividad tanto en el conteo de animales como en la clasificación de
especies. Los criterios específicos incluyen Precisión, Recall,
F1-Score, MAE, RMSE, AC (Average Confusion). Además, se utilizarán los
resultados de HerdNet[^2] como punto de comparación para evaluar el
desempeño del modelo desarrollado por el grupo. HerdNet, una
implementación previa de referencia, obtuvo los siguientes resultados:

  -------------------------------------------------------------------------
    F1-Score (%)         MAE             RMSE           AC (*Confusión
                                                        Promedio*) (%)
  ---------------- ---------------- --------------- -----------------------
        83.5             1.9              3.6                 7.8

  -------------------------------------------------------------------------

**5. Material de apoyo**

- From crowd to herd counting: How to precisely detect and count African
  mammals using aerial imagery and deep learning?
  https://www.sciencedirect.com/science/article/pii/S092427162300031X?via%3Dihub

- Faster R-CNN: Towards Real-Time Object Detection with Region Proposal
  Networks.
  https://proceedings.neurips.cc/paper/2015/hash/14bfa6bb14875e45bba028a21ed38046-Abstract.html

- Deep Layer Aggregation:
  <https://openaccess.thecvf.com/content_cvpr_2018/html/Yu_Deep_Layer_Aggregation_CVPR_2018_paper.html>

También se pueden revisar el siguiente repositorio en Github:

https://github.com/Alexandre-Delplanque/HerdNet.git

Además, HerdNet tiene una implementación para hacer inferencia en el
siguiente repositorio:

https://github.com/microsoft/CameraTraps/blob/main/demo/image_detection_demo_herdnet.ipynb

[^1]: https://www.uniandes.edu.co/es/noticias/ingenieria/inteligencia-artificial-para-la-proteccion-del-amazonas

[^2]: https://orbi.uliege.be/bitstream/2268/304167/1/HerdNet\_%20A%20new%20CNN%20for%20accurate%20counting%20of%20African%20mammal%20herds%20in%20aerial%20imagery.pdf
