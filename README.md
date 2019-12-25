# gamma-log-facies-type-prediction
A continuación comparto mi enfoque que usé en el concurso: **Gamma log facies type prediction**. Este enfoque dío como resultado el score de 0.96892 (puesto 18)

* Los códigos se encuentran en la carpeta **Codigos**
* Los Datos se pueden descargar desde: https://www.crowdanalytix.com/contests/gamma-log-facies-type-prediction

Mi enfoque consiste de una red neuronal con 3 capas bidireccionales LSTM y el uso de TimeDistributed

La librería que he usado es keras

Dentro de la carpeta **Códigos**:

* **My_solution_no_CV.ipynb** está el análisis general
* **01_tratamiento_cv.py** y **01_tratamiento_cv_predict** es el código final que use para obtener el puesto 18.



