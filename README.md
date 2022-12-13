# validacion-algoritmo-roturas-PDI
Repositorio proyecto ramo PDI equipo 7. Tema: validacion de algoritmos de rotura mallas salmoneras

Modulo 4-5 implementados juntos 

Del modulo 5 se cambio el numero de iteraciones de la apertura de 2 a 1 para la operacion morfologica de apertura. De esta forma se pierde menos informacion mejorando el criterio de espacio temporal implementado anteriormente,

![image](https://user-images.githubusercontent.com/67871398/207142685-9b42d144-4fee-4d50-93ce-d13d6434ceda.png)

El numero de vecinos se ajusto a 8:
![image](https://user-images.githubusercontent.com/67871398/207143186-e0f3b281-9358-475c-b00e-5c9d3a87a53b.png)

Del modulo 4 se cambiaron 2 cosas.
La primera es la multiplicacion del promedio de las areas para evaluar menos candidatos a roturas, se cambio de 3 a 6

![image](https://user-images.githubusercontent.com/67871398/207204512-2953eb27-b0ad-4227-b6ec-0468059c38e5.png)

La segunda es el peso que multiplica la el promedio de los vecinos cuando una rotura a sido confirmada, se cambio de 1.5 a 4

![image](https://user-images.githubusercontent.com/67871398/207204759-3b229bff-a241-4694-b1ad-0a3ee6a816ba.png)



Para poder ajustar la mejor configuracion se realizaron multiples pruebas cambiando los valores criticos de las variables de cada modulo para obtener el mejor resultado posible al converger los 2 modulos y obtener una mejora considerable.

Para mas informacion de las pruebas efectuadas mirar la wiki de convergencia del modulo 4-5:
https://github.com/fransa27/validacion-algoritmo-roturas-PDI/wiki/Modulo-4---5-ajustados-y-combinados

Documentacion modulo 5:
https://github.com/fransa27/validacion-algoritmo-roturas-PDI/wiki/Modulo-5:-Mejoramiento-correcci%C3%B3n-de-ruido-y-movimiento-de-la-camara

Documentacion modulo 4:
https://github.com/fransa27/validacion-algoritmo-roturas-PDI/wiki/Modulo-4:-Mejoramiento-deteccion-de-da%C3%B1o-en-la-malla
