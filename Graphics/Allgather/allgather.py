import matplotlib.pyplot as plt
import numpy as np

#v1 = [0.0052 , 0.0052 , 0.0051 , 0.0051 , 0.0051 , 0.0051 , 0.0051 , 0.0051 , 0.0051 , 0.0051 , 0.0051 , 0.0051 , 0.0051 , 0.0051 , 0.0051 , 0.0076 , 0.0052]
v1 = [0.48233 , 0.48307 , 0.48257 , 0.48095 , 0.48037 , 0.48097 , 0.48136 , 0.48131 , 0.48277 , 0.48128 , 0.48136 , 0.48112 , 0.48061 , 0.48107 , 0.48144 , 0.48117 , 0.48334]
v2 = [0.49433 , 0.49417 , 0.49357 , 0.49395 , 0.49679 , 0.49097 , 0.49114 , 0.49125 , 0.49471 , 0.50121 , 0.50216 , 0.50241 , 0.50332 , 0.50429 , 0.51164 , 0.51571 , 0.51635]
n  = [1 ,2 ,4 ,8 ,16 ,32 ,64 ,128 ,256 ,512 ,1024 ,2048 ,4096 ,8192 ,16384 ,32768 ,65536]

plt.plot(n, v1, color="red", label="Wrapper_Hy_Allgather")
plt.plot(n, v2, color="green", label="MPI_Allgather")
plt.xlabel("Tamaño del mensaje")
plt.ylabel("Tiempo")
plt.legend()
#plt.title("PRUEBAS CON LA MEMORIA CACHE - BUCLES FOR ANIDADOS")
plt.show()