
#ode/ode.py

import numpy as np


def funcion_ode(_t, _x):
    """En este ambiente se define la función utilizada por las métodos metodo_euler(), runge_kutta2() y runge_kutta4().
    
    Args:
        _t (double): valor del lapso temporal
        _x (double): valor de la función base de la función diferencial en el momento _t.
    
    Salida:
        double: El método devuelve el valor de la ecuación diferencial.
    
    Ejemplo:
        >>> print(funcion_ode(2, 1))
        4.0
    
    """
    
    return (lambda _t, _x: 2*_x/_t)


def metodo_euler(_a, _b, _h, _x_inicial):
    """Devuelve la solución numérica de una ecuación diferencial ordinaria utilizando el método de Euler. Para su ejecución requiere el método funcion_ode(), que calcula contiene la ecuación diferencial a analizar.

    Args:
        a (double): inicio del lapso a analizar.
        b (double): final del lapso a analizar
        h (double): esparcimiento, amplitud de los pasos
        x_inicial (double): valor inicial de x
        
    Salidas:
        El método devuelve dos vectores. 
        vector[double]: El primer vector contiene los valores de la función
        vector[double]: El segundo vector contiene los valores sobre el eje x correspondientes.
    
    Ejemplo:
        a = -1
        b = 0
        h = 0.01
        x_inicial = -1
        
        f, h = metodo_euler(a, b, h, x_inicial)
        
        >>> print(f)
        [-1.00000000e+00 -9.80000000e-01 ... -2.02020202e-04  8.80914265e-18]
        
        >>> print(h)
        [-1.   -0.99 ...  -0.02 -0.01]
        
    """

    
    #generar un arreglos que inicia en a, termina en b y que tenga esparcimiento equidistante
    t = np.arange(_a, _b, _h)
    #crear un arreglo de ceros para x de la misma longitud de t
    x = np.zeros(len(t))
    t[0] = _a
    x[0] = _x_inicial
    
    f_ode = funcion_ode(t, x)
    
    for i in range(0, len(t) - 1):
        x[i + 1] = x[i] + _h*f_ode(t[i],x[i])
    return x, t



def runge_kutta_2(_a, _b, _h, _x_inicial):
    """Devuelve la solución numérica de una ecuación diferencial ordinaria utilizando el método de Runge Kutta de segundo orden. Para su ejecución requiere el método funcion_ode(), que calcula contiene la ecuación diferencial a analizar.

    Args:
        a (double): inicio del lapso a analizar.
        b (double): final del lapso a analizar
        h (double): esparcimiento, amplitud de los pasos
        x_inicial (double): valor inicial de x
        
    Salidas:
        El método devuelve dos vectores. 
        vector[double]: El primer vector contiene los valores de la función
        vector[double]: El segundo vector contiene los valores sobre el eje x correspondientes.
    
    Ejemplo:
        a = -1
        b = 0
        h = 0.01
        x_inicial = -1
        
        f, h = runge_kutta_2(a, b, h, x_inicial)
        
        >>> print(f)
        [-1.00000000e+00 -9.80100503e-01 ... -4.34527665e-04 -1.44842555e-04]
        
        >>> print(h)
        [-1.   -0.99 ...  -0.02 -0.01]
        
    """
    
    
    #generar un arreglos que inicia en a, termina en b y que tenga esparcimiento equidistante
    t = np.arange(_a, _b, _h)
    
    #crear un arreglo de ceros para x de la misma longitud de t
    x = np.zeros(len(t))
    t[0] = _a
    x[0] = _x_inicial
    
    f_ode = funcion_ode(t, x)

    for i in range(0, len(t) - 1):
        k_1 = h * f_ode(t[i], x[i])
        k_2 = h * f_ode((t[i] + h*0.5), (x[i] + k_1*0.5))
        
        x[i + 1] = x[i] + k_2
        
    return x, t



def runge_kutta_4(_a, _b, _h, _x_inicial):
    """Devuelve la solución numérica de una ecuación diferencial ordinaria utilizando el método de Runge Kutta de cuarto orden. Para su ejecución requiere el método funcion_ode(), que calcula contiene la ecuación diferencial a analizar.

    Args:
        a (double): inicio del lapso a analizar.
        b (double): final del lapso a analizar
        h (double): esparcimiento, amplitud de los pasos
        x_inicial (double): valor inicial de x
        
    Salidas:
        El método devuelve dos vectores. 
        vector[double]: El primer vector contiene los valores de la función
        vector[double]: El segundo vector contiene los valores sobre el eje x correspondientes.
    
    
    Ejemplo:
        a = -1
        b = 0
        h = 0.01
        x_inicial = -1
        
        f, h = runge_kutta_4(a, b, h, x_inicial)
        
        >>> print(f)
        [-1.00000000e+00 -9.80100503e-01 ... -4.34527665e-04 -1.44842555e-04]
        
        >>> print(h)
        [-1.   -0.99 ...  -0.02 -0.01]
        
    """
    
    #generar un arreglos que inicia en a, termina en b y que tenga esparcimiento equidistante
    t = np.arange(_a, _b, _h)
    #crear un arreglo de ceros para x de la misma longitud de t
    x = np.zeros(len(t))
    t[0] = _a
    x[0] = _x_inicial
    
    f_ode = funcion_ode(t, x)

    for i in range(0, len(t) - 1):
        k_1 = h * f_ode(t[i], x[i])
        k_2 = h * f_ode((t[i] + h/2), (x[i] + h*k_1/2))
        k_3 = h * f_ode((t[i] + h/2), (x[i] + h*k_2/2))
        k_4 = h * f_ode((t[i] +h), (x[i] + h*k_3))
        x[i + 1] = x[i] + (k_1 + 2*K_2 + 2*k_3 +k_4)*h/6
        
    return x, t

