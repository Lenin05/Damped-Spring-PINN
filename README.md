# Damped-Spring-PINN
This project demonstrates a Physics-Informed Neural Network (PINN) applied to the underdamped harmonic oscillator equation. By integrating the laws of physics into the training process, the model learns to approximate the spring’s oscillatory motion with high accuracy. The app allows users to visualize the predicted dynamics compared to the analytical solution, showcasing how machine learning can solve differential equations in physics and engineering.

# Equations
En el sistema no consideraremos una fuerza externa aplicada al sistema, el sistema mostrado es el siguiente:

![DCL](Damped_Spring\artifacts\dcl.png)

Donde: <br>
$m:$ Masa <br>
$k:$ Rigidez del resorte <br>
$c:$ Coeficiente de amortiguamiento viscoso <br>
$x(0):$ Desplazamiento inicial <br>
$v(0):$ Velocidad inicial <br>
 
La fórmula que gobierna al sistema es: <br>
$ m\ddot{x} + c\dot{x} + k = 0$

Para el caso de un sistema subamortiguado, donde: <br>
$\Delta = c^2 - 4km<0$ <br>

La solución del sistema es: <br>
$ x(t) = e^{-\zeta \omega_n t} \left( A \cos(\omega_d t) + B \sin(\omega_d t) \right)$<br>

Donde;<br>
$\omega_n = \sqrt{\frac{k}{m}}$<br>
$\zeta = \frac{c}{2m\omega_n} = \frac{c}{2 \sqrt{km}}$<br>
$\omega_d = \omega_n \sqrt{1 - \zeta^2}$<br>