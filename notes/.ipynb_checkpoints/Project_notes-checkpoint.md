<center>
    
# Semester Project Notes 
*autor*
## Alejandro Noguerón
*alejandro.nogueronaramburu@epfl.ch*
*supervised by*:
## Dr. Pol Del Aguila Pla,
## Prof. Michael Ünser
</center>
---

## Index
1. [Introduction to the Project]
2. [Theoretical Notes]
3. [Code Notes]
4. [To Do]
5. [Doubts]

## <a name="1.Intro"></a> 1. Introduction to the Project

This section is meant to be a compendium of theoretical bases of the proect, paired with a description of it. 

An imaging problem takes the following form:
$$y[m, n] = (h[m', n']\circledast x[m', n'])[m, n] + w[m, n]$$
where $y$ corresponds to the measurements, $h$ corresponds to an operator (e.g. the Point Spread Function in a $2D$ imaging problem) and *w* is simply noise. In this case, we have complete access to $y$, we should know the operator $h$ and, $w$ the white noise, can be approximated, usually with a Gaussian.  

#### Proximal Operator
$$prox_f(v) = argmin_x(f(x)+\frac{1}{2}||x - v||_2^2)$$

## <a name="5.Doubts"></a> 4. To do 

## <a name="5.Doubts"></a> 5. Doubts for Pol


#### <a name="week1_doubts"></a> Week 1
1. Is min squares always the origonal assumption? Yes, and no. It is always part of the $obj$ because in the end we want some similarity of the ground truth + perturbation. On the other hand, nonnegative constraints (or indicator functions) do not perturbe convexity since they are valid in all the domain (but evaluate to infinity). Are all regularizers convex? In the sense that they are norms, they are.  
2. How do we choose $x^k$? - not answered, random? Related to regularizers.
3. What is a solver, an algorithm? In short, yes
4. What the hell is reduced splitting? (in project description) --> Operator splitting es dividir el número de variables (añadir variables) para hacer un problema más sencillo. Operator splitting es lo mismo que decir proximal optimization?
5. When a $\nabla f$ is Lipschitz constant with value $L?
6. what kind of algorithm will we use?


*There is a nice interpretation on page 125*

Ill posed problem: 
1. No existe una solución (e.g., deconvolution con algo que incluya ruido Gaussiano, imposible encontrar un operador H)

Intro al Proyecto, 
$D = I$

Scope del proycto:

Pro si proxf(x) = prox()prox(delta(x))


No vale la pena leer paper de Pol y Jaldén buscar prox l1. 
* Leer proximal l1
* Entender por que delta (non negativity es la projection)
* Hacer un setup9
