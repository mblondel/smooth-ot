.. -*- mode: rst -*-

smooth-OT
=========

Python implementation of smooth optimal transport.

What is it?
-----------

Entropic regularization of optimal transport, as popularized by [1], is quickly emerging as a new standard. 
However, other strongly convex regularizations are possible. In particular, squared L2 regularization is
interesting, since it typically results in sparse transportations plans. As shown in [2], OT with general strongly convex
regularization can be solved using alternate Bregman projection algorithms. This package follows instead 
the dual and semi-dual approaches introduced in [3], which converge much faster in practice, especially when the
regularization is low.

Supported features
------------------

* L-BFGS solver for the smoothed dual
* L-BFGS solver for the smoothed semi-dual
* Regularizations: negative entropy, squared L2 norm

Example
--------

.. code-block:: python

    import numpy as np

    from smoothot.dual_solvers import solve_dual, solve_semi_dual
    from smoothot.dual_solvers import NegEntropy, SquaredL2
    from smoothot.dual_solvers import get_plan_from_dual, get_plan_from_semi_dual

    # Generate artificial data.
    rng = np.random.RandomState(0)
    a = rng.rand(5)
    b = rng.rand(8)
    a /= np.sum(a)
    b /= np.sum(b)
    C = rng.rand(5, 8)

    # Set regularization term.
    # Can also use NegEntropy(gamma=1.0).
    regul = SquaredL2(gamma=1.0)

    # Solve smoothed dual.
    alpha, beta = solve_dual(a, b, C, regul, max_iter=1000)
    T = get_plan_from_dual(alpha, beta, C, regul)
    print("Dual")
    print("Value:", np.sum(T * C) + regul.Omega(T))
    print("Sparsity:", np.sum(T != 0) / T.size)
    print()

    # Solve smoothed semi-dual.
    alpha = solve_semi_dual(a, b, C, regul, max_iter=1000)
    T = get_plan_from_semi_dual(alpha, b, C, regul)
    print("Semi-dual")
    print("Value:", np.sum(T * C) + regul.Omega(T))
    print("Sparsity:", np.sum(T != 0) / T.size)
    print()

Installation
------------

This project can be installed from its git repository. 

1. Obtain the sources by::

    git clone https://github.com/mblondel/smooth-ot.git

or, if `git` is unavailable, `download as a ZIP from GitHub <https://github.com/mblondel/smooth-ot/archive/master.zip>`_.


2. Install the dependencies::

    # via pip

    pip install numpy scipy scikit-learn nose


    # via conda

    conda install numpy scipy scikit-learn nose


3. Install smooth-ot::

    cd smooth-ot
    sudo python setup.py install


References
----------

.. [1] Marco Cuturi.
       *Sinkhorn distances: Lightspeed computation of optimal transport.*
       In: Proc. of NIPS 2013.
       
.. [2] Arnaud Dessein, Nicolas Papadakis, Jean-Luc Rouas. 
       *Regularized optimal transport and the rot mover’s distance.* 
       arXiv preprint arXiv:1610.06447, 2016.
  
.. [3] Mathieu Blondel, Vivien Seguy, Antoine Rolet.
       *Smooth and Sparse Optimal Transport.*
       In: Proc. of AISTATS 2018.
       [`PDF <https://arxiv.org/abs/1710.06276>`_]

Author
------

- Mathieu Blondel, 2018
