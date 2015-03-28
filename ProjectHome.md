In this repository, we made available the code produced while researching the use of genetic programming methods to model the lens distortion of cameras.

Two implementations are included. One is made in matlab, using the gptips library:
(https://sites.google.com/site/gptips4matlab/)
I had to make some "non-usual" changes to work with 2 inputs and 2 outputs. Maybe there is a better way, but I didn't find it.

The second implementation uses python with DEAP
(https://code.google.com/p/deap/)
which is absolutely wonderful. Depending on the release version, you might need the beta code from the repository for the code to work. The ppl that keep DEAP are so nice that they included some features I needed in the next version. It works out of the box with scoop to distribute the processing. This is the recommended version.

Both implementations should work, but be advised that I didn't reach better results than the plumb line model (radial, tangential, thin prism). Maybe my lens are good, maybe it needs more work, maybe I needed to let it process more. All common problems when dealing with GP.

Anyway, code is available and so am I, for any questions that might arise.

FAPESP 2012/015811-1