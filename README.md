# mnist
Deep learning concepts tried out on mnist

# GAN
Simple GAN architecture in gan.py. Generator network has two layers with leaky Relu activation with hidden layer having 128 units. 
Optimizer used is Adam. Discriminator and generator trained alternatively for 0-8. '9' proves to be a harder one to learn so discriminator to generator training steps is 10:1.

## Results for gan - 

![01](samples/0/9.png) ![01](samples/0/10.png) ![01](samples/0/11.png) ![01](samples/0/12.png)
![01](samples/1/8.png) ![01](samples/1/10.png) ![01](samples/1/11.png) ![01](samples/1/12.png)

![01](samples/2/7.png) ![01](samples/2/10.png) ![01](samples/2/11.png) ![01](samples/2/12.png)
![01](samples/3/10.png) ![01](samples/3/11.png) ![01](samples/3/12.png) ![01](samples/3/13.png)

![01](samples/4/10.png) ![01](samples/4/11.png) ![01](samples/4/12.png) ![01](samples/4/13.png)
![01](samples/5/10.png) ![01](samples/5/11.png) ![01](samples/5/12.png) ![01](samples/5/7.png)

![01](samples/6/10.png) ![01](samples/6/11.png) ![01](samples/6/12.png) ![01](samples/6/13.png)
![01](samples/7/10.png) ![01](samples/7/11.png) ![01](samples/7/12.png) ![01](samples/7/13.png)

![01](samples/8/10.png) ![01](samples/8/11.png) ![01](samples/8/12.png) ![01](samples/8/13.png)
![01](samples/9/45.png) ![01](samples/9/46.png) ![01](samples/9/47.png) ![01](samples/9/48.png)

## What worked
Wasserstein loss works well without label flipping (D(x) = 1 if z is a real sample and D(x) = 0 if z is a fake sample).
Although the training could be unstable. Adding noise to generated sample before feeding to generated results in better convergence. Although with this architecture y = D(x), y is not bounded between 0 to 1 due to no sigmoid layer at the end. This leads to better convergence, but at loss of some interpretability.

## What doesn't work
Adaptive k (number of steps discriminator is trained before generator gets updated). Doesn't really solve the mode collapse problem if the architecture is bad.
