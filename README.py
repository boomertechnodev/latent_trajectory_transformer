# Latent Trajectory Flow Transformer


Okay this one is pretty cool. 

Contained within is a discrete/text transformer, with a time dependent latent space suitable for planning in trajectory space. 

This is implimented as A latent ode (sde works as well). Which, is honestly pretty fasinating that this works. One of the big wins here would be the fact 
you can potentially roll out the latent trajectory without growing the context length of the backbone. 

I think this is a good candidate for continual learning and inference time planning. 

There is still some work todo on tweaking params, and I am not sure what can be ablated. But it's working as intended atleast some of the time.

```
Samples that share a Z
?B>!_______________________________________________________________
?B>_!_____________________________BBBBBBBB_________________________
?B>!____________________________BBBBBBBBB__________________________
?B>!_____________________________BBBBBBBB__________________________
?B>!___________________!_________BBBBBBBB__________________________
?B>!________BBBBBBB_____________!_________!________________________
?B>!_______________________________________!_______________________
?B>!______________________________BBBBBBBB_________________________

Samples with a random Z
?I>_!_____________________________!IIIIIIIII__!!_!_________________
?R>!___________________!_________!________!____!___________________
?E>_!EEEEEEEE___________________!__________________________________
?B>__BBBBBBBB_____________________!____!___________________________
?V>!______________________________!________!_!___!_!_______________
?Q>_!QQQQQQQQ_____________________!____!_______!!_!___!_!__________
?R>__!____RRRRRRR___________!__________!__!_!_____!__!_____________
?B>!______________BBBBBBBB_______!_!______________!________________
````
