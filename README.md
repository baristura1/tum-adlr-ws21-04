<div align="center">
<h1>Harnessing Reinforcement Learning for Neural Motion Planning</h1>
<h3> <i>Marc Hauck, Baris Tura</i></h3>
 <h4> <i>Technical University of Munich</i></h4>
 

</div>


<div align="justify">
This is the repository of the project "Harnessing Reinforcement Learning for Neural Motion Planning" for the course Advanced Deep Learning in Robotics. In this work, RL is employed to help a mobile or planar joint robot navigate through an unseen environment around the obstacles within. We make use of different environment representation options and runtime options, the list of which is accessible from the parser script.   
</div>

--------

### Models

models.py/MobileFinder -- Mobile robot

models.py/PlanarFinder -- Planar joint robot with arbitrary DoF

models.py/PlanarFinderImage -- 3 DoF planar joint robot with raw images as environment representation

models.py/PlanarFinderImage2 -- 2 DoF planar joint robot with raw images as environment representation


### Acknowledgments
 
 This work was inspired by [Harnessing Reinforcement Learning for Neural Motion Planning](https://arxiv.org/pdf/1906.00214.pdf).