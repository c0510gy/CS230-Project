# CS230-Project

This repository contains implementation of our web app application for graph layout optimization using proposed distributed readability measuring algorithms.

* Project Title: Large Graph Layout's Readability Evaluation on Distributed Environment
* Team members (Team 9):
  * Sanggeon Yun (sanggeoy@uci.edu)
  * SungHeon Jung (sungheoj@uci.edu)
  * Ryozo Masukawa (rmasukaw@uci.edu)

## Project Overview

Applying readability metrics is one of the most common ways to evaluate and optimize graph layouts. The metrics quantify the aesthetic quality of the layouts, i.e., how easy the layout is to be visually understandable or readable. Measuring readability metrics is useful in comparing a number of different layouts generated with different parameters or determining whether the layout has converged or not during layout generation. However, graph readability metrics often suffer from poor scalability; for example, node occlusion and edge crossing metrics have a computational complexity of $O(N^2)$, making them hardly applicable to large graphs. Even though layout generation algorithms are becoming faster using GPU acceleration or distributed environments, the lack of speed hinders further applicability, especially for large-scale graphs. In this research, we propose distributed algorithms for readability metrics that enable scalable evaluation of graph layouts. We first introduce exact algorithms which parallelize the computation in distributed machines, and also propose their enhanced algorithms where we reduce the computations through grid-based approximation.

## Deliverables

1.	A CLI or web interface-based application for users to manage graph layout optimization built using our proposed graph readability evaluation algorithms.
2.	Scalable graph layout readability evaluation algorithms.
3.	Evaluations of our algorithms in terms of running time, accuracy compared to the single-machine algorithms, and scalability analysis.

## Running the application

### Requirements

* JAVA 8
* Python 3.8.10
* PySpark 3.1.1
* GraphFrames 0.8.2
* bayesian-optimization 1.4.3
* Django 4.2.11 (For web app)

### Running in CLI

By running the following command line, you can run layout optimization using Bayesian Optimization:

```
cd SparkApp
spark-submit --packages graphframes:graphframes:0.8.2-spark3.1-s_2.12 --repositories https://repos.spark-packages.org opt.py
```

### Running in Web Interface

By running the following command line, you can run web application:

```
cd WebApp
python3 manage.py runserver
```

## Demonstration

[![DemoVideo](https://img.youtube.com/vi/2j242gQATyU/0.jpg)](https://www.youtube.com/watch?v=2j242gQATyU)

## Conclusion

The absence of scalable and accurate evaluation algorithms reduces our capability to handle and analyze large graphs and their layout. We propose two types of scalable readability evaluation algorithms—exact and novel enhanced algorithms—in distributed environment to alleviate such an issue. Our experiments showed that our approach has a strong advantage in terms of running time, accuracy, and scalability on large-scale graphs compared to a single-machine algorithm. We also demonstrated high applicability by presenting an application on layout optimization. 
