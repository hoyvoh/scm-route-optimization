# Supply Chain Route Optimization Service

## Abstract


The Supply Chain Route Optimization Service is a backend module designed to solve a variation of the Vehicle Routing Problem (VRP) for supply chain management. This service optimizes procurement and delivery plans by minimizing costs while meeting demand and deadlines. It supports inquiries for goods procurement, checks supplier and vehicle availability, and calculates optimal delivery routes using data collected from suppliers, vehicles, and parking areas. The service leverages the GoMaps API for route mapping and integrates optimization algorithms such as Linear Programming, Genetic Algorithms, and Graph Convolutional Networks to generate cost-effective delivery solutions.

Key features include an API-based interface for seamless integration, robust data collection and preprocessing mechanisms, and tools for constructing realistic logistics networks. The optimization process accounts for Vietnam-specific trucking costs, including labor and fuel expenses, while excluding fixed costs for simplification. 

For further details, refer to the (Project Report)[https://docs.google.com/document/d/1CVOrCFgL8f4RweLlvXj8RGPTKg59RrnCVaBfGjlTdNg/edit?tab=t.0]

## Running the Service
To set up and run the service:

Ensure Docker and Docker Compose are installed on your system.
### Clone the repository:

``` bash
git clone https://github.com/hoyvoh/scm-route-optimization.git  
```
  
### Build and run the service using Docker Compose:
``` bash
docker-compose up --build
```

The service will be accessible at http://localhost:8000 as specified in the docker-compose.yml file.
