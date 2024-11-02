from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

import os
import sys
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path))
if project_root not in sys.path:
    sys.path.append(project_root)

import genetic_algorithm
from routes_selection import route_selection
from data_collection import suppliers, parkings, vehicles, destination_v2
import graph_neural_network
import linear_algorithm
from routes_selection import hard_route

def check_dest(destination):
    if destination['address'] in destination_v2['address'].to_list():
        # return selected routes
        return hard_route[destination['address']]
    else:
        print("Route selection process started for a new destination...\n\nIt will takes up to 10 minutes to compute.\n\n")
        return route_selection(suppliers=suppliers[:30], parking_areas=parkings[:10], destination=destination, vehicles=vehicles[:5])

class Plan(BaseModel):
    plan_id: str
    destination: str
    deadline: str
    demand: float
    priority: int

origins = [
    "http://localhost:8501",
]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/optimize-plan/")
async def root(plan:Plan):
    destination = {
        'id':plan.plan_id,
        'destination':plan.destination
    }

    all_possible_routes = check_dest(destination=destination)
    
    plan1 = genetic_algorithm.optimize(routes_data=all_possible_routes, 
                                      demand=plan.demand,
                                      deadline=plan.deadline)
    plan3 = linear_algorithm.optimize(routes_data=all_possible_routes, 
                                      demand=plan.demand,
                                      deadline=plan.deadline)
    plan2 = graph_neural_network.optimize(routes_data=all_possible_routes, 
                                      demand=plan.demand,
                                      deadline=plan.deadline)
    
    results = {
        "plan_id":plan.plan_id,
        "solutions":{
            "genetic_algorithm":plan1,
            "graph_neural_network":plan2,
            "linear_algorithm":plan3
        }
    }
    return results
