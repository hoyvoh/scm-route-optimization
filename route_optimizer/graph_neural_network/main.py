import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import datetime
import pandas as pd
import os
import sys
from datetime import datetime
from tqdm import tqdm


current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path))
if project_root not in sys.path:
    sys.path.append(project_root)

from routes_selection import hard_route

model_configs = {
    'learning_rate':0.1,
    'epochs':100,
    'penalty_factor': 100000 # set to a test num
}

class RouteOptimizerGCN(nn.Module):
    def __init__(self, num_features):
        super(RouteOptimizerGCN, self).__init__()
        self.conv1 = GCNConv(num_features, 32)
        self.conv2 = GCNConv(32, 32)
        self.fc = nn.Linear(32, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        scores = self.fc(x).squeeze()
        return scores

    def select_routes(self, scores, data, demand, available_hours, routes_data_df):
        probabilities = torch.sigmoid(scores)
        sorted_indices = torch.argsort(probabilities, descending=True)

        total_weight = 0
        total_cost = 0
        selected_routes = []
        
        for idx in sorted_indices:
            idx = idx.item()
            route = routes_data_df.iloc[idx]  # Get the actual route data

            # Check if this route can be added without violating demand and travel time
            if total_weight + route['weight'] > demand:
                continue  # Skip if adding this route exceeds demand
            if route['travel_hours'] > available_hours:
                continue  # Skip if the route's travel time exceeds available hours

            # If feasible, add this route
            total_weight += route['weight']
            total_cost += route['total_cost']  # Accumulate cost
            selected_routes.append(idx)  # Append index of selected route

            # Break early if we meet the demand
            if total_weight >= demand:
                break
        print(selected_routes)
        print(total_cost)
        print(total_weight)
        # Final check: if demand is met
        if total_weight >= demand:
            return selected_routes, total_cost  # Return selected indices and total cost
        else:
            return None, float('inf')  # No feasible solution found

def optimize(routes_data, demand, deadline, args=model_configs):
    node_features = []
    for route in routes_data:
        node_features.append([
            route['weight'],          # Include weight as a feature
            route['travel_hours']     # Include travel time as a feature
        ])
    routes_data_df = pd.DataFrame(routes_data)
    routes_data_df['idx'] = [i for i in range(len(routes_data))]

    # Convert node features to tensor
    x = torch.tensor(node_features, dtype=torch.float)
    num_nodes = len(routes_data)
    edge_index = torch.combinations(torch.arange(num_nodes), r=2).T

    # Create data object and initialize the GCN
    data = Data(x=x, edge_index=edge_index)
    model = RouteOptimizerGCN(num_features=x.shape[1])
    optimizer = torch.optim.SGD(model.parameters(), lr=args['learning_rate'])

    # Convert deadline to total available hours from now
    current_time = datetime.now()
    deadline_datetime = datetime.combine(deadline, datetime.min.time())
    total_available_hours = (deadline_datetime - current_time).total_seconds() / 3600  # Convert seconds to hours

    # Training
    model.train()
    for epoch in tqdm(range(args['epochs'])):
        optimizer.zero_grad()
        
        # Forward pass to get scores
        scores = model(data)

        # Route selection based on scores
        selected_indices, total_cost = model.select_routes(scores, data, demand, total_available_hours, routes_data_df)
        
        # Loss: total cost + penalties if constraints aren't met
        loss = torch.tensor(total_cost, dtype=torch.float, requires_grad=True)
        if selected_indices is not None:
            total_weight = sum(routes_data_df.iloc[idx]['weight'] for idx in selected_indices)
            total_travel_hours = max(routes_data_df.iloc[idx]['travel_hours'] for idx in selected_indices)

            if total_weight < demand:
                demand_penalty = (demand - total_weight) * args['penalty_factor']
                loss += demand_penalty
            if total_travel_hours > total_available_hours:
                deadline_penalty = (total_travel_hours - total_available_hours) * args['penalty_factor']
                loss += deadline_penalty
            
            # Backpropagate and optimize
            loss.backward()
            optimizer.step()

    # Final evaluation to get the best delivery plan
    model.eval()
    with torch.no_grad():
        scores = model(data)
        selected_indices, total_cost = model.select_routes(scores, data, demand, total_available_hours, routes_data_df)
    print('at model, selected indecies', selected_indices)
    if selected_indices:
        selected_routes_df = routes_data_df.iloc[selected_indices]
        selected_routes_list = selected_routes_df.to_dict(orient='records')
        return selected_routes_list  # Return selected routes
    else:
        print("No feasible plan meets the demand.")
        return None


if __name__ == '__main__':
    demand = 180
    sample_dest = '5/219, Đ. Thủ Khoa Huân/Tổ 4A Đ.Đ1, Thuận Giao, Thuận An, Bình Dương 75000'
    routes = hard_route[sample_dest]
    deadline = datetime(2024, 11, 16).date() 
    selected_routes = optimize(routes_data=routes, demand=demand, deadline=deadline, args=model_configs)
    print(selected_routes)
