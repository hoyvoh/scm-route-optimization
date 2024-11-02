import os
import sys
from time import time 
import requests
from geopy.distance import geodesic
from dotenv import load_dotenv
import regex as re
from tqdm import tqdm
import os
import sys
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path))
if project_root not in sys.path:
    sys.path.append(project_root)
from data_collection import suppliers, vehicles, parkings, destination_v2

# Load environment variables
load_dotenv()

# Add project root to the system path once
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# 'AlzaSy-L-OycIT3WSfwTR-dyE_Q0TmtfqArvD9a'(0)
# AlzaSyXYUfjJjtewNv2GSgZjonLnDcZLWQN1rFz
#
# 
# 

API_KEY = 'AlzaSyT_RJLWIUDUGuUDO3XrtNEdiGlOsBpbNlf' #os.getenv('GOMAP_API_KEY')

def convert_to_hours(time_str):
    """Convert Vietnamese time format to hours."""
    pattern = r"(?:(\d+)\s*ngày)?\s*(?:(\d+)\s*giờ)?\s*(?:(\d+)\s*phút)?\s*(?:(\d+)\s*giây)?"
    match = re.match(pattern, time_str)
    if not match:
        return 0.0

    days, hours, minutes, seconds = (int(match.group(i) or 0) for i in range(1, 5))
    return round(days * 24 + hours + minutes / 60 + seconds / 3600, 2)

def get_travel_data(origin, destination):
    """Fetch travel time and distance between addresses using GoMap API."""
    url = 'https://routes.gomaps.pro/directions/v2:computeRoutes'
    headers = {
        'X-Goog-FieldMask': 'routes.distanceMeters,routes.duration',
        'Content-Type': 'application/json',
        'X-Goog-Api-Key': API_KEY
    }
    data = {
        "origin": {"address": origin},
        "destination": {"address": destination},
        "travelMode": 'drive'
    }

    response = requests.post(url, headers=headers, json=data).json()
    duration_text = response.get('routes', [{}])[0].get('localizedValues', {}).get('duration', {}).get('text', '0 giờ')
    distance_text = response.get('routes', [{}])[0].get('localizedValues', {}).get('distance', {}).get('text', '0')

    hours = convert_to_hours(duration_text)
    distance_km = float(re.sub(r'[^\d.]', '', distance_text)) if distance_text else 0
    return hours, distance_km

def find_closest_parking(mid_point, parking_areas):
    """Find the closest parking area to a given midpoint (latitude, longitude)."""
    # Apply .loc to set 'distance' column to avoid SettingWithCopyWarning
    parking_areas = parking_areas.copy()  # Explicitly make a copy to ensure no warning
    parking_areas.loc[:, 'distance'] = parking_areas.apply(
        lambda row: geodesic(mid_point, (row['latitude'], row['longitude'])).kilometers, axis=1
    )
    closest_parking = parking_areas.loc[parking_areas['distance'].idxmin()]
    return closest_parking if closest_parking['distance'] != float('inf') else None

def fragment_route(supplier, destination, parking_areas):
    """Recursively fragment route if travel time exceeds 8 hours."""
    travel_hours, _ = get_travel_data(supplier['address'], destination['address'])
    
    if travel_hours <= 8:
        return [[supplier['address'], destination['address']]]

    mid_point = (
        (supplier['latitude'] + destination['latitude']) / 2,
        (supplier['longitude'] + destination['longitude']) / 2
    )
    closest_parking = find_closest_parking(mid_point, parking_areas)
    if closest_parking is not None:
        return (fragment_route(supplier, closest_parking, parking_areas) + 
                fragment_route(closest_parking, destination, parking_areas))
    
    return [[supplier['address'], destination['address']]]

def calculate_costs(route, vehicle, weight, supplier_price):
    """Calculate costs based on travel distance and time."""
    total_km, total_hours = 0, 0
    print('Iterating in calculate_cost...')
    for origin, dest in tqdm(route):
        hours, distance = get_travel_data(origin, dest)
        total_km += distance
        total_hours += hours

    driver_cost = 24000 * ((total_hours % 8) + 1) * 0.5 + 24000*total_hours
    fuel_cost = total_km*9.3* 21427.5 / 100 # 9.3/100 (L/Km)
    goods_cost = supplier_price * weight * 1000

    return {
        "driver_cost": driver_cost,
        "fuel_cost": fuel_cost,
        "goods_cost": goods_cost,
        "total_cost": driver_cost + fuel_cost + goods_cost,
        "travel_hours": total_hours,
        "travel_kms": total_km
    }

def get_geocode(destination):
    """Get geocode data for a destination address."""
    url = 'https://maps.gomaps.pro/maps/api/geocode/json'
    params = {
        'key': API_KEY,
        'address': destination['address'],
        'language': 'vi',
        'region': 'vi'
    }
    response = requests.get(url, params=params, timeout=5).json()
    result = response['results'][0] if response['status'] == 'OK' else {}
    return {
        "address": result.get("formatted_address"),
        "latitude": result.get('geometry', {}).get('location', {}).get('lat'),
        "longitude": result.get('geometry', {}).get('location', {}).get('lng'),
        "place_id": result.get("place_id"),
        "address_key": destination['address'],
        "id": destination['id']
    }

def route_selection(suppliers, parking_areas, destination, vehicles):
    start = time()
    """Select the best routes based on suppliers, parking areas, and vehicle data."""
    destination = get_geocode(destination)
    routes = []

    print("iterating suppliers in route_selection...")
    for _, supplier in tqdm(suppliers.iterrows()):
        sub_routes = fragment_route(supplier, destination, parking_areas)
        print("iterating vehicles in route_selection...")
        for _, vehicle in tqdm(vehicles.iterrows()):
            weight = min(vehicle['capacity'], supplier['Capacity'])
            route_data = {
                "route_id": f"{supplier['id']}_{destination['id']}",
                "vehicle": vehicle['registered_num'],
                "weight": weight,
                "routes": sub_routes
            }
            costs = calculate_costs(sub_routes, vehicle, weight, supplier['Price'])
            route_data.update(costs)
            routes.append(route_data)
    
    end = time()
    print(f'== Possible routes calculation finished after {end-start}s! ==')
    return routes

'''
1/haeley05@centreszv.com
0928989820
AlzaSyXYUfjJjtewNv2GSgZjonLnDcZLWQN1rFz
###
2/pastelgreen3@centreszv.com
02488874198
AlzaSycUzNof1yvz35n5Y4oySHTdCZDFb1ICx7k
###
3/heath44@centreszv.com
02855581772
AlzaSyKofIAJhrtKuqcQl5RhAFrdPzPJk7GgObR
###
4/elaina632@centreszv.com
0859280755
AlzaSyT_RJLWIUDUGuUDO3XrtNEdiGlOsBpbNlf
'''

if __name__ == '__main__':

    destination= {
            'id': f'dest_5', #  5
            'address': destination_v2['address'].to_list()[4] #  4
        }
    routes = route_selection(suppliers=suppliers[:30], parking_areas=parkings[:10], destination=destination, vehicles=vehicles[:5])
    hard_routes = {}
    hard_routes[destination_v2['address'].to_list()[4]] = routes
    # for i, dest in enumerate(destination_v2['address'].to_list()[0], start=1):
    #     print('Selecting potential routes for', dest)
    #     print('==START SELECTION==')
    #     destination = {
    #         'id': f'dest_{i}',
    #         'address': dest
    #     }
    #     routes = route_selection(suppliers=suppliers[:30], parking_areas=parkings[:10], destination=destination, vehicles=vehicles[:5])
    #     hard_routes[dest] = routes
    #     print('==END SELECTION==')
    with open('hard_route_4.txt', 'w', encoding='utf') as f: # _4
        f.write(str(hard_routes))
