import re
import pynldas2 as nldas
import pygeohydro as gh
import py3dep
import numpy as np
from math import radians, sin, cos, sqrt, atan2, degrees
from datetime import datetime
import math


def find_node_by_name(connections, target):
    """
    Function to search for the index of an element by name

    Args:
        connections (list): List of connection objects
        target (str): Name of element to fidn

    Returns:
        (int): index of element if it is found
    """
    # Loop through connections
    for i in range (len(connections)):
        # Check if names are equal
        if connections[i].name == target:
            # Return index
            return i     
    # Else return none
    return None

def findNumLoads(bus, loads):
    """
    Counts the number of loads connected to a specified bus.
    
    This function iterates through a list of loads, checking if each load's bus attribute
    matches the specified bus parameter. It counts and returns the total number of matching loads.
    
    Parameters:
    - bus (str): The identifier of the bus for which to count connected loads.
    - loads (list): A list of load objects. Each load object must have a 'bus' attribute.
    
    Returns:
    - int: The total number of loads connected to the specified bus.
    """
    num = 0
    for load in loads:
        if bus == load.bus:
            num = num+1
    return num

def findNodeNum(bus, nodes):
    """
    Finds and returns the numerical identifier of a node with a specified name.
    
    This function iterates through a list of node objects, comparing each node's name attribute
    with the specified bus parameter. If a match is found, it returns the numerical identifier
    (num attribute) of the matching node.
    
    Parameters:
    - bus (str): The name of the node to find.
    - nodes (list): A list of node objects. Each node object must have 'name' and 'num' attributes.
    
    Returns:
    - int or None: The numerical identifier of the matching node, or None if no match is found.
    """
    if '.' in bus:
        bus = nodeNameSplit(bus)

    for node in nodes:
        if bus == node.name:
            return node.num
 
def getWeatherByCoords(lon,lat,start,end):
    """
    Grabs an hourly weather dataset at the specified longitude and latitude.

    Args:
    lon (str): Longitude of desired location.
    lat (str): Latitude of desired location.
    start (str): Start Time of Desired Event
    end (str): Stop Time of Desired Event

    NLDAS Hourly Weather Data Identifiers
    - prcp: precipitation hourly total (kg/m^2)
    - rlds: surface downward longwave radiation (W/m^2)
    - rsds: surface downward shortwave radiation (W/m^2)
    - temp: air temperature (K) ** at 2 meters above the surface
    - humidity: specific humidity (kg/kg) ** at 2 meters above the surface
    - wind_u: U wind component (m/s) at 10 meters above the surface
    - wind_v: V wind component (m/s) at 10 meters above the surface

    Indexing: data['wind_u'][1] gives the wind_u value for second hour of event
    - Wind Speed m/sec = sqrt(data['wind_u']**2 + data['wind_v']**2)
    - getWeather(-120.82899598, 36.50996789,"2010-01-08", "2010-01-08")

    Returns:
    data (dict): Dictionary corresponding to the hour climatology data associated with the data and location
    """

    data =nldas.get_bycoords(list(zip([lon],[lat])),start,end) 
    return data

def getLandCover(coords):
    """
    Grabs the tree canopy coverage (in year 2016) at the specified longitude and latitude.

    Args:
    coords (tuple): Longitude, and Latitude of desired location.

    Returns:
    tcc (float): Value corresponding to the tree canopy coverage in that area
    """
    # Grab latitude and longitude from coords
    lon = coords[0]
    lat = coords[1]
    
    # OR get the data for specific coordinates using nlcd_bycoords (cover_statistics does not work with this method)
    land_usage_land_cover = gh.nlcd_bycoords(list(zip([lon],[lat])), years={"canopy": [2019]})

    # Grab the tree canopy coverage
    tcc = land_usage_land_cover.canopy_2019[0]

    # Return the land usage and cover data
    return tcc

def getElevationByCoords(coords):
   """
    Grabs the elevation in meters at the specified longitude and latitude using Py3DEP.

    Args:
    coords (tuple): Longitude, and Latitude of desired location.

    Returns:
    elevation (float): Value corresponding to the elevation in meters
    """
   # Elevation Acquisition (in meters)
   elevation = py3dep.elevation_bycoords(coords, crs=4326) 
   return elevation

def findEdgeElevation(bus1,bus2, nodes):
    """
    Determines edge elevation by finding midpoint between two nodes and finding the elevation at that point.

    Args:
    bus1 (int): Number of first node connected to the edge
    bus2 (int): Number of second node connected to the edge
    nodes (list): List of nodes in the network

    Returns:
    edgeElevation (float): Value corresponding to the elevation for the edge in meters
    """

    # Loops through all the nodes
    for node in nodes:
        # Finds the node number for the first node and grabs it coords
        if bus1 == node.num:
            coord1 = node.coords
        # Finds the node number for the second node and grabs it coords
        if bus2 == node.num:
            coord2 = node.coords
    # Creates a list of the two coordinates 
    test_list = [coord1,coord2]

    # Average the two coordinates to get the midpoint
    res = [sum(ele) / len(test_list) for ele in zip(*test_list)] 

    # Find elevation at midpoint
    edgeElevation = getElevationByCoords(tuple(res))
    
    return edgeElevation
    
def roundup(x):
    """
    Function to round a float to integer using ceiling

    Args:
    x (float): Number to round

    Returns:
    rounded (int): Value corresponding to the rounded float
    """

    rounded = int(math.ceil(x / 100.0)) * 100
    return rounded

def cft(input_tuple):
    """
    Function to remove white space from a tuple while also converting values to floats

    Args:
    input_tuple (tuple): Tuple to modify

    Returns:
    newTuple (tuple): Modified Tuple
    """
    # Strip leading and trailing whitespace and convert to float
    newTuple = tuple(float(item.strip()) for item in input_tuple)

    return newTuple

def parseDate(date):
    """
    Parses a datetime string into a date string.

    Args:
        date (str): A string representing a datetime in the format '%Y-%m-%d %H:%M:%S'.

    Returns:
        str: A string representing the date extracted from the input datetime string
             in the format '%Y-%m-%d'.
             
    Raises:
        ValueError: If the input date string is not in the expected format or cannot be parsed.

    """
    datetime_obj = datetime.strptime(str(date), '%Y-%m-%d %H:%M:%S')

    date_str = datetime_obj.strftime('%Y-%m-%d')

    return date_str

def parseTime(time):
    """
    Parses a time string into a standardized format.

    Args:
        time (str): A string representing time in the format 'HHMM', where HH represents
                    hours (00 to 23) and MM represents minutes (00 to 59).

    Returns:
        str: A string representing the time in a standardized format 'HHMM', where hours
             and minutes are zero-padded if necessary.
    """
    
    time = str(time)
    if len(time) != 4:
        time = ("0" * (4 - len(time))) + time
    return time

def nodeNameSplit(text):
    """
    Splits the input text by the dot ('.') character and returns the first part.

    Parameters:
    text (str): The input text to be split.

    Returns:
    str: The first part of the input text before the first dot ('.') character.
    """
    return text.split('.')[0]

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points on the Earth's surface 
    using the Haversine formula. This formula is useful for calculating the shortest 
    distance over the Earth's surface, giving an 'as-the-crow-flies' distance between 
    the coordinates.

    Parameters:
    lat1 (float): Latitude of the first point in decimal degrees.
    lon1 (float): Longitude of the first point in decimal degrees.
    lat2 (float): Latitude of the second point in decimal degrees.
    lon2 (float): Longitude of the second point in decimal degrees.

    Returns:
    float: Distance between the two points in kilometers.

    """
    # Radius of Earth in kilometers
    R = 6371.0 
    
    # Calculate the differences in coordinates in radians
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    
    # Apply the Haversine formula
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    # Return the calculated distance in kilometers
    return R * c

def interpolate_points(lat1, lon1, lat2, lon2, n):
    """
    Generate a list of `n` equally spaced points on the great-circle path between two geographic coordinates. 
    The points are interpolated using spherical trigonometry.

    Parameters:
    lat1 (float): Latitude of the first point in decimal degrees.
    lon1 (float): Longitude of the first point in decimal degrees.
    lat2 (float): Latitude of the second point in decimal degrees.
    lon2 (float): Longitude of the second point in decimal degrees.
    n (int): Number of points to interpolate (including the start and end points).

    Returns:
    tuple: Two lists (LAT, LON) containing the interpolated latitudes and longitudes of the points, in decimal degrees.

    """
    # Convert input coordinates from decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Calculate the great-circle distance between the two points
    d = haversine(lat1, lon1, lat2, lon2)
    # Initialize lists to hold the interpolated latitude and longitude points
    LAT = []
    LON = []
    # Loop through the number of required points
    for i in range(n):
        # Calculate the fractional distance along the great-circle path
        fraction = i / (n - 1)
        
        # Apply the interpolation formula using spherical trigonometry
        A = sin((1 - fraction) * d) / sin(d)
        B = sin(fraction * d) / sin(d)
        
        # Determine the coordinates of the interpolated point in Cartesian coordinates
        x = A * cos(lat1) * cos(lon1) + B * cos(lat2) * cos(lon2)
        y = A * cos(lat1) * sin(lon1) + B * cos(lat2) * sin(lon2)
        z = A * sin(lat1) + B * sin(lat2)
        
        # Convert back to latitude and longitude in radians
        lat = atan2(z, sqrt(x ** 2 + y ** 2))
        lon = atan2(y, x)
        
        # Append the interpolated latitude and longitude in decimal degrees to the result lists
        LAT.append(degrees(lat))
        LON.append(degrees(lon))
    # Return the lists of interpolated latitude and longitude points
    return LAT,LON

def findAvgLineVegetation(bus1,bus2, nodes, n):
    """
    Find the average vegetation canopy cover between two nodes specified by their IDs (bus1 and bus2).
    The function interpolates points between the two nodes and calculates the average canopy cover
    using vegetation data.

    Parameters:
    bus1 (int or str): ID of the first bus (node).
    bus2 (int or str): ID of the second bus (node).
    nodes (list): A list of node objects. Each node is expected to have attributes `num` (ID) and `coords` (longitude, latitude).
    n (int): The number of points to interpolate between the two nodes.

    Returns:
    float: The average vegetation canopy cover percentage over the interpolated path between the two nodes.

    """
    # Loops through all the nodes
    for node in nodes:
        # Finds the node number for the first node and grabs it coords
        if bus1 == node.num:
            start = node.coords
        # Finds the node number for the second node and grabs it coords
        if bus2 == node.num:
            end = node.coords
    
    # Extract the coordinates from the nodes
    start_lon, start_lat = start
    end_lon, end_lat = end
    
    # Interpolate points between the two coordinates
    lat,lon = interpolate_points(start_lat, start_lon, end_lat, end_lon, n)
    
    # Retrieve vegetation data (e.g., canopy cover) for the interpolated points
    tcc = gh.nlcd_bycoords(zip(lon,lat),years={"canopy": [2019]})
    
    # Convert the canopy cover data to a NumPy array
    lineVeg = np.array(tcc['canopy_2019'])
    
    # Calculate the average vegetation canopy cover over the interpolated path
    avgVeg = np.sum(lineVeg) / n
    
    # Return the calculated average vegetation canopy cover
    return avgVeg

def fixBusName(buses):
    newBuses = []
    for bus in buses:
        newBus = bus.split('.')[0]
        newBuses.append(newBus)
    return newBuses