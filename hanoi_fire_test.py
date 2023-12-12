# -*- coding: utf-8 -*-

# Import necessary libraries
import pandas as pd
import ai_wonder as wonder

# Input with default values
def user_input(prompt, default):
    response = input(f"{prompt} (default: {default}): ")
    return response if response else default

# The driver
if __name__ == "__main__":
    print(f"Hanoi Fire 'Damage_Scale' Predictor")
    print("Powered by AI Wonder\n")
    
    # User inputs
    BuildingType = user_input("Building_Type", "'Residential'")
    SprinklerSystemPresent = user_input("Sprinkler_System_Present", "'Yes'")
    FireSafetyTrainingConducted = user_input("Fire_Safety_Training_Conducted", "'Yes'")
    NearestFireStationLocation = user_input("Nearest_Fire_Station_Location", "'West'")
    TypesofNearbyBuildings = user_input("Types_of_Nearby_Buildings", "'Industrial'")
    ElectricalEquipmentInspectionConducted = user_input("Electrical_Equipment_Inspection_Conducted", "'No'")
    GasEquipmentInspectionConducted = user_input("Gas_Equipment_Inspection_Conducted", "'No'")
    RecentRepairReplacementHistory = user_input("Recent_Repair_Replacement_History", "'Over 3 years'")
    Month = int(user_input("Month", 5))
    BuildingAge = int(user_input("Building_Age", 47))
    BuildingAreasqm = int(user_input("Building_Area_(sqm)", 665))
    BuildingHeightm = int(user_input("Building_Height_(m)", 5))
    NumberofFloors = int(user_input("Number_of_Floors", 1))
    TimetoExtinguishmin = int(user_input("Time_to_Extinguish_(min)", 53))
    ResponseTimemin = int(user_input("Response_Time_(min)", 34))
    NumberofFireExtinguishers = int(user_input("Number_of_Fire_Extinguishers", 1))
    NumberofEmergencyExits = int(user_input("Number_of_Emergency_Exits", 1))
    NumberofFireAlarms = int(user_input("Number_of_Fire_Alarms", 1))
    WidthofNearbyRoadsm = int(user_input("Width_of_Nearby_Roads_(m)", 10))
    DistancetoNearbyBuildingsm = int(user_input("Distance_to_Nearby_Buildings_(m)", 44))
    TemperatureC = float(user_input("Temperature_(_C)", 16.96))
    Humidity = float(user_input("Humidity_(%)", 69.24))
    WindSpeedms = float(user_input("Wind_Speed_(m_s)", 4.16))
    Precipitationmm = float(user_input("Precipitation_(mm)", 10.45))

    # Make datapoint from user input
    point = pd.DataFrame([{
        'Building_Type': BuildingType,
        'Sprinkler_System_Present': SprinklerSystemPresent,
        'Fire_Safety_Training_Conducted': FireSafetyTrainingConducted,
        'Nearest_Fire_Station_Location': NearestFireStationLocation,
        'Types_of_Nearby_Buildings': TypesofNearbyBuildings,
        'Electrical_Equipment_Inspection_Conducted': ElectricalEquipmentInspectionConducted,
        'Gas_Equipment_Inspection_Conducted': GasEquipmentInspectionConducted,
        'Recent_Repair_Replacement_History': RecentRepairReplacementHistory,
        'Month': Month,
        'Building_Age': BuildingAge,
        'Building_Area_(sqm)': BuildingAreasqm,
        'Building_Height_(m)': BuildingHeightm,
        'Number_of_Floors': NumberofFloors,
        'Time_to_Extinguish_(min)': TimetoExtinguishmin,
        'Response_Time_(min)': ResponseTimemin,
        'Number_of_Fire_Extinguishers': NumberofFireExtinguishers,
        'Number_of_Emergency_Exits': NumberofEmergencyExits,
        'Number_of_Fire_Alarms': NumberofFireAlarms,
        'Width_of_Nearby_Roads_(m)': WidthofNearbyRoadsm,
        'Distance_to_Nearby_Buildings_(m)': DistancetoNearbyBuildingsm,
        'Temperature_(_C)': TemperatureC,
        'Humidity_(%)': Humidity,
        'Wind_Speed_(m_s)': WindSpeedms,
        'Precipitation_(mm)': Precipitationmm,
    }])

    # Predict
    model = wonder.load_model('hanoi_fire_model.pkl')
    prediction = str(model.predict(point)[0])
    print(f"\nPrediction of 'Damage_Scale' is '{prediction}'.")
###
