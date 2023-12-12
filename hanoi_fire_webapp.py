# -*- coding: utf-8 -*-

# Import necessary libraries
import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import ai_wonder as wonder

# The driver
if __name__ == "__main__":
    # Streamlit interface
    st.subheader(f"Hanoi Fire 'Damage_Scale' Predictor")
    st.markdown("Powered by :blue[**AI Wonder**]")
    st.markdown("")

    # Arrange radio buttons horizontally
    st.write('<style> div.row-widget.stRadio > div { flex-direction: row; } </style>',
        unsafe_allow_html=True)

    # User inputs
    BuildingType = st.radio("Building_Type", ['Commercial', 'Industrial', 'Residential'], index=0)
    SprinklerSystemPresent = st.radio("Sprinkler_System_Present", ['Yes', 'No'], index=0)
    FireSafetyTrainingConducted = st.radio("Fire_Safety_Training_Conducted", ['No', 'Yes'], index=0)
    NearestFireStationLocation = st.selectbox("Nearest_Fire_Station_Location", ['East', 'South', 'North', 'West'], index=2)
    TypesofNearbyBuildings = st.selectbox("Types_of_Nearby_Buildings", ['Residential', 'Public', 'Industrial', 'Commercial'], index=0)
    ElectricalEquipmentInspectionConducted = st.radio("Electrical_Equipment_Inspection_Conducted", ['Yes', 'No'], index=0)
    GasEquipmentInspectionConducted = st.radio("Gas_Equipment_Inspection_Conducted", ['Yes', 'No'], index=0)
    RecentRepairReplacementHistory = st.selectbox("Recent_Repair_Replacement_History", ['Over 3 years', 'None', 'Within 1 year', '1-3 years'], index=0)
    Month = st.number_input("Month", value=5)
    BuildingAge = st.number_input("Building_Age", value=47)
    BuildingAreasqm = st.number_input("Building_Area_(sqm)", value=665)
    BuildingHeightm = st.number_input("Building_Height_(m)", value=5)
    NumberofFloors = st.number_input("Number_of_Floors", value=1)
    TimetoExtinguishmin = st.number_input("Time_to_Extinguish_(min)", value=53)
    ResponseTimemin = st.number_input("Response_Time_(min)", value=34)
    NumberofFireExtinguishers = st.number_input("Number_of_Fire_Extinguishers", value=1)
    NumberofEmergencyExits = st.number_input("Number_of_Emergency_Exits", value=1)
    NumberofFireAlarms = st.number_input("Number_of_Fire_Alarms", value=1)
    WidthofNearbyRoadsm = st.number_input("Width_of_Nearby_Roads_(m)", value=10)
    DistancetoNearbyBuildingsm = st.number_input("Distance_to_Nearby_Buildings_(m)", value=44)
    TemperatureC = st.number_input("Temperature_(_C)", value=16.96)
    Humidity = st.number_input("Humidity_(%)", value=69.24)
    WindSpeedms = st.number_input("Wind_Speed_(m_s)", value=4.16)
    Precipitationmm = st.number_input("Precipitation_(mm)", value=10.45)

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

    st.markdown("")

    # Predict and Explain
    if st.button('Predict'):
        st.markdown("")

        with st.spinner("Loading trained model..."):
            state = wonder.load_state('hanoi_fire_state.pkl')
            model = wonder.input_piped_model(state)

        with st.spinner("Making predictions..."):
            prediction = str(model.predict(point)[0])
            st.success(f"Prediction of **{state.target}** is **{prediction}**.")
            st.markdown("")

        with st.spinner("Making explanations..."):
            st.info("Feature Importances")
            importances = pd.DataFrame(wonder.local_explanations(state, point), columns=["Feature", "Value", "Importance"])
            st.dataframe(importances.round(2))

            st.info("Some Counterfactuals")
            counterfactuals = wonder.whatif_instances(state, point).iloc[:20]
            st.dataframe(counterfactuals.round(2))
