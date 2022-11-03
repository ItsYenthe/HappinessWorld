import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import geopandas as gpd
import pycountry 
from statsmodels.formula.api import ols
from streamlit_option_menu import option_menu

# -------------- SETTINGS ------------

page_title = 'Gelukkigheidsscore'
page_icon = ':smile:'
layout = 'wide'

#-------------------------------------

st.set_page_config(page_title = page_title, page_icon = page_icon, layout = layout)
st.title(page_title + '' + page_icon)

# ------------------------------------

selected = option_menu(
    menu_title = None,
    options = ['Datasets','Kaart', 'Regressie', '1D en 2D plots'],
    icons = ['clipboard-data','map', 'graph-up', 'bar-chart-line'],
    orientation = 'horizontal')

# ------------------------------------

# Datasets inladen
happiness = pd.read_csv('2019.csv')
countries = pd.read_csv('countries of the world.csv')
location = pd.read_csv('countries.csv')

# Kolomnaam veranderen zodat er geen spaties tussen zitten
happiness = happiness.rename(columns = {'Country or region': 'Country_Region'})

# Functie aanmaken voor country code
def alpha3code(column):
    CODE = []
    for country in column:
        try:
            code = pycountry.countries.get(name=country).alpha_3
            CODE.append(code)
        except:
            CODE.append('None')
    return CODE

# Nieuwe kolom voor de code
happiness['CODE'] = alpha3code(happiness.Country_Region)

# Nieuwe dataset met geometry per code
world_code = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# Kolommen hernoemen
world_code.columns = ['pop_est', 'continent', 'name', 'CODE', 'gdp_md_est', 'geometry']

# Goede naam geven bij world_code en location
world_code['name'] = world_code['name'].replace('United States of America', 'United States')
world_code['name'] = world_code['name'].replace('Dem. Rep. Congo', 'Congo (Kinshasa)')
world_code['name'] = world_code['name'].replace('Congo', 'Congo (Brazzaville)')
world_code['name'] = world_code['name'].replace('S. Sudan', 'South Sudan')
world_code['name'] = world_code['name'].replace('Central African Rep.', 'Central African Republic')
world_code['name'] = world_code['name'].replace('Czechia', 'Czech Republic')
location['Country'] = location['Country'].replace('Korea, South', 'South Korea')
location['Country'] = location['Country'].replace('Czechia', 'Czech Republic')
location['Country'] = location['Country'].replace('US', 'United States')

# Mergen met happiness dataset
happiness_world = pd.merge(world_code,happiness, left_on='name', right_on = 'Country_Region')

# Opnieuw mergen met dataset over longitude en latitude per land
happiness_world = happiness_world.merge(location,left_on='name', right_on = 'Country').sort_values(by='Score',ascending=False).reset_index()

# Spatie weghalen bij Country kolom
countries['Country'] = countries['Country'].str.lstrip().str.rstrip()

# Goede namen bij countries
countries['Country'] = countries['Country'].replace('Central African Rep.', 'Central African Republic')
countries['Country'] = countries['Country'].replace('Congo, Dem. Rep.', 'Congo (Kinshasa)')
countries['Country'] = countries['Country'].replace('Congo, Repub. of the', 'Congo (Brazzaville)')
countries['Country'] = countries['Country'].replace('Korea, South', 'South Korea')

# Nieuwe dataset mergen
happy = happiness_world.merge(countries, left_on = 'Country_Region', right_on = 'Country')

# Kolommen selecteren
happy = happy[['Country_Region', 'continent', 'CODE_x' ,'Score', 'GDP per capita', 'Healthy life expectancy', 'Pop. Density (per sq. mi.)', 'Phones (per 1000)']]

# Nieuwe kolomnamen
happy = happy.rename(columns = {'continent': 'Continent', 'GDP per capita': 'GDP_per_cap', 'Healthy life expectancy':'Healthy_life_expectancy', 'Pop. Density (per sq. mi.)': 'Pop_density_sq_mile', 'Phones (per 1000)':'Phones_per_1000'})

# Komma vervangen naar een punt en daarna omzetten naar een float
happy['Pop_density_sq_mile'] = happy['Pop_density_sq_mile'].str.replace(',', '.')
happy['Phones_per_1000'] = happy['Phones_per_1000'].str.replace(',', '.')
happy['Pop_density_sq_mile'] = happy['Pop_density_sq_mile'].astype('float')
happy['Phones_per_1000'] = happy['Phones_per_1000'].astype('float')



# Sorteren op score
happy = happy.sort_values('Score', ascending = False)

if selected == 'Datasets':
    st.title('Happy dataset')
    st.dataframe(happy.head())
    st.write('De gecleande dataset waarin alle datasets met enkel de benodigde kolommen gemerged.')
    st.write('')
    st.write('')
    st.write('De gebruikte datasets: \n 1. Wereld gelukkigheidsscore: "2019.csv" \n 2. Wereldlanden statistieken: "countries of the world.csv" \n 3. Geodata wereldlanden: "countries.csv" \n 4. Geodata via API: get_path("naturalearth_lowres")')

# Nieuwe dataset per continent
Europe_happy = happy[happy['Continent'] == 'Europe']
Oceania_happy = happy[happy['Continent'] == 'Oceania']
North_America_happy = happy[happy['Continent'] == 'North America']
South_America_happy = happy[happy['Continent'] == 'South America']
Asia_happy = happy[happy['Continent'] == 'Asia']
Africa_happy = happy[happy['Continent'] == 'Africa']

if selected == 'Kaart':
    st.title('Gelukkigheidsscore per land')
    fig = go.Figure()
    
    # Dropdown buttons
    dropdown_buttons = [
        {'label': 'Wereld', 'method':'update',
        'args':[{'visible':[True, False, False, False, False, False, False]}, {'title':'Gelukkigheidsscore van de wereld'}]},
        {'label': 'Europa', 'method':'update',
        'args':[{'visible':[False, True, False, False, False, False, False]}, {'title':'Gelukkigheidsscore van Europa'}]},
        {'label': 'Oceanië', 'method':'update',
        'args':[{'visible':[False, False, True, False, False, False, False]}, {'title':'Gelukkigheidsscore van Oceanië'}]},
        {'label': 'Noord Amerika', 'method':'update',
        'args':[{'visible':[False, False, False, True, False, False, False]}, {'title':'Gelukkigheidsscore van Noord Amerika'}]},
        {'label': 'Zuid Amerika', 'method':'update',
        'args':[{'visible':[False, False, False, False, True, False, False]}, {'title':'Gelukkigheidsscore van Zuid Amerika'}]},
        {'label': 'Azië', 'method':'update',
        'args':[{'visible':[False, False, False, False, False, True, False]}, {'title':'Gelukkigheidsscore van Azië'}]},
        {'label': 'Afrika', 'method':'update',
        'args':[{'visible':[False, False, False, False, False, False, True]}, {'title':'Gelukkigheidsscore van frika'}]}
    ]
    
    fig.add_trace(go.Choropleth(
        locations = happy['CODE_x'],
        z = happy['Score'],
        text = happy['Country_Region'],
        colorscale = 'Reds',
        autocolorscale=False,
        reversescale=False,
        marker_line_color='darkgray',
        marker_line_width=0.5,
        colorbar_title = 'Gelukkigheids<br>Score',
        visible = True))
    
    fig.add_trace(go.Choropleth(
        locations = Europe_happy['CODE_x'],
        z = Europe_happy['Score'],
        text = Europe_happy['Country_Region'],
        colorscale = 'Reds',
        autocolorscale=False,
        reversescale=False,
        marker_line_color='darkgray',
        marker_line_width=0.5,
        colorbar_title = 'Gelukkigheids<br>Score',
        visible = False))
    
    fig.add_trace(go.Choropleth(
        locations = Oceania_happy['CODE_x'],
        z = Oceania_happy['Score'],
        text = Oceania_happy['Country_Region'],
        colorscale = 'Reds',
        autocolorscale=False,
        reversescale=False,
        marker_line_color='darkgray',
        marker_line_width=0.5,
        colorbar_title = 'Gelukkigheids<br>Score',
        visible = False))
    
    fig.add_trace(go.Choropleth(
        locations = North_America_happy['CODE_x'],
        z = North_America_happy['Score'],
        text = North_America_happy['Country_Region'],
        colorscale = 'Reds',
        autocolorscale=False,
        reversescale=False,
        marker_line_color='darkgray',
        marker_line_width=0.5,
        colorbar_title = 'Gelukkigheids<br>Score',
        visible = False))
    
    fig.add_trace(go.Choropleth(
        locations = South_America_happy['CODE_x'],
        z = South_America_happy['Score'],
        text = South_America_happy['Country_Region'],
        colorscale = 'Reds',
        autocolorscale=False,
        reversescale=False,
        marker_line_color='darkgray',
        marker_line_width=0.5,
        colorbar_title = 'Gelukkigheids<br>Score',
        visible = False))
    
    fig.add_trace(go.Choropleth(
        locations = Asia_happy['CODE_x'],
        z = Asia_happy['Score'],
        text = Asia_happy['Country_Region'],
        colorscale = 'Reds',
        autocolorscale=False,
        reversescale=False,
        marker_line_color='darkgray',
        marker_line_width=0.5,
        colorbar_title = 'Gelukkigheids<br>Score',
        visible = False))
    
    fig.add_trace(go.Choropleth(
        locations = Africa_happy['CODE_x'],
        z = Africa_happy['Score'],
        text = Africa_happy['Country_Region'],
        colorscale = 'Reds',
        autocolorscale=False,
        reversescale=False,
        marker_line_color='darkgray',
        marker_line_width=0.5,
        colorbar_title = 'Gelukkigheids<br>Score',
        visible = False))
    
    
    # Updaten layout
    fig.update_layout({'updatemenus':[{'type': 'dropdown',
                                     'x':1.35, 'y':0.8,
                                     'showactive':True,
                                     'active': 0,
                                     'buttons':dropdown_buttons}]},
                      title_text = '2019 Gelukkigheidsscore', 
                      geo=dict(
                      showframe=False,
                      showcoastlines=False,
                      projection_type='equirectangular'))
    
    st.plotly_chart(fig, use_container_width = True)
    st.write('De gelukkigheidsscore per land. De continenten zijn selecteerbaar.')
    st.write('')
    a1, a2 = st.columns(2, gap="small")
    with a1:
        st.write('Top 3 landen met de hoogste gelukkigheidsscore:\n 1. Finland\n 2. Denemarken\n 3. Noorwegen')
    with a2:
        st.write('Top 3 landen met de laagste gelukkigheidsscore:\n 1. Centraal Afrikaanse Republiek\n 2. Afghanistan\n 3. Tanzanië')

    
# Model maken
mdl_healthy_life_exp_vs_score = ols('Score ~ Healthy_life_expectancy', data = happy).fit()

# Voorspellende data
explanatory_data = pd.DataFrame({"Healthy_life_expectancy": np.arange(0, 1.5, 0.25)})

# Voorspelde data
prediction_data = explanatory_data.assign(Score = mdl_healthy_life_exp_vs_score.predict(explanatory_data))

# Toekomst voorspellen
little_happy = pd.DataFrame({'Healthy_life_expectancy':np.arange(1.25, 3, 0.25)})
pred_little_happy = little_happy.assign(Score = mdl_healthy_life_exp_vs_score.predict(little_happy))

print(f'R^2: {round(mdl_healthy_life_exp_vs_score.rsquared, 3)}')

# Outliers

# Healthy_life_expectancy:
# Q1 en Q3 definieren
q1_life = happy.Healthy_life_expectancy.quantile(0.25)
q3_life = happy.Healthy_life_expectancy.quantile(0.75)

# IQR life expectancy
iqr_life = q3_life-q1_life

# Outliers life expectancy eruit halen
outlier_life = (happy.Healthy_life_expectancy <= q3_life + 1.5*iqr_life)
happy_outlier = happy.loc[outlier_life]

# Score:
# Q1 en Q3 definieren
q1_score = happy.Score.quantile(0.25)
q3_score = happy.Score.quantile(0.75)

# IQR Score
iqr_score = q3_score-q1_score

# Outliers Score eruit halen
outlier_score = (happy.Score <= q3_score + 1.5*iqr_score)
happy_outlier = happy.loc[outlier_score]

# Conclusie: Geen uitschieters, want len(happy) = len(happy_outer)

if selected == 'Regressie':
    st.title('Gezonde levensverwachting tegen de gelukkigheidsscore')

    # Figuur maken
    fig0 = go.Figure()
    
    # Dropdown buttons
    dropdown_buttons = [
        {'label': 'Voorspelling nu', 'method':'update',
        'args':[{'visible':[True, True, False]}, {'title':'Voorspelling nu', 'xaxis': {'title':'Gezonde levensverwachting'}, 'yaxis': {'title':'Gelukkigheidsscore'}}]},
        {'label': 'Voorspelling toekomst', 'method':'update',
        'args':[{'visible':[True, True, True]}, {'title':'Voorspelling toekomst', 'xaxis': {'title':'Gezonde levensverwachting'}, 'yaxis': {'title':'Gelukkigheidsscore'}}]}    
    ]
    
    # Traces toevoegen
    fig0.add_trace(go.Scatter(x = happy.Healthy_life_expectancy, y = happy.Score, opacity = 0.8, mode = 'markers', name = 'Punten', visible = True))
    fig0.add_trace(go.Scatter(x=prediction_data["Healthy_life_expectancy"], y=prediction_data["Score"], mode = 'lines', name = 'Voorspelling nu', visible = False))
    fig0.add_trace(go.Scatter(x=pred_little_happy["Healthy_life_expectancy"], y=pred_little_happy["Score"], mode = 'lines', name = 'Voorspelling Toekomst', visible = False))
    
    # Updaten layout
    fig0.update_layout({'updatemenus':[{'type': 'dropdown',
                                     'x':1.3, 'y':0.8,
                                     'showactive':True,
                                     'active': 0,
                                     'buttons':dropdown_buttons}]},
                      title_text = 'Gezonde levensverwachting tegen de gelukkigheidsscore',
                      xaxis_title = 'Gezonde levensverwachting', 
                      yaxis_title = 'Gelukkigheidsscore'
                     )
    
    st.plotly_chart(fig0, use_container_width = True)
    st.write('Deze regressielijn is het verband tussen de gezonde levensverwachting en de gelukkigheidsscore te zien. Hieruit komt een lineaire lijn. Dit model heeft een R^2 van 0,625 en er zijn geen uitschieters.')
    st.write('Met deze informatie is een model gemaakt dat de toekomstige gelukkigheidsscore voorspelt. Dit is weergeven met de groene lijn. Deze is aan te klikken in het dropdown-menu.')
    
# GDP in groepen verdelen
bins= [0,0.4,0.8,1,1.2,1.4,1.8]
labels = ['0-0,4','0,4-0,8','0,8-1','1-1,2', '1,2-1,4', '1,4+']
happy['GDP_group'] = pd.cut(happy['GDP_per_cap'], bins=bins, labels=labels, right=False)

# Pop Density in groepen verdelen
bins2 = [0,30,70,120,160,250,1026]
labels2 = ['0-30','30-70','70-120','120-160','160-250','250+']
happy['Pop_density_group'] = pd.cut(happy['Pop_density_sq_mile'], bins=bins2, labels=labels2)    

# Model maken
mdl_phone_vs_score = ols('Score ~ Phones_per_1000', data = happy).fit()

# Voorspellende data
explanatory_data2 = pd.DataFrame({"Phones_per_1000": np.arange(0, 800, 0.25)})

# Voorspelde data
prediction_data2 = explanatory_data2.assign(Score = mdl_phone_vs_score.predict(explanatory_data2))

# Outliers

# Phones:
# Q1 en Q3 definieren
q1_phone = happy.Phones_per_1000.quantile(0.25)
q3_phone = happy.Phones_per_1000.quantile(0.75)

# IQR life expectancy
iqr_phone = q3_phone-q1_phone

# Outliers life expectancy eruit halen
outlier_phone = (happy.Phones_per_1000 <= q3_phone + 1.5*iqr_phone)
happy_outlier_phone = happy.loc[~outlier_phone]


if selected == '1D en 2D plots':
    st.title('Visualisaties en verbanden')
    
    st.header('Histogram gelukkigheidsscore')

    # Figuur maken
    fig1 = go.Figure()
    
    # Histogram plotten
    fig1.add_trace(go.Histogram(x = happy.Score, name='Aantal landen per gelukkigheidsscore', nbinsx=10))
    
    # Lijnen toevoegen bij het gemiddelde en de mediaan
    fig1.add_trace(go.Scatter(x = [happy.Score.mean(), happy.Score.mean()], y = [0, 25], 
                            mode = 'lines', line = {'color':'red'}, name = 'Gemiddelde', visible=False))
    fig1.add_trace(go.Scatter(x = [happy.Score.median(), happy.Score.median()], y = [0, 25], 
                            mode = 'lines', line = {'color':'green'}, name = 'Mediaan', visible=False))
    
    # Annotatie bij het gemiddelde en mediaan toevoegen
    annotation_gem = [{'x': happy.Score.mean(), 'y':10, 
                      'showarrow': True, 'arrowhead': 4, 'arrowcolor':'black', 
                        'font': {'color': 'black', 'size':15}, 'text': 'Gemiddelde'}]
    annotation_median = [{'x': happy.Score.median(), 'y':10, 'showarrow': True, 'arrowhead': 4,
                        'font': {'color': 'black', 'size':15}, 'text': 'Mediaan'}]
    
    # Button voor het kiezen van de mediaan of het gemiddelde
    buttons = [{'label': "Geen lijnen", 'method': "update", 'args': [{"visible": [True, False, False]}, {"annotations": None}]}, 
    {'label': "Gemiddelde", 'method': "update", 'args': [{"visible": [True, True, False]}, {"annotations": annotation_gem}]},
    {'label': "Mediaan", 'method': "update", 'args': [{"visible": [True, False, True]}, {"annotations": annotation_median}]}
    ]
    
    fig1.update_layout({
        'updatemenus':[{
                'type': "buttons",
                'direction': 'down',
                'x': 1.18,'y': 0.7, 'buttons': buttons
              }]})
    
    # Updaten layout
    fig1.update_layout(
                      title_text = 'Aantal landen per gelukkigheidsscore',
                     )
    fig1.update_xaxes(title_text = 'Gelukkigheidsscore')
    fig1.update_yaxes(title_text = 'Aantal landen')
    
    st.plotly_chart(fig1, use_container_width = True)
    st.write('Hierin is de verdeling van de gelukkigheidsscore van alle landen weergegeven. Dit heeft een normale verdeling.')
    st.write('Via de knoppen aan de rechterzijde zijn de mediaan en het gemiddelde aan te klikken.')
    st.write('')
    st.header('Verbanden met gelukkigheidsscore')

    # Figuur maken
    fig2 = go.Figure()
    
    # Dropdown buttons
    dropdown_buttons = [
        {'label': 'Continent', 'method':'update',
        'args':[{'visible':[True, False, False]}, {'title':'Score per Continent', 'xaxis': {'title': 'Continent'}}]},
        {'label': 'GDP', 'method':'update',
        'args':[{'visible':[False, True, False]}, {'title':'Score per GDP', 'xaxis': {'title': 'GDP per capita', 'categoryorder' :'array', 'categoryarray' : ['0-0,4','0,4-0,8','0,8-1','1-1,2', '1,2-1,4', '1,4+']}}]},
        {'label': 'Bevolkingsdichtheid', 'method':'update',
        'args':[{'visible':[False, False, True]}, {'title':'Score per Bevolkingsdichtheid', 'xaxis': {'title': 'Bevolkingsdichtheid (sq. mile)', 'categoryorder' :'array', 'categoryarray' : ['0-30','30-70','70-120','120-160','160-250','250+']}}]}
    ]
    
    # Traces
    fig2.add_trace(go.Box(x=happy.Continent, y=happy.Score, name = 'Score per Continent'))
    fig2.add_trace(go.Box(x=happy.GDP_group, y=happy.Score, name = 'Score per GDP', visible=False))
    fig2.add_trace(go.Box(x=happy.Pop_density_group, y=happy.Score, name = 'Score per Bevolkingsdichtheid', visible=False))
    
    # Updaten layout
    fig2.update_layout({'updatemenus':[{'type': 'dropdown',
                                     'x':1.3, 'y':0.5,
                                     'showactive':True,
                                     'active': 0,
                                     'buttons':dropdown_buttons}]},
                      title_text = 'Gelukkigheidsscore per Continent, GDP en Bevolkingsdichtheid',
                     )
    fig2.update_xaxes(title_text = 'Continent')
    fig2.update_yaxes(title_text = 'Gelukkigheidsscore')
    
    st.plotly_chart(fig2, use_container_width = True)
    st.write('In deze grafiek zijn drie verschillende boxplots weergegeven waartussen geswitched kan worden via het dropdown-menu.')
    st.write('In de eerste grafiek is per continent de verdeling van de gelukkigheidsscore per land te zien.')
    st.write('In de tweede grafiek is per GDP (Bruto Nationaal Product per inwoner) de verdeling van de gelukkigheidsscore te zien.')
    st.write('In de derde grafiek is per bevolkingsdichtheid in een land de verdeling van de gelukkigheidsscore te zien.')
    st.header('Maken telefoons gelukkig?')
    
    # Figuur maken
    fig3 = go.Figure()
    
    # Button voor het kiezen van de mediaan of het gemiddelde
    dropdown = [ 
    {'label': "Zonder trend", 'method': "update", 'args': [{"visible": [True, False, False]}]},
    {'label': "Met trend", 'method': "update", 'args': [{"visible": [True, True, False]}]},
    {'label': "Uitschieter", 'method': "update", 'args': [{"visible": [True, True, True]}]}   
    ]
    
    # Trace toevoegen
    fig3.add_trace(go.Scatter(x = happy.Phones_per_1000, y = happy.Score, opacity = 0.8, mode = 'markers', name = 'Punten', visible = True))
    fig3.add_trace(go.Scatter(x = prediction_data2["Phones_per_1000"], y = prediction_data2["Score"], mode = "lines",name="Trendlijn", marker_color = "red", visible = False))
    fig3.add_trace(go.Scatter(x = happy_outlier_phone.Phones_per_1000, y = happy_outlier_phone.Score, mode = "markers",name="Outlier", marker_color = "red", visible = False))
    
    # Titel en labels toevoegen. 
    fig3.update_layout({
        'updatemenus':[{
                'type': "dropdown",
                'direction': 'down',
                'x': 1.19,'y': 0.7, 'buttons': dropdown
                }]})
    
    # Updaten layout
    fig3.update_layout(title_text = 'Gelukkigheidsscore per aantal telefoons', xaxis_title = 'Aantal telefoons per 1000', yaxis_title = 'Gelukkigheidsscore')
    
    st.plotly_chart(fig3, use_container_width = True)
    st.write('Hierin is het verband tussen het aantal telefoons (per 1000 inwoners) en de gelukkigheidsscore te zien.')
    st.write('Via het dropdown-menu kan de trendlijn en de uitschieter worden weergegeven.')