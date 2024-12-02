#REQUIREMENTS
# skyfield
# numpy
# pandas
# python-terrier

from datetime import datetime, timedelta
from skyfield.api import load
import pandas as pd
import numpy as np
import re
import pyterrier as pt
import subprocess


#Time Converter

def convert_two_digit_year_to_four_digit(two_digit_year):
    current_year = datetime.now().year
    current_century = current_year // 100 * 100

    if two_digit_year <= current_year % 100:
        # Assume it belongs to the 21st century
        return current_century + two_digit_year
    else:
        # Assume it belongs to the 20th century
        return current_century - 100 + two_digit_year

def timeconverter(tle_epoch):

    # Extract the year and day from the integer part
    year = convert_two_digit_year_to_four_digit(int(tle_epoch / 1000))  # Assuming the year is in the thousands
    day_of_year = int(tle_epoch % 1000)

    # Convert the year and day of the year to a datetime object
    epoch_datetime = datetime(year, 1, 1) + timedelta(days=day_of_year - 1)

    # Extract the time from the fractional part
    fractional_part = tle_epoch % 1
    seconds_in_a_day = 24 * 60 * 60
    time_seconds = int(fractional_part * seconds_in_a_day)
    time_timedelta = timedelta(seconds=time_seconds)

    # Add the time to the datetime object
    result_datetime = epoch_datetime + time_timedelta
    # Parse the datetime string into a datetime object
    #datetime_obj = datetime.strptime(result_datetime, "%Y-%m-%d %H:%M:%S.%f")

    formatted_date = epoch_datetime.strftime("%Y-%m-%d")

    return formatted_date

def timeconverter2(datetime_obj):


    formatted_date = datetime_obj.strftime("%Y-%m-%d")

    return formatted_date
    
#Open the text document with TLE data

with open('Satellites15.txt',"r") as f:
    lines = f.readlines()

#lines = response.text.strip().split('\r\n')

# Extract the TLE data from the lines and store them in an array
tle_data = []
for i in range(0, len(lines), 3):
    name = lines[i]
    line2 = lines[i+1]
    line3 = lines[i+2]
    tle_data.append([name, line2, line3])
    
    
# Data Structure
# Fengyun-2D
# 1 29640U 06053A   23193.42992718 -.00000210  00000-0  00000+0 0  9998
# 2 29640   8.8955  48.2577 0012379  22.2917 184.8793  0.98195112 60480

satellites = []
satellites = tle_data


# Extract various information for each satellite

names = []
norad_ids = []
international_designator = []
BSTAR_drag_term = []
checksum_line1 = []
checksum_line2 = []
element_set_type = []
element_number = []
right_ascension = []
revolution_number_at_epoch = []
epochs = []
inclinations = []
eccentricities = []
semi_major_axes = []
mean_motion = []
for satellite in satellites:
    # Extract the name and Norad ID
    name = satellite[0]
    norad_id = int(satellite[2][2:7])
    names.append(name)
    norad_ids.append(norad_id)

    # Extract the epoch time
    epoch = timeconverter(float(satellite[1][18:32]))
    epochs.append(epoch)

    # Extract the inclination
    inclination = float(satellite[2][8:16])
    inclinations.append(inclination)

    # Extract the eccentricity
    eccentricity = float(satellite[2][26:33])
    eccentricities.append(eccentricity)

    # Extract the mean motion
    mean_motion_angular = float(satellite[2][52:62])
    mean_motion.append(mean_motion_angular)


df = pd.DataFrame({'Name': names,
                  'Norad ID': norad_ids,
                  'Epoch': epochs,
                  'Inclination': inclinations,
                  'Eccentricity': eccentricities,
                  'Mean Motion': mean_motion})

df = df.replace(r'\n','', regex=True)

#Open the csv with manoeuvre data

df_sample_data = pd.read_csv("sample_output.csv")

#Sample

# date	NORAD ID	satellite name	anomaly confidence level
# 13/3/2024 13:18	26997	Jason-1	low
# 13/3/2024 12:42	33105	Jason-2	medium
# 13/3/2024 12:10	41335	Sentinel-3A	low
# 13/3/2024 5:18	22076	TOPEX	medium

data = df.to_dict('records')

df_sample_data['date'] = pd.to_datetime(df_sample_data['date'])

strs1 = ["" for x in range(len(data)+1)]
strs2 = ["" for x in range(len(data)+1)]

units = ["", "", "",	"degrees", "", "revolutions per day"]
# Create a paragraph from the dictionary data
i = 0
while i < len(data):
    paragraph = ""
    key1 = data[i]['Name']
    key2 = data[i]['Norad ID']
      #paragraph += f"Name is {key1}. "
    j=0
    for key, value in data[i].items():
      paragraph += f"{key} of {key1} is {value} {units[j]}. "
      j = j+1
        #paragraph += f"{key} of {key1} is {value}. "
    k = i
    
    df_temp = df_sample_data.loc[df_sample_data['satellite name'] == key1]
    df_temp = df_temp.drop(columns = ['satellite name'])
    df_temp =df_temp.sort_values(by = ['date'])
    data_temp = df_temp.to_dict('records')

    #Manoeuvre data summary

    paragraph += f"{key1} (Norad ID {key2}) had {len(data_temp)} manoeuvres. "

    #How many manoeuvres last week

    today = datetime.today()
    start_of_week = today - timedelta(days=today.weekday())
    paragraph += f"{key1} had "#{len(data_temp['date'] <= today)} manoeuvres. "

    

    #first and last manoeuvre
    df_temp

    paragraph += f"The first manoeuvre of {key1} (Norad ID {key2}) occured on {(df_temp['date'].min())}, and the confidence level is {df_temp[df_temp['date'] == df_temp['date'].min()]['anomaly confidence level'].iloc[0]}. "

    paragraph += f"The latest manoeuvre of {key1} (Norad ID {key2}) occured on {(df_temp['date'].max())}, and the confidence level is {df_temp[df_temp['date'] == df_temp['date'].max()]['anomaly confidence level'].iloc[0]} confidence. "


    for year in range(df_temp['date'].dt.year.min(), df_temp['date'].dt.year.max()):
        paragraph += f"{key1} (Norad ID {key2}) had {(df_temp['date'].dt.year == year).sum()} manoeuvres in {year}. "



    #Manoeuvre data
    for j in range(round(len(data_temp))):
        if j == 0:
            paragraph += f"{key1} (Norad ID {key2}) manoeuvred on {data_temp[j]['date']} and the confidence level is {data_temp[j]['anomaly confidence level']}. "
        elif j == 1:
            paragraph += f"{key1} (Norad ID {key2}) manoeuvred on {data_temp[j]['date']} and the confidence level is {data_temp[j]['anomaly confidence level']}. "
        else:
            paragraph += f"{key1} (Norad ID {key2}) manoeuvred on {data_temp[j]['date']} and the confidence level is {data_temp[j]['anomaly confidence level']} . "
       #paragraph += f"and it is a {data_temp[j]['anomaly type']} with {data_temp[j]['anomaly type confidence level']} confidence. "
    for year in range(df_temp['date'].dt.year.min(), df_temp['date'].dt.year.max()):
        paragraph += f"{key1} (Norad ID {key2}) had {(df_temp['date'].dt.year == year).sum()} manoeuvres in {year} "

    

    strs1[k] = paragraph
    strs2[k] = f"{key1}"
    i = i+1

df_temp =df_sample_data.sort_values(by = ['date'])
df_temp.reset_index(inplace=True)
paragraph = ""


today1 = datetime.now().date()
today = datetime.now().date().strftime("%Y-%m-%d")


last_week_start = (today1 - timedelta(days=7)).strftime("%Y-%m-%d")

last_2week_start = (today1 - timedelta(days=14)).strftime("%Y-%m-%d")
#last_week_end = last_week_start + timedelta(days=6)

last_month_end = datetime(today1.year, today1.month, 1) - timedelta(days=1)
last_month_start = datetime(last_month_end.year, last_month_end.month, 1)

if today1.month < 6:
    last_six_months_start = datetime(today1.year - 1, today1.month + 6, 1)
else:
    last_six_months_start = datetime(today1.year, today1.month - 5, 1)
last_six_months_end = datetime(today1.year, today1.month, 1) - timedelta(days=1)

current_year_start = datetime(today1.year, 1, 1)
current_year_end = today

last_year_end = datetime(today1.year - 1, 12, 31)
last_year_start = datetime(today1.year - 1, 1, 1)

last_two_years_end = last_year_end
last_two_years_start = datetime(today1.year - 2, 1, 1)

paragraph += f"The most recent or the latest manoeuvre of all {k+1} satellites occurred from {df_temp['satellite name'][len(df_temp)-1]} and it was on {df_temp['date'][len(df_temp)-1]}. "
paragraph += f"There were {df_sample_data[df_sample_data['date']==today]['date'].count()} manoeuvres today. "
satellite_names_list1 = df_sample_data[df_sample_data['date']==today]['satellite name']
satellite_names_list1 = satellite_names_list1.unique()
satellite_names1 = ', '.join(satellite_names_list1)
if len(satellite_names_list1) > 0:
    paragraph += f" and the manoeuvred satellites were {satellite_names1}. "
else:
    paragraph += ". "

paragraph += f"There were {df_sample_data[(df_sample_data['date']>=last_week_start) & (df_sample_data['date']<=today)]['date'].count()} manoeuvres last week"
satellite_names_list2 = df_sample_data[(df_sample_data['date'] >= last_week_start) & (df_sample_data['date'] <= today)]['satellite name']
satellite_names_list2 = satellite_names_list2.unique()
satellite_names2 = ', '.join(satellite_names_list2)
if len(satellite_names_list2) > 0:
    paragraph += f" and the manoeuvred satellites were {satellite_names2}. "
else:
    paragraph += ". "


paragraph += f"There were {df_sample_data[(df_sample_data['date']>=last_2week_start) & (df_sample_data['date']<=today)]['date'].count()} manoeuvres last two week"
satellite_names_list = df_sample_data[(df_sample_data['date'] >= last_2week_start) & (df_sample_data['date'] <= today)]['satellite name']
satellite_names_list = satellite_names_list.unique()
satellite_names = ', '.join(satellite_names_list)
if len(satellite_names_list) > 0:
    paragraph += f" and the manoeuvred satellites were {satellite_names}. "
else:
    paragraph += ". "


paragraph += f"There were {df_sample_data[(df_sample_data['date']>=last_month_start) & (df_sample_data['date']<=today)]['date'].count()} manoeuvres last month"
satellite_names_list3 = df_sample_data[(df_sample_data['date'] >= last_month_start) & (df_sample_data['date'] <= today)]['satellite name']
satellite_names_list3 = satellite_names_list3.unique()
satellite_names3 = ', '.join(satellite_names_list3)
if len(satellite_names_list3) > 0:
    paragraph += f" and the manoeuvred satellites were {satellite_names3}. "
else:
    paragraph += ". "


paragraph += f"There were {df_sample_data[(df_sample_data['date']>=last_six_months_start) & (df_sample_data['date']<=today)]['date'].count()} manoeuvres last six months"
satellite_names_list4 = df_sample_data[(df_sample_data['date'] >= last_six_months_start) & (df_sample_data['date'] <= today)]['satellite name']
satellite_names_list4 = satellite_names_list4.unique()
satellite_names4 = ', '.join(satellite_names_list4)
if len(satellite_names_list4) > 0:
    paragraph += f" and the manoeuvred satellites were {satellite_names4}. "
else:
    paragraph += ". "


paragraph += f"There were {df_sample_data[(df_sample_data['date']>=current_year_start) & (df_sample_data['date']<=today)]['date'].count()} manoeuvres current (this) year"
satellite_names_list5 = df_sample_data[(df_sample_data['date'] >= current_year_start) & (df_sample_data['date'] <= today)]['satellite name']
satellite_names_list5 = satellite_names_list5.unique()
satellite_names5 = ', '.join(satellite_names_list5)
if len(satellite_names_list5) > 0:
    paragraph += f" and the manoeuvred satellites were {satellite_names5}. "
else:
    paragraph += ". "


paragraph += f"There were {df_sample_data[(df_sample_data['date']>=last_year_start) & (df_sample_data['date']<=today)]['date'].count()} manoeuvres last year"
satellite_names_list6 = df_sample_data[(df_sample_data['date'] >= last_year_start) & (df_sample_data['date'] <= today)]['satellite name']
satellite_names_list6 = satellite_names_list6.unique()
satellite_names6 = ', '.join(satellite_names_list6)
if len(satellite_names_list6) > 0:
    paragraph += f" and the manoeuvred satellites were {satellite_names6}. "
else:
    paragraph += ". "


paragraph += f"There were {df_sample_data[(df_sample_data['date']>=last_two_years_start) & (df_sample_data['date']<=today)]['date'].count()} manoeuvres last two years"
satellite_names_list8 = df_sample_data[(df_sample_data['date'] >= last_two_years_start) & (df_sample_data['date'] <= today)]['satellite name']
satellite_names_list8 = satellite_names_list8.unique()
satellite_names8 = ', '.join(satellite_names_list8)
if len(satellite_names_list8) > 0:
    paragraph += f" and the manoeuvred satellites were {satellite_names8}. "
else:
    paragraph += ". "




lst = df_sample_data['date'].dt.year.unique()
lst.sort()
for i in lst:
    paragraph += f"There were {df_sample_data[df_sample_data['date'].dt.year ==i]['date'].count()} manoeuvres in {i}. "
i=0
while i < len(data):
    key1 = data[i]['Name']
    key2 = data[i]['Norad ID']

    df_temp = df_sample_data.loc[df_sample_data['satellite name'] == key1]
    df_temp = df_temp.drop(columns = ['satellite name'])
    df_temp =df_temp.sort_values(by = ['date'])
    data_temp = df_temp.to_dict('records')

    #Manoeuvre data summary

    paragraph += f"{key1} had {len(data_temp)} manoeuvres. "

    #How many manoeuvres last week

    today = datetime.today()
    start_of_week = today - timedelta(days=today.weekday())
    paragraph += f"{key1} had "#{len(data_temp['date'] <= today)} manoeuvres. "

    
    #first and last manoeuvre

    paragraph += f"{key1} (Norad ID {key2}) manoeuvred on {(df_temp['date'].min())}, and the confidence level is {df_temp[df_temp['date'] == df_temp['date'].min()]['anomaly confidence level'].iloc[0]}. "

    paragraph += f"{key1} (Norad ID {key2}) manoeuvred on {(df_temp['date'].max())}, and it is a {df_temp[df_temp['date'] == df_temp['date'].max()]['anomaly confidence level'].iloc[0]}. "

    for year in range(df_temp['date'].dt.year.min(), df_temp['date'].dt.year.max()):
        paragraph += f"{key1} (Norad ID {key2}) had {(df_temp['date'].dt.year == year).sum()} manoeuvres in {year} "

    i = i+1

strs1[k+1] = paragraph
strs2[k+1] = "Summary"

seq = list(range(len(strs1)))

df1 = pd.DataFrame(zip(seq,strs1,strs2), columns = ['docno','text', 'title'])

subprocess.run(['rm', '-rf', './pd_index'])


# Convert the "text" column to strings
df1["text"] = df1["text"].astype(str)
df1["docno"] = df1["docno"].astype(str)

# Initialize PyTerrier
if not pt.started():
    pt.init()

# Create the DataFrame Indexer
pd_indexer = pt.DFIndexer(index_path="./pd_index")

# Index the "text" column
indexref = pd_indexer.index(df1["text"], df1["docno"])

# Create an index using the index reference
index = pt.IndexFactory.of(indexref)

index = pt.IndexFactory.of('./pd_index/data.properties')

pd_indexer.setProperty("tokeniser", "UTFTokeniser")
pd_indexer.setProperty( "termpipelines", "Stopwords,PorterStemmer")

data1 = []
for row in df1.itertuples():
    data1.append( row.text)
iter_indexer = pt.IterDictIndexer("./pd_index", meta={'docno': 20, 'title': 10000, 'body':100000},
overwrite=True)

n = 0
Satellite_names = [
    "Fengyun-2D",
    "Fengyun-2E",
    "Fengyun-2F",
    "Fengyun-2H",
    "Fengyun-4A",
    "Sentinel-3A",
    "Sentinel-3B",
    "Sentinel-6A",
    "Jason-1",
    "Jason-2",
    "Jason-3",
    "SARAL",
    "CryoSat-2",
    "Haiyang-2A",
    "TOPEX",
    "Summary",
]
for row_data in data1:
    with open(f"{Satellite_names[n]}.txt", "a") as file:
        file.write(row_data)
    n = n+1

print("Data saved to text files successfully.")
