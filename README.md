

Space X Falcon 9 First Stage Landing Prediction
Lab 2: Data wrangling
Estimated time needed: 60 minutes

In this lab, we will perform some Exploratory Data Analysis (EDA) to find some patterns in the data and determine what would be the label for training supervised models.

In the data set, there are several different cases where the booster did not land successfully. Sometimes a landing was attempted but failed due to an accident; for example, True Ocean means the mission outcome was successfully landed to a specific region of the ocean while False Ocean means the mission outcome was unsuccessfully landed to a specific region of the ocean. True RTLS means the mission outcome was successfully landed to a ground pad False RTLS means the mission outcome was unsuccessfully landed to a ground pad.True ASDS means the mission outcome was successfully landed on a drone ship False ASDS means the mission outcome was unsuccessfully landed on a drone ship.

In this lab we will mainly convert those outcomes into Training Labels with 1 means the booster successfully landed 0 means it was unsuccessful.

Falcon 9 first stage will land successfully

Several examples of an unsuccessful landing are shown here:

ObjectivesPerform exploratory Data Analysis and determine Training Labels

Exploratory Data Analysis
Determine Training Labels
Import Libraries and Define Auxiliary Functions
We will import the following libraries.

[1]:
import piplite

await piplite.install(['numpy'])

await piplite.install(['pandas'])

[2]:
# Pandas is a software library written for the Python programming language for data manipulation and analysis.

import pandas as pd

#NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays

import numpy as np

Data Analysis
[3]:
from js import fetch

import io



URL = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_1.csv'

resp = await fetch(URL)

dataset_part_1_csv = io.BytesIO((await resp.arrayBuffer()).to_py())

Load Space X dataset, from last section.

[4]:
df=pd.read_csv(dataset_part_1_csv)

df.head(10)

[4]:
FlightNumberDateBoosterVersionPayloadMassOrbitLaunchSiteOutcomeFlightsGridFinsReusedLegsLandingPadBlockReusedCountSerialLongitudeLatitude012010-06-04Falcon 96104.959412LEOCCAFS SLC 40None None1FalseFalseFalseNaN1.00B0003-80.57736628.561857122012-05-22Falcon 9525.000000LEOCCAFS SLC 40None None1FalseFalseFalseNaN1.00B0005-80.57736628.561857232013-03-01Falcon 9677.000000ISSCCAFS SLC 40None None1FalseFalseFalseNaN1.00B0007-80.57736628.561857342013-09-29Falcon 9500.000000POVAFB SLC 4EFalse Ocean1FalseFalseFalseNaN1.00B1003-120.61082934.632093452013-12-03Falcon 93170.000000GTOCCAFS SLC 40None None1FalseFalseFalseNaN1.00B1004-80.57736628.561857562014-01-06Falcon 93325.000000GTOCCAFS SLC 40None None1FalseFalseFalseNaN1.00B1005-80.57736628.561857672014-04-18Falcon 92296.000000ISSCCAFS SLC 40True Ocean1FalseFalseTrueNaN1.00B1006-80.57736628.561857782014-07-14Falcon 91316.000000LEOCCAFS SLC 40True Ocean1FalseFalseTrueNaN1.00B1007-80.57736628.561857892014-08-05Falcon 94535.000000GTOCCAFS SLC 40None None1FalseFalseFalseNaN1.00B1008-80.57736628.5618579102014-09-07Falcon 94428.000000GTOCCAFS SLC 40None None1FalseFalseFalseNaN1.00B1011-80.57736628.561857
Identify and calculate the percentage of the missing values in each attribute

[5]:
df.isnull().sum()/df.shape[0]*100

[5]:
FlightNumber       0.000000
Date               0.000000
BoosterVersion     0.000000
PayloadMass        0.000000
Orbit              0.000000
LaunchSite         0.000000
Outcome            0.000000
Flights            0.000000
GridFins           0.000000
Reused             0.000000
Legs               0.000000
LandingPad        28.888889
Block              0.000000
ReusedCount        0.000000
Serial             0.000000
Longitude          0.000000
Latitude           0.000000
dtype: float64

[6]:
df.dtypes

[6]:
FlightNumber        int64
Date               object
BoosterVersion     object
PayloadMass       float64
Orbit              object
LaunchSite         object
Outcome            object
Flights             int64
GridFins             bool
Reused               bool
Legs                 bool
LandingPad         object
Block             float64
ReusedCount         int64
Serial             object
Longitude         float64
Latitude          float64
dtype: object

TASK 1: Calculate the number of launches on each siteThe data contains several Space X launch facilities: Cape Canaveral Space Launch Complex 40 VAFB SLC 4E , Vandenberg Air Force Base Space Launch Complex 4E (SLC-4E), Kennedy Space Center Launch Complex 39A KSC LC 39A .The location of each Launch Is placed in the column LaunchSite

Next, let's see the number of launches for each site.

Use the method value_counts() on the column LaunchSite to determine the number of launches on each site:

[8]:
# Apply value_counts() on column LaunchSite

df.value_counts('LaunchSite')

[8]:
LaunchSite
CCAFS SLC 40    55
KSC LC 39A      22
VAFB SLC 4E     13
dtype: int64

Each launch aims to an dedicated orbit, and here are some common orbit types:

LEO: Low Earth orbit (LEO)is an Earth-centred orbit with an altitude of 2,000 km (1,200 mi) or less (approximately one-third of the radius of Earth),[1] or with at least 11.25 periods per day (an orbital period of 128 minutes or less) and an eccentricity less than 0.25.[2] Most of the manmade objects in outer space are in LEO [1].

VLEO: Very Low Earth Orbits (VLEO) can be defined as the orbits with a mean altitude below 450 km. Operating in these orbits can provide a number of benefits to Earth observation spacecraft as the spacecraft operates closer to the observation[2].

GTO A geosynchronous orbit is a high Earth orbit that allows satellites to match Earth's rotation. Located at 22,236 miles (35,786 kilometers) above Earth's equator, this position is a valuable spot for monitoring weather, communications and surveillance. Because the satellite orbits at the same speed that the Earth is turning, the satellite seems to stay in place over a single longitude, though it may drift north to south,” NASA wrote on its Earth Observatory website [3] .

SSO (or SO): It is a Sun-synchronous orbit also called a heliosynchronous orbit is a nearly polar orbit around a planet, in which the satellite passes over any given point of the planet's surface at the same local mean solar time [4] .

ES-L1 :At the Lagrange points the gravitational forces of the two large bodies cancel out in such a way that a small object placed in orbit there is in equilibrium relative to the center of mass of the large bodies. L1 is one such point between the sun and the earth [5] .

HEO A highly elliptical orbit, is an elliptic orbit with high eccentricity, usually referring to one around Earth [6].

ISS A modular space station (habitable artificial satellite) in low Earth orbit. It is a multinational collaborative project between five participating space agencies: NASA (United States), Roscosmos (Russia), JAXA (Japan), ESA (Europe), and CSA (Canada) [7]

MEO Geocentric orbits ranging in altitude from 2,000 km (1,200 mi) to just below geosynchronous orbit at 35,786 kilometers (22,236 mi). Also known as an intermediate circular orbit. These are "most commonly at 20,200 kilometers (12,600 mi), or 20,650 kilometers (12,830 mi), with an orbital period of 12 hours [8]

HEO Geocentric orbits above the altitude of geosynchronous orbit (35,786 km or 22,236 mi) [9]

GEO It is a circular geosynchronous orbit 35,786 kilometres (22,236 miles) above Earth's equator and following the direction of Earth's rotation [10]

PO It is one type of satellites in which a satellite passes above or nearly above both poles of the body being orbited (usually a planet such as the Earth [11]

some are shown in the following plot:

TASK 2: Calculate the number and occurrence of each orbit
Use the method .value_counts() to determine the number and occurrence of each orbit in the column Orbit

[9]:
# Apply value_counts on Orbit column

df.value_counts('Orbit')

[9]:
Orbit
GTO      27
ISS      21
VLEO     14
PO        9
LEO       7
SSO       5
MEO       3
ES-L1     1
GEO       1
HEO       1
SO        1
dtype: int64

TASK 3: Calculate the number and occurence of mission outcome per orbit type
Use the method .value_counts() on the column Outcome to determine the number of landing_outcomes.Then assign it to a variable landing_outcomes.

[12]:
# landing_outcomes = values on Outcome column

landing_outcomes = df.value_counts('Outcome')

True Ocean means the mission outcome was successfully landed to a specific region of the ocean while False Ocean means the mission outcome was unsuccessfully landed to a specific region of the ocean. True RTLS means the mission outcome was successfully landed to a ground pad False RTLS means the mission outcome was unsuccessfully landed to a ground pad.True ASDS means the mission outcome was successfully landed to a drone ship False ASDS means the mission outcome was unsuccessfully landed to a drone ship. None ASDS and None None these represent a failure to land.

[13]:
for i,outcome in enumerate(landing_outcomes.keys()):

    print(i,outcome)

0 True ASDS
1 None None
2 True RTLS
3 False ASDS
4 True Ocean
5 False Ocean
6 None ASDS
7 False RTLS


We create a set of outcomes where the second stage did not land successfully:

[14]:
bad_outcomes=set(landing_outcomes.keys()[[1,3,5,6,7]])

bad_outcomes

[14]:
{'False ASDS', 'False Ocean', 'False RTLS', 'None ASDS', 'None None'}

TASK 4: Create a landing outcome label from Outcome column
Using the Outcome, create a list where the element is zero if the corresponding row in Outcome is in the set bad_outcome; otherwise, it's one. Then assign it to the variable landing_class:

[20]:
# landing_class = 0 if bad_outcome

# landing_class = 1 otherwise

bad_outcome = True

landing_class = 0 if bad_outcome else 1

print(landing_class)

0


This variable will represent the classification variable that represents the outcome of each launch. If the value is zero, the first stage did not land successfully; one means the first stage landed Successfully

[21]:
df['Class']=landing_class

df[['Class']].head(8)

[21]:
Class0010203040506070
[22]:
df.head(5)

[22]:
FlightNumberDateBoosterVersionPayloadMassOrbitLaunchSiteOutcomeFlightsGridFinsReusedLegsLandingPadBlockReusedCountSerialLongitudeLatitudeClass012010-06-04Falcon 96104.959412LEOCCAFS SLC 40None None1FalseFalseFalseNaN1.00B0003-80.57736628.5618570122012-05-22Falcon 9525.000000LEOCCAFS SLC 40None None1FalseFalseFalseNaN1.00B0005-80.57736628.5618570232013-03-01Falcon 9677.000000ISSCCAFS SLC 40None None1FalseFalseFalseNaN1.00B0007-80.57736628.5618570342013-09-29Falcon 9500.000000POVAFB SLC 4EFalse Ocean1FalseFalseFalseNaN1.00B1003-120.61082934.6320930452013-12-03Falcon 93170.000000GTOCCAFS SLC 40None None1FalseFalseFalseNaN1.00B1004-80.57736628.5618570
We can use the following line of code to determine the success rate:

[23]:
df["Class"].mean()

[23]:
0.0

We can now export it to a CSV for the next section,but to make the answers consistent, in the next lab we will provide data in a pre-selected date range.

df.to_csv("dataset_part_2.csv", index=False)

Authors
Pratiksha Verma

Change Log
Date (YYYY-MM-DD)VersionChanged ByChange Description2022-11-091.0Pratiksha VermaConverted initial version to Jupyterlite
IBM Corporation 2022. All rights reserved.
